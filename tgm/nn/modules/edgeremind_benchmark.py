from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import resource as _resource
except ImportError:
    _resource = None

from .edgeremind import (
    BANK_SCOPES,
    NUM_FEATURES,
    EdgeReMINDPredictor,
    EdgeReMINDTrainer,
)
from .edgeremind_index import EdgeReMINDIndex, extract_training_features


def _rss_mb() -> float:
    if _resource is None:
        return 0.0
    try:
        ru = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        if ru > 1 << 30:
            return ru / (1024.0 * 1024.0)
        return ru / 1024.0
    except Exception:
        return 0.0


def _heartbeat(tag: str) -> None:
    now = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        f'[heartbeat {now} pid={_os.getpid()} rss={_rss_mb():.0f}MiB] {tag}',
        flush=True,
    )


def _expand_mask(base_mask: np.ndarray, feat_dim: int) -> np.ndarray:
    q = (feat_dim - NUM_FEATURES) // len(BANK_SCOPES)
    return np.concatenate([
        base_mask,
        np.repeat(base_mask[3], q),
        np.repeat(base_mask[4], q),
        np.repeat(base_mask[5], q),
    ]).astype(np.float32)


def _smooth_val_curve(mrrs: List[float], window: int = 3) -> List[float]:
    if window <= 1:
        return list(mrrs)
    n = len(mrrs)
    half = window // 2
    out: List[float] = []
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out.append(float(np.mean(mrrs[a:b])))
    return out


def build_full_index(
    train_mat, val_mat, test_mat,
    decay_rate: float,
    decay_rate_d: Optional[float] = None,
    bank_lambdas: Optional[np.ndarray] = None,
) -> Tuple[EdgeReMINDIndex, int, int]:
    def arrs(mat):
        return (
            mat.edge_src.cpu().numpy().astype(np.int64),
            mat.edge_dst.cpu().numpy().astype(np.int64),
            mat.edge_time.cpu().numpy().astype(np.int64),
            mat.edge_type.cpu().numpy().astype(np.int64),
        )

    ts_s, td_s, tt_s, tr_s = arrs(train_mat)
    vs_s, vd_s, vt_s, vr_s = arrs(val_mat)
    es_s, ed_s, et_s, er_s = arrs(test_mat)
    src = np.concatenate([ts_s, vs_s, es_s])
    dst = np.concatenate([td_s, vd_s, ed_s])
    ts = np.concatenate([tt_s, vt_s, et_s])
    rel = np.concatenate([tr_s, vr_s, er_s])
    index = EdgeReMINDIndex.build(
        src, dst, ts, rel,
        decay_rate_srd=decay_rate,
        decay_rate_d=decay_rate_d,
        bank_lambdas=bank_lambdas,
    )
    return index, len(ts_s), len(vs_s)


def train_parallel(
    model: EdgeReMINDPredictor,
    trainer: EdgeReMINDTrainer,
    index: EdgeReMINDIndex,
    train_mat,
    num_relations: int,
    first_dst: int,
    last_dst: int,
    num_workers: int,
    epochs: int,
    lr: float,
    opt_bsize: int,
    patience: int = 0,
    min_delta: float = 1e-3,
) -> None:
    feats, rels = extract_training_features(
        index,
        train_mat.edge_src.cpu().numpy(),
        train_mat.edge_dst.cpu().numpy(),
        train_mat.edge_time.cpu().numpy(),
        train_mat.edge_type.cpu().numpy(),
        num_relations=num_relations,
        first_dst=first_dst,
        last_dst=last_dst,
        k=trainer.k,
        feature_bsize=trainer.feature_bsize,
        max_train_queries=trainer.max_train_queries,
        num_workers=num_workers,
        seed=trainer.seed,
    )
    trainer.optimize(
        feats,
        rels,
        epochs=epochs,
        lr=lr,
        opt_bsize=opt_bsize,
        patience=patience,
        min_delta=min_delta,
    )


def train_learned(
    model: EdgeReMINDPredictor,
    trainer: EdgeReMINDTrainer,
    index: EdgeReMINDIndex,
    train_mat,
    val_loader,
    num_relations: int,
    first_dst: int,
    last_dst: int,
    num_workers: int,
    epochs: int,
    lr: float,
    opt_bsize: int,
    n_train: int,
    bsize: int,
    patience: int = 0,
    min_delta: float = 1e-3,
    shrinkage: float = 0.0,
    smooth_window: int = 3,
    val_curve_path: Optional[str] = None,
    num_features: int = NUM_FEATURES,
    feature_mask: Optional[Sequence[int]] = None,
    bank_scope_mask: Optional[Sequence[int]] = None,
) -> Tuple[str, float]:
    if not 1 <= num_features <= NUM_FEATURES:
        raise ValueError(
            f'num_features must be in [1, {NUM_FEATURES}], got {num_features}'
        )

    mask_arr = None
    if feature_mask is not None:
        mask_arr = np.asarray(list(feature_mask), dtype=np.float32)
        if mask_arr.shape != (NUM_FEATURES,):
            raise ValueError(
                f'feature_mask must have length {NUM_FEATURES}, '
                f'got {mask_arr.shape}'
            )
        if not np.all(np.isin(mask_arr, (0.0, 1.0))):
            raise ValueError('feature_mask entries must all be 0 or 1')
        if mask_arr.sum() == 0:
            raise ValueError('feature_mask must keep at least one feature')

    _heartbeat('train_learned: start feature extraction')
    feats, rels = extract_training_features(
        index,
        train_mat.edge_src.cpu().numpy(),
        train_mat.edge_dst.cpu().numpy(),
        train_mat.edge_time.cpu().numpy(),
        train_mat.edge_type.cpu().numpy(),
        num_relations=num_relations,
        first_dst=first_dst,
        last_dst=last_dst,
        k=trainer.k,
        feature_bsize=trainer.feature_bsize,
        max_train_queries=trainer.max_train_queries,
        num_workers=num_workers,
        seed=trainer.seed,
    )
    feat_dim = feats.shape[-1]
    if num_features < NUM_FEATURES:
        feats[:, :, num_features:] = 0.0
        print(
            f'[num-features] training with {num_features} features '
            f'(columns {num_features}..{feat_dim - 1} zeroed at extraction)',
            flush=True,
        )
    mask_ext = None
    if mask_arr is not None:
        mask_ext = _expand_mask(mask_arr, feat_dim)
        dropped = [i for i, m in enumerate(mask_ext) if m == 0.0]
        feats[:, :, dropped] = 0.0
        print(
            f'[feature-mask] training with base mask '
            f'{"".join(str(int(m)) for m in mask_arr)} '
            f'(expanded columns {dropped} zeroed at extraction and eval)',
            flush=True,
        )

    bank_drop_cols = []
    if bank_scope_mask is not None:
        bsm = [int(b) for b in bank_scope_mask]
        if len(bsm) != len(BANK_SCOPES):
            raise ValueError(
                f'bank_scope_mask must have length {len(BANK_SCOPES)}, got {len(bsm)}')
        if feat_dim <= NUM_FEATURES:
            raise ValueError('bank_scope_mask requires the bank to be enabled')
        q = (feat_dim - NUM_FEATURES) // len(BANK_SCOPES)
        for si, keep in enumerate(bsm):
            if not keep:
                c0 = NUM_FEATURES + si * q
                bank_drop_cols.extend(range(c0, c0 + q))
        if bank_drop_cols:
            feats[:, :, bank_drop_cols] = 0.0
            print(
                f'[bank-scope-mask] dropping bank scopes '
                f'{[BANK_SCOPES[i] for i, k in enumerate(bsm) if not k]} '
                f'(bank columns {bank_drop_cols} zeroed at extraction and eval; '
                f'base features retained)',
                flush=True,
            )
    _heartbeat('train_learned: features extracted, fitting theta')
    theta_prior = model.theta.detach().clone() if shrinkage > 0.0 else None

    torch.manual_seed(trainer.seed)
    _, snapshots = trainer.optimize(
        feats,
        rels,
        epochs=epochs,
        lr=lr,
        opt_bsize=opt_bsize,
        patience=patience,
        min_delta=min_delta,
        shrinkage=shrinkage,
        theta_prior=theta_prior,
        return_snapshots=True,
    )
    _heartbeat(f'train_learned: theta fitted, scoring {len(snapshots)} snapshots')

    candidates = [
        s.detach().cpu().numpy().astype(np.float32) for s in snapshots
    ]
    labels = [f'e{i + 1}' for i in range(len(candidates))]

    mrrs = evaluate_parallel_multi(
        val_loader, index, candidates, n_train, bsize, num_workers
    )
    raw_mrrs = [float(m) for m in mrrs]
    smoothed = _smooth_val_curve(raw_mrrs, window=smooth_window)
    best_idx = int(np.argmax(smoothed))

    best_theta = candidates[best_idx].copy()
    if mask_ext is not None:
        dropped = [i for i, m in enumerate(mask_ext) if m == 0.0]
        best_theta[:, dropped] = 0.0
    if bank_drop_cols:
        best_theta[:, bank_drop_cols] = 0.0

    model.theta = torch.tensor(
        best_theta, dtype=model.theta.dtype, device=model.device
    )

    if val_curve_path is not None:
        try:
            with open(val_curve_path, 'w') as f:
                _json.dump({
                    'labels': labels,
                    'raw_val_mrr': raw_mrrs,
                    'smoothed_val_mrr': smoothed,
                    'smooth_window': smooth_window,
                    'best_idx': best_idx,
                    'best_label': labels[best_idx],
                    'best_raw_val_mrr': raw_mrrs[best_idx],
                    'best_smoothed_val_mrr': smoothed[best_idx],
                }, f, indent=2)
            print(f'[val-curve] wrote {val_curve_path}', flush=True)
        except OSError as e:
            print(f'[val-curve] WARN: could not write {val_curve_path}: {e}',
                  flush=True)

    smoothing_tag = f'  smoothed-w{smooth_window}' if smooth_window > 1 else ''
    print(f'[best-epoch] per-epoch val MRR{smoothing_tag}:', flush=True)
    for lbl, raw, sm in zip(labels, raw_mrrs, smoothed):
        marker = '  <-- chosen' if lbl == labels[best_idx] else ''
        sm_field = f'  sm={sm:.6f}' if smooth_window > 1 else ''
        print(f'    {lbl}: {raw:.6f}{sm_field}{marker}', flush=True)
    print(
        f'[best-epoch] picked {labels[best_idx]}  '
        f'val={raw_mrrs[best_idx]:.6f}  '
        f'(smoothed={smoothed[best_idx]:.6f}, window={smooth_window})',
        flush=True,
    )
    _heartbeat('train_learned: done')
    return labels[best_idx], float(raw_mrrs[best_idx])


def evaluate_parallel(
    loader,
    index: EdgeReMINDIndex,
    theta: np.ndarray,
    split_offset: int,
    bsize: int,
    num_workers: int,
) -> float:
    import multiprocessing as mp
    import sys
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    from tgm.nn.modules.edgeremind_index import _eval_chunk, _set_worker_globals

    theta = np.asarray(theta, dtype=np.float32)
    if sys.platform == 'darwin' or sys.platform == 'win32':
        ctx = mp.get_context('spawn')
    else:
        ctx = mp.get_context('fork')
        _set_worker_globals(index, theta)

    state = {'seen': 0}

    def make_task(batch):
        ne = batch.edge_src.numel()
        cutoff = split_offset + state['seen']
        src = batch.edge_src.cpu().numpy().astype(np.int64)
        rel = batch.edge_type.cpu().numpy().astype(np.int64)
        ts = batch.edge_time.cpu().numpy().astype(np.int64)
        dst = batch.edge_dst.cpu().numpy().astype(np.int64)
        cut = np.full(ne, cutoff, dtype=np.int64)
        negs = [
            batch.neg_batch_list[i].cpu().numpy().astype(np.int64) for i in range(ne)
        ]
        state['seen'] += ne
        return (0, src, rel, ts, cut, dst, negs)

    rr_sum = 0.0
    n_q = 0

    if num_workers and num_workers > 0:
        max_inflight = max(2, num_workers * 2)
        with ProcessPoolExecutor(
            max_workers=num_workers, 
            mp_context=ctx,
            initializer=_set_worker_globals,
            initargs=(index, theta)
        ) as ex:
            futures: deque = deque()
            for batch in tqdm(loader):
                if batch.edge_src.numel() == 0:
                    continue
                futures.append(ex.submit(_eval_chunk, make_task(batch)))
                if len(futures) >= max_inflight:
                    _, rr = futures.popleft().result()
                    rr_sum += float(rr.sum())
                    n_q += rr.shape[0]
            while futures:
                _, rr = futures.popleft().result()
                rr_sum += float(rr.sum())
                n_q += rr.shape[0]
    else:
        for batch in tqdm(loader):
            if batch.edge_src.numel() == 0:
                continue
            _, rr = _eval_chunk(make_task(batch))
            rr_sum += float(rr.sum())
            n_q += rr.shape[0]

    return rr_sum / max(n_q, 1)


def evaluate_parallel_multi(
    loader,
    index: EdgeReMINDIndex,
    thetas: List[np.ndarray],
    split_offset: int,
    bsize: int,
    num_workers: int,
) -> List[float]:
    import multiprocessing as mp
    import sys
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    from tgm.nn.modules.edgeremind_index import _eval_chunk_multi, _set_worker_globals

    thetas = [np.asarray(t, dtype=np.float32) for t in thetas]
    if sys.platform == 'darwin' or sys.platform == 'win32':
        ctx = mp.get_context('spawn')
    else:
        ctx = mp.get_context('fork')
        _set_worker_globals(index, None, thetas)
    m = len(thetas)
    state = {'seen': 0}

    def make_task(batch):
        ne = batch.edge_src.numel()
        cutoff = split_offset + state['seen']
        src = batch.edge_src.cpu().numpy().astype(np.int64)
        rel = batch.edge_type.cpu().numpy().astype(np.int64)
        ts = batch.edge_time.cpu().numpy().astype(np.int64)
        dst = batch.edge_dst.cpu().numpy().astype(np.int64)
        cut = np.full(ne, cutoff, dtype=np.int64)
        negs = [
            batch.neg_batch_list[i].cpu().numpy().astype(np.int64) for i in range(ne)
        ]
        state['seen'] += ne
        return (0, src, rel, ts, cut, dst, negs)

    rr_sum = np.zeros(m, dtype=np.float64)
    n_q = 0

    if num_workers and num_workers > 0:
        max_inflight = max(2, num_workers * 2)
        with ProcessPoolExecutor(
            max_workers=num_workers, 
            mp_context=ctx,
            initializer=_set_worker_globals,
            initargs=(index, None, thetas)
        ) as ex:
            futures: deque = deque()
            for batch in tqdm(loader):
                if batch.edge_src.numel() == 0:
                    continue
                futures.append(ex.submit(_eval_chunk_multi, make_task(batch)))
                if len(futures) >= max_inflight:
                    _, rr = futures.popleft().result()
                    rr_sum += rr.sum(axis=1)
                    n_q += rr.shape[1]
            while futures:
                _, rr = futures.popleft().result()
                rr_sum += rr.sum(axis=1)
                n_q += rr.shape[1]
    else:
        for batch in tqdm(loader):
            if batch.edge_src.numel() == 0:
                continue
            _, rr = _eval_chunk_multi(make_task(batch))
            rr_sum += rr.sum(axis=1)
            n_q += rr.shape[1]

    return [float(s / max(n_q, 1)) for s in rr_sum]
