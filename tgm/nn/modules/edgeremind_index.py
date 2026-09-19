from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import multiprocessing as mp

import numpy as np
import torch

from tgm.core import DGBatch, DGraph
from tgm.nn.modules.edgeremind import (
    BANK_SCOPES,
    DEFAULT_WEIGHTS,
    NUM_FEATURES,
)
from tgm.util.logging import _get_logger

logger = _get_logger(__name__)

_WHOLE_GROUP_FACTOR = 64

def calibrate_bank_lambdas(
    src: np.ndarray,
    dst: np.ndarray,
    ts: np.ndarray,
    rel: np.ndarray,
    verbose: bool = True,
    n_timescales: int = 3,
    geometric_factor: float = 2.0,
) -> np.ndarray:
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    ts = np.asarray(ts, dtype=np.int64)
    rel = np.asarray(rel, dtype=np.int64)
    n = ts.shape[0]
    pos = np.arange(n, dtype=np.int64)
    if n_timescales < 1:
        raise ValueError(f'n_timescales must be >= 1, got {n_timescales}')
    if geometric_factor <= 0:
        raise ValueError(
            f'geometric_factor must be > 0, got {geometric_factor}')
    m = int(n_timescales)
    geo_exponents = np.arange(m, dtype=np.float64) - (m - 1) / 2.0

    def scope_gaps(key_cols: Tuple[np.ndarray, ...]) -> np.ndarray:
        if n < 2:
            return np.empty(0, dtype=np.int64)
        order = np.lexsort((pos,) + tuple(reversed(key_cols)))
        same = np.ones(n - 1, dtype=bool)
        for col in key_cols:
            cs = col[order]
            same &= cs[1:] == cs[:-1]
        ts_s = ts[order]
        gaps = (ts_s[1:] - ts_s[:-1])[same]
        return gaps[gaps > 0]

    gaps_by_scope = [
        scope_gaps((src, rel, dst)),
        scope_gaps((rel, dst)),
        scope_gaps((dst,)),
    ]

    lambdas = np.empty((len(BANK_SCOPES), m), dtype=np.float64)
    ln2 = float(np.log(2.0))
    fallback_gaps = gaps_by_scope[2]
    for i, scope in enumerate(BANK_SCOPES):
        gaps = gaps_by_scope[i]
        if gaps.shape[0] == 0:
            gaps = fallback_gaps
        if gaps.shape[0] == 0:
            half_lives = np.ones(m, dtype=np.float64)
        else:
            median_gap = max(float(np.median(gaps.astype(np.float64))), 1e-12)
            half_lives = median_gap * (geometric_factor ** geo_exponents)
            half_lives = np.maximum(half_lives, 1e-12)
        lambdas[i] = ln2 / half_lives
        if verbose:
            logger.info(
                '[bank-calibration] scope=%s  n_gaps=%d  half-lives=%s '
                '(timestamp units, geometric, factor=%g, median-anchored)',
                scope, int(gaps.shape[0]),
                np.array2string(half_lives, precision=3), geometric_factor,
            )
    return lambdas

class _Grouping:

    __slots__ = (
        'key_to_gid',
        'flat_pos',
        'flat_ts',
        'pos_off',
        'members',
        'mem_off',
        'offsets',
    )

    def __init__(
        self,
        key_to_gid: Dict,
        flat_pos: np.ndarray,
        flat_ts: np.ndarray,
        pos_off: np.ndarray,
        members: np.ndarray,
        mem_off: np.ndarray,
        offsets: np.ndarray,
    ) -> None:
        self.key_to_gid = key_to_gid
        self.flat_pos = flat_pos
        self.flat_ts = flat_ts
        self.pos_off = pos_off
        self.members = members
        self.mem_off = mem_off
        self.offsets = offsets

class EdgeReMINDIndex:

    def __init__(
        self,
        decay_rate_srd: float = 1.0 / (7 * 86400),
        decay_rate_rd: Optional[float] = None,
        decay_rate_d: Optional[float] = None,
        bank_lambdas: Optional[np.ndarray] = None,
    ) -> None:
        self.lambda_srd = float(decay_rate_srd)
        self.lambda_rd = float(
            decay_rate_srd if decay_rate_rd is None else decay_rate_rd
        )
        self.lambda_d = float(
            decay_rate_srd if decay_rate_d is None else decay_rate_d
        )
        if bank_lambdas is not None:
            bank_lambdas = np.asarray(bank_lambdas, dtype=np.float64)
            if bank_lambdas.ndim != 2 or bank_lambdas.shape[0] != len(BANK_SCOPES):
                raise ValueError(
                    f'bank_lambdas must have shape ({len(BANK_SCOPES)}, Q), '
                    f'got {tuple(bank_lambdas.shape)}'
                )
            if not np.all(bank_lambdas > 0):
                raise ValueError('bank_lambdas entries must be positive')
        self.bank_lambdas = bank_lambdas
        self.bank_size = 0 if bank_lambdas is None else int(bank_lambdas.shape[1])
        self.feat_dim = NUM_FEATURES + len(BANK_SCOPES) * self.bank_size

        q = self.bank_size
        self._dec_srd_cols = [(3, self.lambda_srd)] + [
            (NUM_FEATURES + j, float(bank_lambdas[0, j])) for j in range(q)
        ]
        self._dec_rd_cols = [(4, self.lambda_rd)] + [
            (NUM_FEATURES + q + j, float(bank_lambdas[1, j])) for j in range(q)
        ]
        self._dec_d_cols = [(5, self.lambda_d)] + [
            (NUM_FEATURES + 2 * q + j, float(bank_lambdas[2, j])) for j in range(q)
        ]

        self._srd: Optional[_Grouping] = None
        self._rd: Optional[_Grouping] = None
        self._sd: Optional[_Grouping] = None
        self._d_members: Optional[np.ndarray] = None
        self._d_offsets: Optional[np.ndarray] = None
        self._d_flat_pos: Optional[np.ndarray] = None
        self._d_flat_ts: Optional[np.ndarray] = None

    @classmethod
    def build(
        cls,
        src: np.ndarray,
        dst: np.ndarray,
        ts: np.ndarray,
        rel: np.ndarray,
        decay_rate_srd: float = 1.0 / (7 * 86400),
        decay_rate_rd: Optional[float] = None,
        decay_rate_d: Optional[float] = None,
        bank_lambdas: Optional[np.ndarray] = None,
    ) -> 'EdgeReMINDIndex':
        idx = cls(decay_rate_srd, decay_rate_rd, decay_rate_d, bank_lambdas)
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        ts = np.asarray(ts, dtype=np.int64)
        rel = np.asarray(rel, dtype=np.int64)
        n = src.shape[0]
        pos = np.arange(n, dtype=np.int64)

        idx._srd = _build_groups(
            group_key=np.stack([src, rel], axis=1),
            member=dst,
            pos=pos,
            ts=ts,
            key_dim=2,
        )
        idx._rd = _build_groups(
            group_key=rel[:, None], member=dst, pos=pos, ts=ts, key_dim=1
        )
        idx._sd = _build_groups(
            group_key=src[:, None], member=dst, pos=pos, ts=ts, key_dim=1
        )

        if n == 0:
            idx._d_members = np.empty(0, dtype=np.int64)
            idx._d_offsets = np.zeros(1, dtype=np.int64)
            idx._d_flat_pos = np.empty(0, dtype=np.int64)
            idx._d_flat_ts = np.empty(0, dtype=np.int64)
        else:
            order = np.lexsort((pos, dst))
            d_sorted = dst[order]
            pos_sorted = pos[order]
            ts_sorted = ts[order]
            change = np.empty(n, dtype=bool)
            change[0] = True
            np.not_equal(d_sorted[1:], d_sorted[:-1], out=change[1:])
            boundaries = np.nonzero(change)[0]
            idx._d_members = d_sorted[boundaries].astype(np.int64, copy=False)
            idx._d_offsets = np.concatenate(
                [boundaries, np.array([n], dtype=np.int64)]
            ).astype(np.int64, copy=False)
            idx._d_flat_pos = pos_sorted.astype(np.int64, copy=False)
            idx._d_flat_ts = ts_sorted.astype(np.int64, copy=False)
        return idx

    def _scatter(
        self,
        grouping: Optional[_Grouping],
        key,
        cutoff: int,
        t: int,
        cand: np.ndarray,
        feats: np.ndarray,
        cnt_col: int,
        dec_cols: Optional[List[Tuple[int, float]]],
    ) -> None:
        if grouping is None:
            return
        gid = grouping.key_to_gid.get(key)
        if gid is None:
            return
        m0 = int(grouping.mem_off[gid])
        m1 = int(grouping.mem_off[gid + 1])
        members = grouping.members[m0:m1]
        if members.shape[0] == 0:
            return

        loc = np.searchsorted(members, cand, side='left')
        loc_clip = np.clip(loc, 0, members.shape[0] - 1)
        valid = members[loc_clip] == cand
        if not valid.any():
            return

        rows = np.nonzero(valid)[0]
        mem_idx = loc_clip[valid]

        p0 = int(grouping.pos_off[gid])
        group_len = int(grouping.pos_off[gid + 1]) - p0
        offsets = grouping.offsets[m0:m1]
        seg_start = offsets[mem_idx]

        if mem_idx.shape[0] * _WHOLE_GROUP_FACTOR < group_len:
            flat_pos = grouping.flat_pos
            m_count = members.shape[0]
            nxt = np.minimum(mem_idx + 1, m_count - 1)
            seg_end = np.where(mem_idx + 1 < m_count, offsets[nxt], group_len)
            c = np.empty(mem_idx.shape[0], dtype=np.int64)
            for i in range(mem_idx.shape[0]):
                a = p0 + int(seg_start[i])
                b = p0 + int(seg_end[i])
                c[i] = np.searchsorted(flat_pos[a:b], cutoff, side='left')
        else:
            before = (grouping.flat_pos[p0 : p0 + group_len] < cutoff).astype(np.int64)
            c = np.add.reduceat(before, offsets)[mem_idx]

        feats[rows, cnt_col] = c.astype(np.float32)

        if dec_cols:
            good = c > 0
            if good.any():
                gi = np.nonzero(good)[0]
                last_flat = p0 + seg_start[gi] + c[gi] - 1
                last_flat = np.clip(last_flat, 0, grouping.flat_ts.shape[0] - 1)
                last_ts = grouping.flat_ts[last_flat]
                dt = (t - last_ts).astype(np.float64)
                good_rows = rows[gi]
                for dec_col, lam in dec_cols:
                    feats[good_rows, dec_col] = np.exp(-lam * dt).astype(np.float32)

    def _query_d_pop(
        self,
        cand: np.ndarray,
        cutoff: int,
        t: int,
        feats: np.ndarray,
        dec_cols: List[Tuple[int, float]],
    ) -> None:
        if self._d_members is None or self._d_members.shape[0] == 0:
            return
        loc = np.searchsorted(self._d_members, cand, side='left')
        loc_clip = np.clip(loc, 0, self._d_members.shape[0] - 1)
        valid = self._d_members[loc_clip] == cand
        if not valid.any():
            return
        rows = np.nonzero(valid)[0]
        mem_idx = loc_clip[valid]

        seg_start = self._d_offsets[mem_idx]
        seg_end = self._d_offsets[mem_idx + 1]
        n_match = mem_idx.shape[0]
        total_entries = self._d_flat_pos.shape[0]

        if n_match * _WHOLE_GROUP_FACTOR < total_entries:
            c = np.empty(n_match, dtype=np.int64)
            for i in range(n_match):
                a = int(seg_start[i])
                b = int(seg_end[i])
                c[i] = np.searchsorted(self._d_flat_pos[a:b], cutoff, side='left')
        else:
            before_mask = (self._d_flat_pos < cutoff).astype(np.int64)
            cumsum = np.concatenate(([0], np.cumsum(before_mask)))
            c = cumsum[seg_end] - cumsum[seg_start]

        good = c > 0
        if not good.any():
            return
        gi = np.nonzero(good)[0]
        last_flat = seg_start[gi] + c[gi] - 1
        last_flat = np.clip(last_flat, 0, self._d_flat_ts.shape[0] - 1)
        last_ts = self._d_flat_ts[last_flat]
        dt = (t - last_ts).astype(np.float64)
        good_rows = rows[gi]
        for dec_col, lam in dec_cols:
            feats[good_rows, dec_col] = np.exp(-lam * dt).astype(np.float32)

    def query_features(
        self,
        s: int,
        r: int,
        t: int,
        cutoff: int,
        cand: np.ndarray,
    ) -> np.ndarray:
        cand = np.asarray(cand, dtype=np.int64)
        feats = np.zeros((cand.shape[0], self.feat_dim), dtype=np.float32)
        self._scatter(self._srd, (s, r), cutoff, t, cand, feats, 0, self._dec_srd_cols)
        self._scatter(self._rd, r, cutoff, t, cand, feats, 1, self._dec_rd_cols)
        self._scatter(self._sd, s, cutoff, t, cand, feats, 2, None)
        self._query_d_pop(cand, cutoff, t, feats, self._dec_d_cols)
        return feats


def _build_groups(
    group_key: np.ndarray,
    member: np.ndarray,
    pos: np.ndarray,
    ts: np.ndarray,
    key_dim: int,
) -> _Grouping:
    n = pos.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        zero = np.zeros(1, dtype=np.int64)
        return _Grouping({}, empty, empty, zero, empty, zero, empty)

    if key_dim == 1:
        gk0_raw = group_key[:, 0]
        order = np.lexsort((pos, member, gk0_raw))
        gk0_s = gk0_raw[order]
        gk1_s = None
        group_change = np.empty(n, dtype=bool)
        group_change[0] = True
        np.not_equal(gk0_s[1:], gk0_s[:-1], out=group_change[1:])
    else:
        gk0_raw = group_key[:, 0]
        gk1_raw = group_key[:, 1]
        order = np.lexsort((pos, member, gk1_raw, gk0_raw))
        gk0_s = gk0_raw[order]
        gk1_s = gk1_raw[order]
        group_change = np.empty(n, dtype=bool)
        group_change[0] = True
        group_change[1:] = (gk0_s[1:] != gk0_s[:-1]) | (gk1_s[1:] != gk1_s[:-1])

    pos_s = pos[order].astype(np.int64, copy=False)
    ts_s = ts[order].astype(np.int64, copy=False)
    mem_s = member[order].astype(np.int64, copy=False)

    group_starts = np.nonzero(group_change)[0]
    g_count = group_starts.shape[0]
    pos_off = np.empty(g_count + 1, dtype=np.int64)
    pos_off[:-1] = group_starts
    pos_off[-1] = n

    member_change = group_change.copy()
    member_change[1:] |= mem_s[1:] != mem_s[:-1]
    member_starts = np.nonzero(member_change)[0]
    members_all = mem_s[member_starts]

    member_group = np.searchsorted(group_starts, member_starts, side='right') - 1
    offsets = (member_starts - group_starts[member_group]).astype(np.int64)
    mem_off = np.empty(g_count + 1, dtype=np.int64)
    mem_off[0] = 0
    np.cumsum(np.bincount(member_group, minlength=g_count), out=mem_off[1:])

    if key_dim == 1:
        keys = gk0_s[group_starts].tolist()
    else:
        keys = list(zip(gk0_s[group_starts].tolist(), gk1_s[group_starts].tolist()))
    key_to_gid = dict(zip(keys, range(g_count)))

    return _Grouping(key_to_gid, pos_s, ts_s, pos_off, members_all, mem_off, offsets)

_WORKER_INDEX: Optional[EdgeReMINDIndex] = None
_WORKER_THETA: Optional[np.ndarray] = None
_WORKER_THETAS: Optional[List[np.ndarray]] = None


def _set_worker_globals(
    index: EdgeReMINDIndex,
    theta: Optional[np.ndarray],
    thetas: Optional[List[np.ndarray]] = None,
) -> None:
    global _WORKER_INDEX, _WORKER_THETA, _WORKER_THETAS
    _WORKER_INDEX = index
    _WORKER_THETA = theta
    _WORKER_THETAS = thetas

def _extract_chunk(args: Tuple) -> Tuple[int, np.ndarray]:
    start, q_src, q_dst, q_ts, q_rel, neg_dst, cutoffs = args
    index = _WORKER_INDEX
    k = neg_dst.shape[1]
    width = index.feat_dim
    out = np.zeros((q_src.shape[0], k + 1, width), dtype=np.float32)
    for i in range(q_src.shape[0]):
        cand = np.empty(k + 1, dtype=np.int64)
        cand[0] = q_dst[i]
        cand[1:] = neg_dst[i]
        out[i] = index.query_features(
            int(q_src[i]), int(q_rel[i]), int(q_ts[i]), int(cutoffs[i]), cand
        )
    return start, out

def _eval_chunk(args: Tuple) -> Tuple[int, np.ndarray]:
    start, q_src, q_rel, q_ts, cutoffs, pos_dst, neg_list = args
    index = _WORKER_INDEX
    theta = _WORKER_THETA
    rr = np.empty(len(q_src), dtype=np.float64)
    for i in range(len(q_src)):
        negs = neg_list[i]
        cand = np.empty(negs.shape[0] + 1, dtype=np.int64)
        cand[0] = pos_dst[i]
        cand[1:] = negs
        feats = index.query_features(
            int(q_src[i]), int(q_rel[i]), int(q_ts[i]), int(cutoffs[i]), cand
        )
        scores = feats @ theta[int(q_rel[i])]
        pos_score = scores[0]
        neg_scores = scores[1:]
        opt = int((neg_scores > pos_score).sum())
        pess = int((neg_scores >= pos_score).sum())
        rr[i] = 1.0 / (0.5 * (opt + pess) + 1.0)
    return start, rr

def _eval_chunk_multi(args: Tuple) -> Tuple[int, np.ndarray]:
    start, q_src, q_rel, q_ts, cutoffs, pos_dst, neg_list = args
    index = _WORKER_INDEX
    thetas = _WORKER_THETAS
    m = len(thetas)
    rr = np.empty((m, len(q_src)), dtype=np.float64)
    for i in range(len(q_src)):
        negs = neg_list[i]
        cand = np.empty(negs.shape[0] + 1, dtype=np.int64)
        cand[0] = pos_dst[i]
        cand[1:] = negs
        feats = index.query_features(
            int(q_src[i]), int(q_rel[i]), int(q_ts[i]), int(cutoffs[i]), cand
        )
        r = int(q_rel[i])
        for j in range(m):
            scores = feats @ thetas[j][r]
            pos_score = scores[0]
            neg_scores = scores[1:]
            opt = int((neg_scores > pos_score).sum())
            pess = int((neg_scores >= pos_score).sum())
            rr[j, i] = 1.0 / (0.5 * (opt + pess) + 1.0)
    return start, rr


def _chunk_ranges(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    return [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]


def parallel_extract_features(
    index: EdgeReMINDIndex,
    q_src: np.ndarray,
    q_dst: np.ndarray,
    q_ts: np.ndarray,
    q_rel: np.ndarray,
    neg_dst: np.ndarray,
    cutoffs: np.ndarray,
    num_workers: int = 0,
    chunk_size: int = 4096,
) -> torch.Tensor:
    n = q_src.shape[0]
    k = neg_dst.shape[1]
    width = index.feat_dim
    out = np.zeros((n, k + 1, width), dtype=np.float32)

    tasks = [
        (
            start,
            q_src[start:end],
            q_dst[start:end],
            q_ts[start:end],
            q_rel[start:end],
            neg_dst[start:end],
            cutoffs[start:end],
        )
        for start, end in _chunk_ranges(n, chunk_size)
    ]

    if num_workers and num_workers > 0:
        mp_method = 'spawn' if (sys.platform == 'darwin' or sys.platform == 'win32') else 'fork'
        ctx = mp.get_context(mp_method)
        if mp_method == 'fork':
            _set_worker_globals(index, None)
        with ProcessPoolExecutor(
            max_workers=num_workers, 
            mp_context=ctx,
            initializer=_set_worker_globals,
            initargs=(index, None)
        ) as ex:
            for start, chunk in ex.map(_extract_chunk, tasks):
                out[start : start + chunk.shape[0]] = chunk
    else:
        _set_worker_globals(index, None)
        for task in tasks:
            start, chunk = _extract_chunk(task)
            out[start : start + chunk.shape[0]] = chunk

    return torch.from_numpy(out)


def parallel_evaluate(
    index: EdgeReMINDIndex,
    theta: np.ndarray,
    q_src: np.ndarray,
    q_rel: np.ndarray,
    q_ts: np.ndarray,
    cutoffs: np.ndarray,
    pos_dst: np.ndarray,
    neg_list: List[np.ndarray],
    num_workers: int = 0,
    chunk_size: int = 4096,
) -> np.ndarray:
    n = len(q_src)
    theta = np.asarray(theta, dtype=np.float32)
    out = np.empty(n, dtype=np.float64)

    tasks = [
        (
            start,
            q_src[start:end],
            q_rel[start:end],
            q_ts[start:end],
            cutoffs[start:end],
            pos_dst[start:end],
            neg_list[start:end],
        )
        for start, end in _chunk_ranges(n, chunk_size)
    ]

    if num_workers and num_workers > 0:
        mp_method = 'spawn' if (sys.platform == 'darwin' or sys.platform == 'win32') else 'fork'
        ctx = mp.get_context(mp_method)
        if mp_method == 'fork':
            _set_worker_globals(index, theta)
        with ProcessPoolExecutor(
            max_workers=num_workers, 
            mp_context=ctx,
            initializer=_set_worker_globals,
            initargs=(index, theta)
        ) as ex:
            for start, rr in ex.map(_eval_chunk, tasks):
                out[start : start + rr.shape[0]] = rr
    else:
        _set_worker_globals(index, theta)
        for task in tasks:
            start, rr = _eval_chunk(task)
            out[start : start + rr.shape[0]] = rr

    return out


def extract_training_features(
    index: EdgeReMINDIndex,
    src: np.ndarray,
    dst: np.ndarray,
    ts: np.ndarray,
    rel: np.ndarray,
    num_relations: int,
    first_dst: int,
    last_dst: int,
    k: int = 20,
    feature_bsize: int = 200,
    max_train_queries: Optional[int] = 2_000_000,
    num_workers: int = 0,
    chunk_size: int = 4096,
    seed: int = 1337,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from tgm.nn.modules.edgeremind import _RelationAwareNegativeSampler

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    ts = np.asarray(ts, dtype=np.int64)
    rel = np.asarray(rel, dtype=np.int64)
    n_total = src.shape[0]
    n_query = n_total if max_train_queries is None else min(max_train_queries, n_total)
    prefix = n_total - n_query

    sampler = _RelationAwareNegativeSampler(
        rel=rel,
        dst=dst,
        num_relations=num_relations,
        first_dst=first_dst,
        last_dst=last_dst,
        seed=seed,
    )

    q_src = src[prefix:]
    q_dst = dst[prefix:]
    q_ts = ts[prefix:]
    q_rel = rel[prefix:]
    neg = sampler.sample_many(q_rel, q_dst, k)
    p_idx = np.arange(n_query, dtype=np.int64)
    cutoffs = prefix + (p_idx // feature_bsize) * feature_bsize

    feats = parallel_extract_features(
        index,
        q_src,
        q_dst,
        q_ts,
        q_rel,
        neg,
        cutoffs,
        num_workers=num_workers,
        chunk_size=chunk_size,
    )
    return feats, torch.from_numpy(q_rel.copy())
