from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from tgm.util.logging import _get_logger

logger = _get_logger(__name__)

DEFAULT_WEIGHTS: Tuple[float, ...] = (
    1.0,
    1e-3,
    1e-2,
    2.0,
    1e-2,
    0.0,
)

NUM_FEATURES = 6

BANK_SCOPES: Tuple[str, ...] = ('srd', 'rd', 'd')

class EdgeReMINDPredictor:

    def __init__(
        self,
        num_relations: int,
        decay_rate_srd: float = 1.0 / (7 * 86400),
        decay_rate_rd: Optional[float] = None,
        decay_rate_d: Optional[float] = None,
        init_weights: Tuple[float, ...] = DEFAULT_WEIGHTS,
        bank_lambdas: Optional[np.ndarray] = None,
        device: torch.device | str = 'cpu',
    ) -> None:
        if num_relations <= 0:
            raise ValueError(f'num_relations must be positive, got {num_relations}')
        if len(init_weights) != NUM_FEATURES:
            raise ValueError(f'init_weights must have length {NUM_FEATURES}')

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

        self.num_relations = int(num_relations)
        self.feat_dim = NUM_FEATURES + len(BANK_SCOPES) * self.bank_size
        self.lambda_srd = float(decay_rate_srd)
        self.lambda_rd = float(
            decay_rate_srd if decay_rate_rd is None else decay_rate_rd
        )
        self.lambda_d = float(
            decay_rate_srd if decay_rate_d is None else decay_rate_d
        )
        self.device = torch.device(device)

        full_init = tuple(init_weights) + (0.0,) * (self.feat_dim - NUM_FEATURES)
        init = torch.tensor(full_init, dtype=torch.float32, device=self.device)
        self._theta = init.unsqueeze(0).repeat(self.num_relations, 1).contiguous()

        self._cnt_srd: Dict[Tuple[int, int, int], int] = {}
        self._cnt_rd: Dict[Tuple[int, int], int] = {}
        self._cnt_sd: Dict[Tuple[int, int], int] = {}
        self._last_srd: Dict[Tuple[int, int, int], int] = {}
        self._last_rd: Dict[Tuple[int, int], int] = {}
        self._last_d_ts: Dict[int, int] = {}

    @property
    def theta(self) -> torch.Tensor:
        return self._theta

    @theta.setter
    def theta(self, value: torch.Tensor) -> None:
        if value.shape != (self.num_relations, self.feat_dim):
            raise ValueError(
                f'theta must have shape ({self.num_relations}, {self.feat_dim}), '
                f'got {tuple(value.shape)}'
            )
        self._theta = value.detach().to(self.device).float().contiguous()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {'theta': self._theta.cpu()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.theta = state['theta']

    def save(self, path: str) -> None:
        np.savez_compressed(path, theta=self._theta.cpu().numpy())

    def load(self, path: str) -> None:
        with np.load(path) as data:
            self.theta = torch.from_numpy(data['theta'])

    def update(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> None:
        s_arr = src.detach().cpu().numpy()
        d_arr = dst.detach().cpu().numpy()
        t_arr = ts.detach().cpu().numpy()
        if edge_type is None:
            r_arr = np.zeros_like(s_arr)
        else:
            r_arr = edge_type.detach().cpu().numpy()

        cnt_srd, cnt_rd, cnt_sd = self._cnt_srd, self._cnt_rd, self._cnt_sd
        last_srd, last_rd, last_d_ts = self._last_srd, self._last_rd, self._last_d_ts

        for s, d, t, r in zip(
            s_arr.tolist(), d_arr.tolist(), t_arr.tolist(), r_arr.tolist()
        ):
            k_srd = (s, r, d)
            k_rd = (r, d)
            k_sd = (s, d)
            cnt_srd[k_srd] = cnt_srd.get(k_srd, 0) + 1
            cnt_rd[k_rd] = cnt_rd.get(k_rd, 0) + 1
            cnt_sd[k_sd] = cnt_sd.get(k_sd, 0) + 1
            if t >= last_srd.get(k_srd, t):
                last_srd[k_srd] = t
            if t >= last_rd.get(k_rd, t):
                last_rd[k_rd] = t
            if t >= last_d_ts.get(d, t):
                last_d_ts[d] = t

    def compute_features(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s_arr = src.detach().cpu().numpy()
        d_arr = dst.detach().cpu().numpy()
        t_arr = ts.detach().cpu().numpy()
        if edge_type is None:
            r_arr = np.zeros_like(s_arr)
        else:
            r_arr = edge_type.detach().cpu().numpy()

        m = s_arr.shape[0]
        feats = np.zeros((m, self.feat_dim), dtype=np.float32)

        cnt_srd, cnt_rd, cnt_sd = self._cnt_srd, self._cnt_rd, self._cnt_sd
        last_srd, last_rd, last_d_ts = self._last_srd, self._last_rd, self._last_d_ts
        lam_srd, lam_rd, lam_d = self.lambda_srd, self.lambda_rd, self.lambda_d

        q = self.bank_size
        if q > 0:
            bank_srd = self.bank_lambdas[0].tolist()
            bank_rd = self.bank_lambdas[1].tolist()
            bank_d = self.bank_lambdas[2].tolist()
        else:
            bank_srd = bank_rd = bank_d = []
        col_srd0 = NUM_FEATURES
        col_rd0 = NUM_FEATURES + q
        col_d0 = NUM_FEATURES + 2 * q

        s_list = s_arr.tolist()
        d_list = d_arr.tolist()
        t_list = t_arr.tolist()
        r_list = r_arr.tolist()

        for i in range(m):
            s, d, t, r = s_list[i], d_list[i], t_list[i], r_list[i]
            k_srd = (s, r, d)
            k_rd = (r, d)
            feats[i, 0] = cnt_srd.get(k_srd, 0)
            feats[i, 1] = cnt_rd.get(k_rd, 0)
            feats[i, 2] = cnt_sd.get((s, d), 0)
            ts_srd = last_srd.get(k_srd)
            if ts_srd is not None:
                dt = t - ts_srd
                feats[i, 3] = math.exp(-lam_srd * dt)
                for j, lam in enumerate(bank_srd):
                    feats[i, col_srd0 + j] = math.exp(-lam * dt)
            ts_rd = last_rd.get(k_rd)
            if ts_rd is not None:
                dt = t - ts_rd
                feats[i, 4] = math.exp(-lam_rd * dt)
                for j, lam in enumerate(bank_rd):
                    feats[i, col_rd0 + j] = math.exp(-lam * dt)
            ts_d = last_d_ts.get(d)
            if ts_d is not None:
                dt = t - ts_d
                feats[i, 5] = math.exp(-lam_d * dt)
                for j, lam in enumerate(bank_d):
                    feats[i, col_d0 + j] = math.exp(-lam * dt)

        return torch.from_numpy(feats).to(self.device)

    def score_features(
        self, features: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        rel = edge_type.to(self.device).long().clamp_(0, self.num_relations - 1)
        w = self._theta[rel]
        return (w * features.to(self.device)).sum(dim=-1)

    def __call__(
        self,
        query_src: torch.Tensor,
        query_dst: torch.Tensor,
        query_ts: torch.Tensor,
        query_type: torch.Tensor,
    ) -> torch.Tensor:
        feats = self.compute_features(query_src, query_dst, query_ts, query_type)
        return self.score_features(feats, query_type)

class _RelationAwareNegativeSampler:

    def __init__(
        self,
        rel: np.ndarray,
        dst: np.ndarray,
        num_relations: int,
        first_dst: int,
        last_dst: int,
        seed: int = 1337,
    ) -> None:
        self.num_relations = num_relations
        self.first_dst = int(first_dst)
        self.last_dst = int(last_dst)
        self.rng = np.random.default_rng(seed)

        self.pools: Dict[int, np.ndarray] = {}
        order = np.argsort(rel, kind='stable')
        rel_sorted = rel[order]
        dst_sorted = dst[order]
        boundaries = np.searchsorted(
            rel_sorted, np.arange(num_relations + 1), side='left'
        )
        for r in range(num_relations):
            lo, hi = boundaries[r], boundaries[r + 1]
            if hi > lo:
                self.pools[r] = np.unique(dst_sorted[lo:hi])

    def sample(self, r: int, pos_dst: int, k: int) -> np.ndarray:
        pool = self.pools.get(r)
        if pool is None or pool.shape[0] < k + 1:
            negs = self.rng.integers(self.first_dst, self.last_dst + 1, size=k)
            return negs
        idx = self.rng.integers(0, pool.shape[0], size=k)
        negs = pool[idx]
        collide = negs == pos_dst
        if collide.any():
            negs[collide] = pool[
                self.rng.integers(0, pool.shape[0], size=int(collide.sum()))
            ]
        return negs

    def sample_many(self, rels: np.ndarray, pos_dsts: np.ndarray, k: int) -> np.ndarray:
        rels = np.asarray(rels, dtype=np.int64)
        pos_dsts = np.asarray(pos_dsts, dtype=np.int64)
        n = rels.shape[0]
        out = np.empty((n, k), dtype=np.int64)
        if n == 0:
            return out

        order = np.argsort(rels, kind='stable')
        rels_sorted = rels[order]
        bounds = np.searchsorted(
            rels_sorted, np.arange(self.num_relations + 1), side='left'
        )
        for r in range(self.num_relations):
            lo, hi = int(bounds[r]), int(bounds[r + 1])
            if hi <= lo:
                continue
            rows = order[lo:hi]
            cnt = hi - lo
            pool = self.pools.get(r)
            if pool is None or pool.shape[0] < k + 1:
                out[rows] = self.rng.integers(
                    self.first_dst, self.last_dst + 1, size=(cnt, k)
                )
            else:
                negs = pool[self.rng.integers(0, pool.shape[0], size=(cnt, k))]
                collide = negs == pos_dsts[rows][:, None]
                if collide.any():
                    ncol = int(collide.sum())
                    negs[collide] = pool[self.rng.integers(0, pool.shape[0], size=ncol)]
                out[rows] = negs
        return out

class EdgeReMINDTrainer:

    def __init__(
        self,
        predictor: EdgeReMINDPredictor,
        first_dst: int,
        last_dst: int,
        k: int = 20,
        feature_bsize: int = 200,
        max_train_queries: Optional[int] = 2_000_000,
        seed: int = 1337,
    ) -> None:
        self.predictor = predictor
        self.first_dst = int(first_dst)
        self.last_dst = int(last_dst)
        self.k = int(k)
        self.feature_bsize = int(feature_bsize)
        self.max_train_queries = max_train_queries
        self.seed = int(seed)

    def extract_features(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        ts: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_all = src.detach().cpu().numpy()
        d_all = dst.detach().cpu().numpy()
        t_all = ts.detach().cpu().numpy()
        r_all = edge_type.detach().cpu().numpy()

        n_total = s_all.shape[0]
        if self.max_train_queries is None:
            n_query = n_total
        else:
            n_query = min(self.max_train_queries, n_total)
        prefix = n_total - n_query

        sampler = _RelationAwareNegativeSampler(
            rel=r_all,
            dst=d_all,
            num_relations=self.predictor.num_relations,
            first_dst=self.first_dst,
            last_dst=self.last_dst,
            seed=self.seed,
        )

        if prefix > 0:
            self.predictor.update(
                src[:prefix], dst[:prefix], ts[:prefix], edge_type[:prefix]
            )

        k = self.k
        b = self.feature_bsize
        feats = np.zeros((n_query, k + 1, self.predictor.feat_dim), dtype=np.float32)
        rels = r_all[prefix:].astype(np.int64)

        neg_all = sampler.sample_many(rels, d_all[prefix:].astype(np.int64), k)

        for start in range(prefix, n_total, b):
            end = min(start + b, n_total)
            bs = end - start
            n_cand = bs * (k + 1)

            cand_s = np.empty(n_cand, dtype=np.int64)
            cand_d = np.empty(n_cand, dtype=np.int64)
            cand_t = np.empty(n_cand, dtype=np.int64)
            cand_r = np.empty(n_cand, dtype=np.int64)

            for j in range(bs):
                gi = start + j
                s, d, t, r = (
                    int(s_all[gi]),
                    int(d_all[gi]),
                    int(t_all[gi]),
                    int(r_all[gi]),
                )
                negs = neg_all[gi - prefix]
                base = j * (k + 1)
                cand_s[base] = s
                cand_d[base] = d
                cand_t[base] = t
                cand_r[base] = r
                cand_s[base + 1 : base + 1 + k] = s
                cand_d[base + 1 : base + 1 + k] = negs
                cand_t[base + 1 : base + 1 + k] = t
                cand_r[base + 1 : base + 1 + k] = r

            batch_feats = self.predictor.compute_features(
                torch.from_numpy(cand_s),
                torch.from_numpy(cand_d),
                torch.from_numpy(cand_t),
                torch.from_numpy(cand_r),
            )
            feats[start - prefix : end - prefix] = (
                batch_feats.cpu().numpy().reshape(bs, k + 1, self.predictor.feat_dim)
            )

            self.predictor.update(
                src[start:end], dst[start:end], ts[start:end], edge_type[start:end]
            )

        return torch.from_numpy(feats), torch.from_numpy(rels)

    def optimize(
        self,
        features: torch.Tensor,
        rels: torch.Tensor,
        epochs: int = 5,
        lr: float = 1e-2,
        opt_bsize: int = 1024,
        patience: int = 0,
        min_delta: float = 1e-3,
        shrinkage: float = 0.0,
        theta_prior: Optional[torch.Tensor] = None,
        return_snapshots: bool = False,
        verbose: bool = True,
    ):
        device = self.predictor.device
        features = features.to(device)
        rels = rels.to(device).long()

        theta = self.predictor.theta.clone().detach().to(device).requires_grad_(True)
        if shrinkage > 0.0:
            prior = (
                theta_prior.detach().to(device)
                if theta_prior is not None
                else theta.detach().clone()
            )
        else:
            prior = None
        optimizer = torch.optim.Adam([theta], lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        n = features.shape[0]
        target = torch.zeros(opt_bsize, dtype=torch.long, device=device)
        epoch_losses = []
        snapshots: list = []

        best_loss = float('inf')
        best_theta = None
        no_improve = 0

        for epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            total, count = 0.0, 0
            for start in range(0, n, opt_bsize):
                idx = perm[start : start + opt_bsize]
                f = features[idx]  # (b, K+1, 6)
                r = rels[idx]  # (b,)
                w = theta[r].unsqueeze(1)  # (b, 1, 6)
                scores = (f * w).sum(dim=-1)  # (b, K+1)
                tgt = target[: scores.shape[0]]
                loss = loss_fn(scores, tgt)
                if prior is not None:
                    loss = loss + shrinkage * (theta - prior).pow(2).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += loss.item() * scores.shape[0]
                count += scores.shape[0]

            mean_loss = total / max(count, 1)
            epoch_losses.append(mean_loss)
            if return_snapshots:
                snapshots.append(theta.detach().clone())

            meaningful = mean_loss < best_loss * (1.0 - min_delta)
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_theta = theta.detach().clone()
            no_improve = 0 if meaningful else no_improve + 1

            if verbose:
                logger.info(
                    '[EdgeReMIND] epoch %d/%d  loss=%.4f', epoch + 1, epochs, mean_loss
                )

            if patience > 0 and no_improve >= patience:
                if verbose:
                    logger.info(
                        '[EdgeReMIND] early stop at epoch %d/%d: no >=%.2g relative '
                        'loss improvement for %d epochs (best loss=%.4f)',
                        epoch + 1,
                        epochs,
                        min_delta,
                        patience,
                        best_loss,
                    )
                break

        if patience > 0 and best_theta is not None:
            self.predictor.theta = best_theta
        else:
            self.predictor.theta = theta.detach()
        if return_snapshots:
            return epoch_losses, snapshots
        return epoch_losses
