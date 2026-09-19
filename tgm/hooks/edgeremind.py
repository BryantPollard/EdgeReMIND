from typing import List

import numpy as np
import torch

from tgm.core import DGBatch, DGraph
from tgm.hooks.base import StatelessHook
from tgm.nn.modules.edgeremind_index import EdgeReMINDIndex

class EdgeReMINDFeatureHook(StatelessHook):
    _cls_requires = {'edge_src', 'edge_dst', 'edge_time', 'edge_type', 'neg_batch_list'}
    _cls_produces = {'edgeremind_feats'}

    def __init__(self, index: EdgeReMINDIndex, split_offset: int, id: str | None = None):
        super().__init__()
        self.index = index
        self.split_offset = int(split_offset)
        self._seen = 0
        self._id = id

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        n_edges = batch.edge_src.numel()
        feats_list: List[torch.Tensor] = []
        if n_edges > 0:
            cutoff = self.split_offset + self._seen
            src = batch.edge_src.cpu().numpy()
            dst = batch.edge_dst.cpu().numpy()
            ts = batch.edge_time.cpu().numpy()
            etype = batch.edge_type.cpu().numpy()
            for idx in range(n_edges):
                neg = batch.neg_batch_list[idx].cpu().numpy()
                cand = np.empty(neg.shape[0] + 1, dtype=np.int64)
                cand[0] = dst[idx]
                cand[1:] = neg
                f = self.index.query_features(
                    int(src[idx]), int(etype[idx]), int(ts[idx]), int(cutoff), cand
                )
                feats_list.append(torch.from_numpy(f))
        self._seen += n_edges
        self.add_batch_attribute(batch, 'edgeremind_feats', feats_list)
        return batch
