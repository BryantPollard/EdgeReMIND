import argparse
import os
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)

import numpy as np
import torch
from tgb.linkproppred.evaluate import Evaluator
from tqdm import tqdm

from tgm import DGraph
from tgm.constants import METRIC_TGB_LINKPROPPRED
from tgm.data import DGData, DGDataLoader
from tgm.hooks import HookManager, TGBTHGNegativeEdgeSamplerHook
from tgm.nn import (
    EdgeReMINDPredictor,
    EdgeReMINDTrainer,
    build_full_index,
    calibrate_bank_lambdas,
    evaluate_parallel,
    train_learned,
)
from tgm.util.logging import enable_logging, log_latency, log_metric
from tgm.util.seed import seed_everything

parser = argparse.ArgumentParser(
    description='EdgeReMIND LinkPropPred Example for heterogeneous graphs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--dataset', type=str, default='thgl-github', help='dataset name')
parser.add_argument(
    '--num-workers',
    type=int,
    default=None,
    help='parallel workers',
)
parser.add_argument(
    '--log-file-path', type=str, default=None, help='log file path'
)


@log_latency
def eval(
    loader: DGDataLoader,
    model: EdgeReMINDPredictor,
    evaluator: Evaluator,
) -> float:
    perf_list = []
    for batch in tqdm(loader):
        if batch.edge_src.numel() == 0:
            continue

        cand_src, cand_dst, cand_ts, cand_type, spans = [], [], [], [], []
        for idx, neg_batch in enumerate(batch.neg_batch_list):
            n = len(neg_batch) + 1
            cand_src.append(batch.edge_src[idx].repeat(n))
            cand_dst.append(torch.cat([batch.edge_dst[idx].unsqueeze(0), neg_batch]))
            cand_ts.append(batch.edge_time[idx].repeat(n))
            cand_type.append(batch.edge_type[idx].repeat(n))
            spans.append(n)

        all_src = torch.cat(cand_src)
        all_dst = torch.cat(cand_dst)
        all_ts = torch.cat(cand_ts)
        all_type = torch.cat(cand_type)
        all_scores = model(all_src, all_dst, all_ts, all_type)

        offset = 0
        for n in spans:
            y_pred = all_scores[offset : offset + n]
            offset += n
            input_dict = {
                'y_pred_pos': y_pred[0].unsqueeze(0),
                'y_pred_neg': y_pred[1:],
                'eval_metric': [METRIC_TGB_LINKPROPPRED],
            }
            perf_list.append(evaluator.eval(input_dict)[METRIC_TGB_LINKPROPPRED])

        model.update(batch.edge_src, batch.edge_dst, batch.edge_time, batch.edge_type)

    return float(np.mean(perf_list))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.num_workers is None:
        try:
            args.num_workers = len(os.sched_getaffinity(0))
        except AttributeError:
            args.num_workers = os.cpu_count() or 1
    enable_logging(log_file_path=args.log_file_path)

    seed_everything(args.seed)
    evaluator = Evaluator(name=args.dataset)

    data = DGData.from_tgb(args.dataset)
    min_node = data.edge_index.min().int()
    max_node = data.edge_index.max().int()
    if data.edge_type is None:
        raise ValueError(
            'EdgeReMIND requires a multi-relational dataset (edge_type is None)'
        )
    num_relations = int(data.edge_type.max().item()) + 1

    train_data, val_data, test_data = data.split()
    train_dg = DGraph(train_data)
    val_dg = DGraph(val_data)
    test_dg = DGraph(test_data)

    train_mat = train_dg.materialize(materialize_features=False)

    hm = HookManager(keys=['val', 'test'])
    hm.register(
        'val',
        TGBTHGNegativeEdgeSamplerHook(
            args.dataset,
            split_mode='val',
            first_node_id=min_node,
            last_node_id=max_node,
            node_type=data.node_type,
        ),
    )
    hm.register(
        'test',
        TGBTHGNegativeEdgeSamplerHook(
            args.dataset,
            split_mode='test',
            first_node_id=min_node,
            last_node_id=max_node,
            node_type=data.node_type,
        ),
    )

    val_loader = DGDataLoader(val_dg, 200, hook_manager=hm)
    test_loader = DGDataLoader(test_dg, 200, hook_manager=hm)

    bank_lambdas = calibrate_bank_lambdas(
        train_mat.edge_src.cpu().numpy(),
        train_mat.edge_dst.cpu().numpy(),
        train_mat.edge_time.cpu().numpy(),
        train_mat.edge_type.cpu().numpy(),
        n_timescales=3,
        geometric_factor=2.0,
    )
    _hl = np.log(2.0) / bank_lambdas
    print(
        '[bank] geometric calibration, factor 2.0 (median-anchored half-lives, '
        'timestamp units):\n'
        f'  srd: {np.array2string(_hl[0], precision=3)}\n'
        f'  rd:  {np.array2string(_hl[1], precision=3)}\n'
        f'  d:   {np.array2string(_hl[2], precision=3)}',
        flush=True,
    )

    model = EdgeReMINDPredictor(
        num_relations=num_relations,
        decay_rate_srd=1.0 / (7 * 86400),
        decay_rate_d=None,
        init_weights=(1.0, 1e-3, 1e-2, 2.0, 1e-2, 0.0),
        bank_lambdas=bank_lambdas,
    )
    trainer = EdgeReMINDTrainer(
        model,
        first_dst=int(min_node),
        last_dst=int(max_node),
        k=20,
        feature_bsize=200,
        max_train_queries=2_000_000,
        seed=args.seed,
    )


    val_mat = val_dg.materialize(materialize_features=False)
    test_mat = test_dg.materialize(materialize_features=False)
    index, n_train, n_val = build_full_index(
        train_mat, val_mat, test_mat, 1.0 / (7 * 86400), None,
        bank_lambdas=bank_lambdas,
    )
    with hm.activate('val'):
        _, val_mrr = train_learned(
            model,
            trainer,
            index,
            train_mat,
            val_loader,
            num_relations=num_relations,
            first_dst=int(min_node),
            last_dst=int(max_node),
            num_workers=args.num_workers,
            epochs=30,
            lr=1e-3,
            opt_bsize=1024,
            n_train=n_train,
            bsize=200,
            patience=0,
            min_delta=1e-3,
            shrinkage=0.0,
            smooth_window=3,
        )
    theta = model.theta.detach().cpu().numpy()

    with hm.activate('val'):
        log_metric(f'Validation {METRIC_TGB_LINKPROPPRED}', val_mrr)
    with hm.activate('test'):
        test_mrr = evaluate_parallel(
            test_loader, index, theta, n_train + n_val, 200, args.num_workers
        )
        log_metric(f'Test {METRIC_TGB_LINKPROPPRED}', test_mrr)
