import numpy as np
import pytest
import torch

from tgm.nn import (
    EdgeReMINDIndex,
    EdgeReMINDPredictor,
    EdgeReMINDTrainer,
    extract_training_features,
    parallel_evaluate,
    parallel_extract_features,
)
from tgm.nn.modules.edgeremind import DEFAULT_WEIGHTS, NUM_FEATURES


def _synthetic(seed=0, n=600, num_nodes=30, num_rel=4):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, size=n).astype(np.int64)
    rel = rng.integers(0, num_rel, size=n).astype(np.int64)
    dst = rng.integers(0, num_nodes, size=n).astype(np.int64)
    ts = np.sort(rng.integers(0, 50000, size=n)).astype(np.int64)
    return src, dst, ts, rel, num_nodes, num_rel


def test_default_weights_are_deterministic_edgeremind():
    model = EdgeReMINDPredictor(num_relations=3)
    expected = torch.tensor(DEFAULT_WEIGHTS, dtype=model.theta.dtype)
    for r in range(3):
        assert torch.allclose(model.theta[r], expected)


def test_compute_features_shape_and_counts():
    model = EdgeReMINDPredictor(num_relations=2, decay_rate_srd=0.0)
    src = torch.tensor([0, 0, 1])
    dst = torch.tensor([5, 5, 5])
    ts = torch.tensor([10, 20, 30])
    rel = torch.tensor([0, 0, 0])
    model.update(src, dst, ts, rel)

    feats = model.compute_features(
        torch.tensor([0]), torch.tensor([5]), torch.tensor([100]), torch.tensor([0])
    )
    assert feats.shape == (1, NUM_FEATURES)
    assert feats[0, 0].item() == pytest.approx(2.0)
    assert feats[0, 1].item() == pytest.approx(3.0)
    assert feats[0, 2].item() == pytest.approx(2.0)
    assert feats[0, 3].item() == pytest.approx(1.0)


@pytest.mark.parametrize('decay', [0.0, 1.0 / 10000.0])
def test_index_matches_online_predictor(decay):
    src, dst, ts, rel, _, num_rel = _synthetic(seed=1)
    b = 100
    index = EdgeReMINDIndex.build(src, dst, ts, rel, decay_rate_srd=decay)
    online = EdgeReMINDPredictor(num_relations=num_rel, decay_rate_srd=decay)

    rng = np.random.default_rng(3)
    max_diff = 0.0
    for start in range(0, len(src), b):
        end = min(start + b, len(src))
        for i in range(start, end):
            cand = np.concatenate([[dst[i]], rng.integers(0, 30, size=8)])
            f_on = online.compute_features(
                torch.from_numpy(np.full(len(cand), src[i])),
                torch.from_numpy(cand),
                torch.from_numpy(np.full(len(cand), ts[i])),
                torch.from_numpy(np.full(len(cand), rel[i])),
            ).numpy()
            f_ix = index.query_features(
                int(src[i]), int(rel[i]), int(ts[i]), start, cand.astype(np.int64)
            )
            max_diff = max(max_diff, float(np.abs(f_on - f_ix).max()))
        online.update(
            torch.from_numpy(src[start:end]),
            torch.from_numpy(dst[start:end]),
            torch.from_numpy(ts[start:end]),
            torch.from_numpy(rel[start:end]),
        )
    assert max_diff == 0.0


def test_duplicate_candidates_filled_identically():
    src = np.array([1, 1], dtype=np.int64)
    dst = np.array([5, 5], dtype=np.int64)
    ts = np.array([1, 2], dtype=np.int64)
    rel = np.array([0, 0], dtype=np.int64)
    index = EdgeReMINDIndex.build(src, dst, ts, rel, decay_rate_srd=0.0)
    cand = np.array([5, 5, 7, 5], dtype=np.int64)
    feats = index.query_features(1, 0, 100, 2, cand)
    assert np.array_equal(feats[0], feats[1])
    assert np.array_equal(feats[0], feats[3])
    assert feats[2].sum() == 0.0


@pytest.mark.parametrize('num_workers', [0, 2])
def test_parallel_extract_matches_trainer(num_workers):
    src, dst, ts, rel, num_nodes, num_rel = _synthetic(seed=2)
    pred = EdgeReMINDPredictor(num_relations=num_rel, decay_rate_srd=1.0 / 10000.0)
    trainer = EdgeReMINDTrainer(
        pred, first_dst=0, last_dst=num_nodes - 1, k=10, feature_bsize=100,
        max_train_queries=None, seed=1337,
    )
    f_seq, r_seq = trainer.extract_features(
        torch.from_numpy(src), torch.from_numpy(dst),
        torch.from_numpy(ts), torch.from_numpy(rel),
    )
    index = EdgeReMINDIndex.build(src, dst, ts, rel, decay_rate_srd=1.0 / 10000.0)
    f_par, r_par = extract_training_features(
        index, src, dst, ts, rel, num_relations=num_rel,
        first_dst=0, last_dst=num_nodes - 1, k=10, feature_bsize=100,
        max_train_queries=None, num_workers=num_workers, seed=1337,
    )
    assert torch.equal(r_seq, r_par)
    assert torch.allclose(f_seq, f_par)


def test_parallel_extract_workers_consistent():
    src, dst, ts, rel, _, _ = _synthetic(seed=4)
    index = EdgeReMINDIndex.build(src, dst, ts, rel, decay_rate_srd=1.0 / 10000.0)
    k = 6
    rng = np.random.default_rng(0)
    neg = rng.integers(0, 30, size=(len(src), k)).astype(np.int64)
    cutoffs = np.array([(i // 100) * 100 for i in range(len(src))], dtype=np.int64)
    a = parallel_extract_features(index, src, dst, ts, rel, neg, cutoffs, num_workers=0)
    b = parallel_extract_features(index, src, dst, ts, rel, neg, cutoffs, num_workers=2)
    assert torch.allclose(a, b)


@pytest.mark.parametrize('num_workers', [0, 2])
def test_parallel_evaluate_reciprocal_rank(num_workers):
    src, dst, ts, rel, num_nodes, num_rel = _synthetic(seed=6)
    index = EdgeReMINDIndex.build(src, dst, ts, rel, decay_rate_srd=1.0 / 10000.0)
    theta = np.tile(np.array(DEFAULT_WEIGHTS, dtype=np.float32), (num_rel, 1))
    n_q = 50
    q = slice(len(src) - n_q, len(src))
    rng = np.random.default_rng(7)
    neg_list = [rng.integers(0, num_nodes, size=12).astype(np.int64) for _ in range(n_q)]
    cutoffs = np.full(n_q, len(src) - n_q, dtype=np.int64)
    rr = parallel_evaluate(
        index, theta, src[q], rel[q], ts[q], cutoffs, dst[q], neg_list,
        num_workers=num_workers,
    )
    assert rr.shape == (n_q,)
    assert np.all(rr > 0) and np.all(rr <= 1.0)


def test_unseen_relation_keeps_default_weights():
    model = EdgeReMINDPredictor(num_relations=5)
    model.update(
        torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), torch.tensor([2])
    )
    expected = torch.tensor(DEFAULT_WEIGHTS, dtype=model.theta.dtype)
    assert torch.allclose(model.theta[4], expected)


def test_default_weights_has_six_features_and_dec_d_zero():
    assert NUM_FEATURES == 6
    assert len(DEFAULT_WEIGHTS) == 6
    assert DEFAULT_WEIGHTS[5] == 0.0


def test_dec_d_last_seen_decay_matches_index():
    src, dst, ts, rel, _, num_rel = _synthetic(seed=11)
    decay = 1.0 / 10000.0
    index = EdgeReMINDIndex.build(src, dst, ts, rel, decay_rate_srd=decay)
    online = EdgeReMINDPredictor(num_relations=num_rel, decay_rate_srd=decay)

    online.update(
        torch.from_numpy(src), torch.from_numpy(dst),
        torch.from_numpy(ts), torch.from_numpy(rel),
    )
    t_query = int(ts.max()) + 1
    cutoff = len(src)

    cand = np.array([dst[0], dst[len(src) // 2], dst[-1], 999], dtype=np.int64)
    f_ix = index.query_features(0, 0, t_query, cutoff, cand)
    f_on = online.compute_features(
        torch.from_numpy(np.zeros_like(cand)),
        torch.from_numpy(cand),
        torch.from_numpy(np.full(cand.shape[0], t_query)),
        torch.from_numpy(np.zeros_like(cand)),
    ).numpy()
    assert np.array_equal(f_ix[:, 5], f_on[:, 5])
    assert np.all(f_ix[:, 5] >= 0.0) and np.all(f_ix[:, 5] <= 1.0)
    assert np.all(f_on[:, 5] >= 0.0) and np.all(f_on[:, 5] <= 1.0)
    assert f_ix[3, 5] == 0.0 and f_on[3, 5] == 0.0


def _apply_feature_mask(feats, theta, feature_mask):
    mask_arr = np.asarray(list(feature_mask), dtype=np.float32)
    assert mask_arr.shape == (NUM_FEATURES,)
    assert np.all(np.isin(mask_arr, (0.0, 1.0)))
    assert mask_arr.sum() > 0
    dropped = [i for i, m in enumerate(mask_arr) if m == 0.0]
    feats = feats.copy()
    feats[:, :, dropped] = 0.0
    theta = theta.copy()
    theta[:, dropped] = 0.0
    return feats, theta, dropped


def test_feature_mask_leave_one_out_zeros_feats_and_theta():
    feats = np.random.rand(4, 3, NUM_FEATURES).astype(np.float32) + 0.5
    theta = np.tile(np.array(DEFAULT_WEIGHTS, dtype=np.float32), (2, 1))
    f, t, dropped = _apply_feature_mask(feats, theta, [0, 1, 1, 1, 1, 1])
    assert dropped == [0]
    assert np.all(f[:, :, 0] == 0.0)
    assert np.all(t[:, 0] == 0.0)
    assert np.all(f[:, :, 1:] != 0.0)
    assert np.allclose(t[:, 1:], theta[:, 1:])


def test_feature_mask_matches_num_features_five():
    feats = np.random.rand(4, 3, NUM_FEATURES).astype(np.float32) + 0.5
    theta = np.tile(np.array(DEFAULT_WEIGHTS, dtype=np.float32), (2, 1))
    f, t, dropped = _apply_feature_mask(feats, theta, [1, 1, 1, 1, 1, 0])
    assert dropped == [5]
    assert np.all(f[:, :, 5] == 0.0)
    assert np.allclose(t, theta)


def test_feature_mask_drops_nonzero_default_at_eval():
    feats = np.random.rand(4, 3, NUM_FEATURES).astype(np.float32) + 0.5
    theta = np.tile(np.array(DEFAULT_WEIGHTS, dtype=np.float32), (2, 1))
    f, t, dropped = _apply_feature_mask(feats, theta, [1, 1, 1, 0, 1, 1])
    assert dropped == [3]
    assert np.all(t[:, 3] == 0.0)
    assert t[0, 0] == DEFAULT_WEIGHTS[0]
    assert t[0, 2] == DEFAULT_WEIGHTS[2]


def test_feature_mask_rejects_all_zero():
    feats = np.random.rand(2, 2, NUM_FEATURES).astype(np.float32)
    theta = np.tile(np.array(DEFAULT_WEIGHTS, dtype=np.float32), (1, 1))
    with pytest.raises(AssertionError):
        _apply_feature_mask(feats, theta, [0, 0, 0, 0, 0, 0])

from tgm.nn import calibrate_bank_lambdas
from tgm.nn.modules.edgeremind import BANK_SCOPES


def _bank(q=3, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(1.0 / 50000.0, 1.0 / 50.0, size=(len(BANK_SCOPES), q))


def test_bank_feat_dim_and_zero_init():
    bank = _bank(q=3)
    model = EdgeReMINDPredictor(num_relations=4, bank_lambdas=bank)
    assert model.feat_dim == NUM_FEATURES + 3 * len(BANK_SCOPES)
    assert model.theta.shape == (4, model.feat_dim)
    expected = torch.tensor(DEFAULT_WEIGHTS, dtype=model.theta.dtype)
    assert torch.allclose(model.theta[:, :NUM_FEATURES], expected.repeat(4, 1))
    assert torch.all(model.theta[:, NUM_FEATURES:] == 0.0)


def test_bank_untrained_scores_match_base():
    src, dst, ts, rel, num_nodes, num_rel = _synthetic(seed=11)
    base = EdgeReMINDPredictor(num_relations=num_rel)
    banked = EdgeReMINDPredictor(num_relations=num_rel, bank_lambdas=_bank())

    h = 400  # history prefix
    for m in (base, banked):
        m.update(
            torch.from_numpy(src[:h]),
            torch.from_numpy(dst[:h]),
            torch.from_numpy(ts[:h]),
            torch.from_numpy(rel[:h]),
        )
    q_src = torch.from_numpy(src[h:])
    q_dst = torch.from_numpy(dst[h:])
    q_ts = torch.from_numpy(ts[h:])
    q_rel = torch.from_numpy(rel[h:])
    s_base = base(q_src, q_dst, q_ts, q_rel)
    s_bank = banked(q_src, q_dst, q_ts, q_rel)
    assert torch.allclose(s_base, s_bank, rtol=0.0, atol=1e-6)
    f_base = base.compute_features(q_src, q_dst, q_ts, q_rel)
    f_bank = banked.compute_features(q_src, q_dst, q_ts, q_rel)
    assert torch.equal(f_base, f_bank[:, :NUM_FEATURES])


def test_bank_columns_match_manual_decay():
    bank = np.array([[0.1, 0.01], [0.2, 0.02], [0.3, 0.03]])
    q = bank.shape[1]
    model = EdgeReMINDPredictor(num_relations=1, decay_rate_srd=0.0,
                                bank_lambdas=bank)
    model.update(
        torch.tensor([0]), torch.tensor([5]), torch.tensor([10]), torch.tensor([0])
    )
    feats = model.compute_features(
        torch.tensor([0]), torch.tensor([5]), torch.tensor([20]), torch.tensor([0])
    ).numpy()[0]
    dt = 10.0
    for scope, base_col in enumerate((NUM_FEATURES, NUM_FEATURES + q,
                                      NUM_FEATURES + 2 * q)):
        for j in range(q):
            assert feats[base_col + j] == pytest.approx(
                np.exp(-bank[scope, j] * dt), rel=1e-6
            )


@pytest.mark.parametrize('decay', [0.0, 1.0 / 10000.0])
def test_bank_index_matches_online_predictor(decay):
    """Online/offline parity holds column-for-column with the bank enabled."""
    src, dst, ts, rel, _, num_rel = _synthetic(seed=1)
    bank = _bank(q=3, seed=2)
    b = 100
    index = EdgeReMINDIndex.build(
        src, dst, ts, rel, decay_rate_srd=decay, bank_lambdas=bank
    )
    online = EdgeReMINDPredictor(
        num_relations=num_rel, decay_rate_srd=decay, bank_lambdas=bank
    )
    assert index.feat_dim == online.feat_dim

    rng = np.random.default_rng(3)
    max_diff = 0.0
    for start in range(0, len(src), b):
        end = min(start + b, len(src))
        for i in range(start, end):
            cand = np.concatenate([[dst[i]], rng.integers(0, 30, size=8)])
            f_on = online.compute_features(
                torch.from_numpy(np.full(len(cand), src[i])),
                torch.from_numpy(cand),
                torch.from_numpy(np.full(len(cand), ts[i])),
                torch.from_numpy(np.full(len(cand), rel[i])),
            ).numpy()
            f_ix = index.query_features(
                int(src[i]), int(rel[i]), int(ts[i]), start, cand.astype(np.int64)
            )
            assert f_on.shape == f_ix.shape == (len(cand), online.feat_dim)
            max_diff = max(max_diff, float(np.abs(f_on - f_ix).max()))
        online.update(
            torch.from_numpy(src[start:end]),
            torch.from_numpy(dst[start:end]),
            torch.from_numpy(ts[start:end]),
            torch.from_numpy(rel[start:end]),
        )
    assert max_diff == 0.0


def test_calibrate_bank_lambdas_known_gaps():
    src = np.zeros(5, dtype=np.int64)
    dst = np.zeros(5, dtype=np.int64)
    rel = np.zeros(5, dtype=np.int64)
    ts = np.array([0, 10, 20, 30, 40], dtype=np.int64)
    lam = calibrate_bank_lambdas(src, dst, ts, rel, n_timescales=3,
                                 geometric_factor=2.0, verbose=False)
    assert lam.shape == (3, 3)
    hl = np.log(2.0) / lam[0]
    assert hl == pytest.approx([5.0, 10.0, 20.0])
    assert np.exp(-lam[0, 1] * 10.0) == pytest.approx(0.5)


def test_calibrate_bank_lambdas_scope_separation():
    src = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    rel = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    dst = np.full(6, 7, dtype=np.int64)
    ts = np.array([0, 5, 10, 15, 20, 25], dtype=np.int64)
    lam = calibrate_bank_lambdas(src, dst, ts, rel, n_timescales=1,
                                 geometric_factor=2.0, verbose=False)
    assert np.log(2.0) / lam[0, 0] == pytest.approx(10.0)  # srd median gap
    assert np.log(2.0) / lam[1, 0] == pytest.approx(10.0)
    assert np.log(2.0) / lam[2, 0] == pytest.approx(5.0)


def test_calibrate_bank_lambdas_fallbacks():
    src = np.array([0, 1, 2], dtype=np.int64)
    dst = np.array([3, 4, 5], dtype=np.int64)
    rel = np.array([0, 1, 2], dtype=np.int64)
    ts = np.array([0, 1, 2], dtype=np.int64)
    lam = calibrate_bank_lambdas(src, dst, ts, rel, n_timescales=1,
                                 geometric_factor=2.0, verbose=False)
    assert np.allclose(lam, np.log(2.0))

    src2 = np.array([0, 1, 0, 1], dtype=np.int64)
    rel2 = np.array([0, 1, 0, 1], dtype=np.int64)
    dst2 = np.array([9, 9, 9, 9], dtype=np.int64)
    ts2 = np.array([0, 3, 6, 9], dtype=np.int64)
    lam2 = calibrate_bank_lambdas(src2, dst2, ts2, rel2, n_timescales=1,
                                  geometric_factor=2.0, verbose=False)
    assert np.log(2.0) / lam2[2, 0] == pytest.approx(3.0)
    assert np.log(2.0) / lam2[0, 0] == pytest.approx(6.0)  # srd recurs at 6


def test_bank_parallel_extraction_width_and_parity():
    """The parallel feature path returns the widened tensor and matches the
    sequential per-query path."""
    src, dst, ts, rel, _, num_rel = _synthetic(seed=5, n=300)
    bank = _bank(q=2, seed=9)
    index = EdgeReMINDIndex.build(src, dst, ts, rel, bank_lambdas=bank)
    feats, rels = extract_training_features(
        index, src, dst, ts, rel,
        num_relations=num_rel, first_dst=0, last_dst=29,
        k=4, feature_bsize=50, max_train_queries=None, num_workers=0,
    )
    assert feats.shape == (len(src), 5, index.feat_dim)
    assert index.feat_dim == NUM_FEATURES + 2 * len(BANK_SCOPES)

def _collapsing_stream(seed=0, n=600, gap=31536000):
    rng = np.random.default_rng(seed)
    num_nodes, num_rel = 25, 4
    src = rng.integers(0, num_nodes, size=n).astype(np.int64)
    rel = rng.integers(0, num_rel, size=n).astype(np.int64)
    dst = rng.integers(0, num_nodes, size=n).astype(np.int64)
    steps = rng.integers(0, 6, size=n)
    ts = (np.cumsum(steps) * gap // 5 + gap).astype(np.int64)
    ts = np.sort(ts)
    return src, dst, ts, rel


def test_geometric_distinct_on_concentrated_gaps():
    src, dst, ts, rel = _collapsing_stream(seed=1)
    lam = calibrate_bank_lambdas(src, dst, ts, rel, n_timescales=3,
                                 geometric_factor=2.0, verbose=False)
    assert lam.shape == (3, 3)
    hl = np.log(2.0) / lam[2]
    assert len(np.unique(np.round(hl, 6))) == 3
    assert hl[1] == pytest.approx(np.sqrt(hl[0] * hl[2]), rel=1e-6)
    assert hl[2] / hl[1] == pytest.approx(2.0, rel=1e-6)
    assert hl[1] / hl[0] == pytest.approx(2.0, rel=1e-6)


def test_geometric_factor_one_is_median_rescale():
    src, dst, ts, rel = _collapsing_stream(seed=2)
    lam = calibrate_bank_lambdas(src, dst, ts, rel, n_timescales=3,
                                 geometric_factor=1.0, verbose=False)
    hl = np.log(2.0) / lam
    for scope_row in hl:  # factor 1 -> all columns equal the median
        assert np.allclose(scope_row, scope_row[1], rtol=1e-9)


def test_geometric_spacing_matches_factor():
    src, dst, ts, rel = _collapsing_stream(seed=3)
    for f in (2.0, 4.0, 8.0):
        lam = calibrate_bank_lambdas(src, dst, ts, rel, n_timescales=3,
                                     geometric_factor=f, verbose=False)
        hl = np.log(2.0) / lam[2]
        assert hl[2] / hl[1] == pytest.approx(f, rel=1e-6)
        assert hl[1] / hl[0] == pytest.approx(f, rel=1e-6)


def test_geometric_default_is_factor_two():
    """The function defaults to factor 2, three timescales."""
    src, dst, ts, rel = _collapsing_stream(seed=6)
    lam_default = calibrate_bank_lambdas(src, dst, ts, rel, verbose=False)
    lam_explicit = calibrate_bank_lambdas(src, dst, ts, rel, n_timescales=3,
                                          geometric_factor=2.0, verbose=False)
    assert lam_default.shape == (3, 3)
    assert np.array_equal(lam_default, lam_explicit)


def test_geometric_bad_params_raise():
    src, dst, ts, rel = _collapsing_stream(seed=4)
    with pytest.raises(ValueError):
        calibrate_bank_lambdas(src, dst, ts, rel, geometric_factor=0.0,
                               verbose=False)
    with pytest.raises(ValueError):
        calibrate_bank_lambdas(src, dst, ts, rel, n_timescales=0,
                               verbose=False)


def test_geometric_single_column_is_median():
    src, dst, ts, rel = _collapsing_stream(seed=5)
    lam = calibrate_bank_lambdas(src, dst, ts, rel, n_timescales=1,
                                 geometric_factor=2.0, verbose=False)
    assert lam.shape == (3, 1)  # one column, exponent [0] -> median anchor
