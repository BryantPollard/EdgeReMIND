import pytest


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['thgl-software'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=4',
        '--mem=16G',
        '--time=2:00:00',
    ]
)
def test_edgeremind_linkprop_pred_thgl(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/thgl/edgeremind.py" \
    --dataset {dataset} \
    --num-workers 2"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['thgl-software'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=8',
        '--mem=16G',
        '--time=1:30:00',
    ]
)
def test_edgeremind_linkprop_pred_thgl_parallel(slurm_job_runner, dataset):
    """Test EdgeReMIND with multiple workers to verify parallel processing."""
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/thgl/edgeremind.py" \
    --dataset {dataset} \
    --num-workers 4"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


# Removed test_edgeremind_no_train_thgl - --no-train argument does not exist


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tkgl-smallpedia'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=4',
        '--mem=16G',
        '--time=4:00:00',
    ]
)
def test_edgeremind_linkprop_pred_tkgl(slurm_job_runner, dataset):
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/tkgl/edgeremind.py" \
    --dataset {dataset} \
    --num-workers 2"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


@pytest.mark.integration
@pytest.mark.parametrize('dataset', ['tkgl-smallpedia'])
@pytest.mark.slurm(
    resources=[
        '--partition=main',
        '--cpus-per-task=8',
        '--mem=16G',
        '--time=3:30:00',
    ]
)
def test_edgeremind_linkprop_pred_tkgl_parallel(slurm_job_runner, dataset):
    """Test EdgeReMIND with multiple workers to verify parallel processing."""
    cmd = f"""
python "$ROOT_DIR/examples/linkproppred/tkgl/edgeremind.py" \
    --dataset {dataset} \
    --num-workers 4"""
    state = slurm_job_runner(cmd)
    assert state == 'COMPLETED'


# Removed test_edgeremind_no_train_tkgl - --no-train argument does not exist


# Removed test_edgeremind_with_bank_tkgl - --bank argument does not exist (bank is always enabled)


# Removed test_edgeremind_without_bank_tkgl - --bank argument does not exist
