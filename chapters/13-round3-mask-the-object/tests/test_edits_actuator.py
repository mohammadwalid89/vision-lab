import numpy as np
import pytest

from vision_lab.wsp import edits as E
from vision_lab.wsp.actuator import head_actuator, module_actuator
from vision_lab.wsp.heads import init_mlp, mlp_copy, mlp_forward


@pytest.fixture
def mlp():
    return init_mlp(24, 8, seed=3)


@pytest.fixture
def x():
    return np.random.default_rng(0).normal(size=(50, 24))


def test_six_noops_preserve_function(mlp, x):
    suite = E.noop_suite(mlp)
    assert tuple(suite) == E.NOOPS
    for name, (ref, q) in suite.items():
        assert np.max(np.abs(mlp_forward(q, x) - mlp_forward(ref, x))) <= 1e-12, name


def test_function_preserving_noops_change_many_weights(mlp):
    """The WSR false-positive trap: big weight diffs, zero behaviour change."""
    suite = E.noop_suite(mlp)
    for name in ("unit_permutation", "tanh_sign_flip"):
        ref, q = suite[name]
        assert E.weight_diff_stats(ref, q)["frac_changed"] > 0.3, name
    for name in ("identity", "scale_one", "zero_delta", "dtype_roundtrip"):
        ref, q = suite[name]
        assert E.weight_diff_stats(ref, q)["n_changed"] == 0, name


def test_deletion_and_random_deletion_counts(mlp):
    r = E.n_deletions(0.05, mlp["W1"].size)
    assert r == round(0.05 * 192)
    keep = E.random_deletion_keep(mlp["W1"].shape, r, seed=1)
    assert int((~keep).sum()) == r
    assert np.array_equal(keep, E.random_deletion_keep(mlp["W1"].shape, r, seed=1))
    W = E.delete(mlp["W1"], keep)
    assert np.all(W[~keep] == 0) and np.array_equal(W[keep], mlp["W1"][keep])
    assert E.n_deletions(0.05, 49152) == 2458                     # slot-3 budget on 1536x32


def test_direction_removal_and_noise(mlp):
    v = np.random.default_rng(1).normal(size=24)
    W = E.remove_direction(mlp["W1"], v)
    assert np.allclose(v @ W, 0, atol=1e-12)
    Wn = E.norm_matched_noise(mlp["W1"], 0.7, seed=2)
    assert np.linalg.norm(Wn - mlp["W1"]) == pytest.approx(0.7)
    assert np.allclose(E.scale_rows(mlp["W1"], [0, 1], 2.0)[:2], 2 * mlp["W1"][:2])


def test_head_actuator_verdicts(mlp, x):
    assert head_actuator(mlp, mlp_copy(mlp), x)["verdict"] == "NO_EDIT"
    keep = E.random_deletion_keep(mlp["W1"].shape, 20, seed=0)
    q = mlp_copy(mlp)
    q["W1"] = E.delete(q["W1"], keep)
    assert head_actuator(mlp, q, x)["verdict"] == "ACTIVE"
    _, perm = E.noop_suite(mlp)["unit_permutation"]
    assert head_actuator(mlp, perm, x)["verdict"] == "INERT"       # weights+hidden move, output does not
    q2 = mlp_copy(mlp)
    q2["W2"] = q2["W2"] * 2  # edit beside the claimed W1 path
    assert head_actuator(mlp, q2, x)["verdict"] == "WRONG_PATH"


def test_module_actuator_detects_wrong_path():
    """Known answer for the W3 failure: editing weights the forward pass never uses."""
    import torch
    torch.manual_seed(0)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(4, 3)
            self.unused = torch.nn.Linear(4, 3)        # instantiated but off the path
            self.out = torch.nn.Linear(3, 1)

        def forward(self, x):
            return self.out(torch.tanh(self.used(x)))

    net = Net()
    xin = torch.randn(5, 4)

    def edit_used(m):
        with torch.no_grad():
            m.used.weight.mul_(0.5)

    def edit_unused(m):
        with torch.no_grad():
            m.unused.weight.mul_(0.5)

    assert module_actuator(net, "used", xin, edit_used)["verdict"] == "ACTIVE"
    assert module_actuator(net, "used", xin, edit_unused)["verdict"] == "WRONG_PATH"
