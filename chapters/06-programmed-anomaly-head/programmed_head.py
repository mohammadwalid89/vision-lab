#!/usr/bin/env python3
"""Write an anomaly head in closed form and compare it with controls.

Input is a cached feature table (CLIP RN50 + ResNet18 embeddings and the
PatchCore/CLIP scalar scores of nine MVTec AD categories). No image is read and
no model is run.

Arms, all scored on the same rows with the same preprocessing:
  ZERO        same architecture, all weights zero
  PROGRAMMED  weights written by formula from fit-split class statistics; no
              optimizer, no gradient step
  PLACEBO     same shape and row norms, random defect direction
  TRAINED     same architecture fit by Adam on the same labelled rows
  PATCHCAL    per-category calibration of the existing PatchCore score
  CLIPCAL     per-category calibration of the existing CLIP score

Every arm's threshold is the 95th percentile of the same good calibration rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vision_lab.metrics import auc  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

SEED = 20261104
GATE = 20.0
TRAIN_STEPS = 300
TRAIN_LR = 0.02


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()


def bootstrap_auc(y, s, reps=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    pos_idx, neg_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                              rng.choice(neg_idx, len(neg_idx), replace=True)])
        vals.append(auc(y[idx], s[idx]))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def operating_point(y, s, thr, band) -> dict:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    pred = (s >= thr).astype(int)
    defect, good = y == 1, y == 0
    wrong_release = int(((pred == 0) & defect).sum())
    false_hold = int(((pred == 1) & good).sum())
    escalated = int(((s >= thr - band) & (s <= thr + band)).sum())
    return {"n": int(len(y)), "defects": int(defect.sum()), "goods": int(good.sum()),
            "wrong_releases": wrong_release,
            "wrong_release_rate": round(wrong_release / max(1, int(defect.sum())), 4),
            "false_holds": false_hold,
            "false_hold_rate": round(false_hold / max(1, int(good.sum())), 4),
            "human_escalation": escalated,
            "human_escalation_rate": round(escalated / max(1, len(y)), 4),
            "accuracy": round(float((pred == y).mean()), 4)}


def load_cohort(cohort_dir: Path):
    rows = [json.loads(line) for line in
            (cohort_dir / "heldout_ad_scores.jsonl").read_text().splitlines() if line.strip()]
    clip = np.load(cohort_dir / "clip_embeddings.npz")
    resnet = np.load(cohort_dir / "resnet_embeddings.npz")
    cats = sorted({r["category"] for r in rows})
    cat_index = {c: i for i, c in enumerate(cats)}
    feats, labels, roles, categories, shas, patch, clip_score = [], [], [], [], [], [], []
    for r in rows:
        sha = r["sha256"]
        if sha not in clip.files or sha not in resnet.files:
            raise SystemExit(f"cached feature missing for {sha}")
        z = np.concatenate([clip[sha].astype(np.float32), resnet[sha].astype(np.float32)])
        onehot = np.zeros(len(cats), np.float32)
        onehot[cat_index[r["category"]]] = 1.0
        feats.append(np.concatenate([z, onehot]))
        labels.append(1 if r["truth"] == "not_ok" else 0)
        roles.append(r["role"])
        categories.append(r["category"])
        shas.append(sha)
        patch.append(r["patch_score"])
        clip_score.append(r["clip_score"])
    return (np.stack(feats), np.array(labels, np.int64), np.array(roles, object),
            np.array(categories, object), cats, np.array(shas, object),
            np.array(patch, float), np.array(clip_score, float))


def leakage_audit(shas, protocol_path: Path) -> dict:
    """Overlap of the cohort with the rows of a role-labelled protocol file."""
    protocol = json.loads(protocol_path.read_text())
    by_role: dict[str, set] = {}
    for r in protocol["rows"]:
        by_role.setdefault(r["role"], set()).add(r["sha256"])
    out = {role: {"protocol_rows": len(s), "overlap_with_cohort": len(set(shas) & s)}
           for role, s in by_role.items()}
    out["final60_overlap"] = len(set(shas) & by_role.get("final_reserved", set()))
    out["gate1_overlap"] = len(set(shas) & by_role.get("gate1", set()))
    return out


def standardise(X, train_mask, n_cont):
    mu = X[train_mask].mean(0)
    sd = X[train_mask].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    mu[n_cont:], sd[n_cont:] = 0.0, 1.0  # one-hot block stays as is
    return ((X - mu) / sd).astype(np.float32), mu, sd


def _gated_units(Z, d, cats, direction, midpoint, spread):
    """Two signed ReLU units per category, gated by the category one-hot."""
    W1 = np.zeros((2 * len(cats), Z.shape[1]), np.float32)
    b1 = np.zeros(2 * len(cats), np.float32)
    w2 = np.zeros(2 * len(cats), np.float32)
    for c in range(len(cats)):
        for sign, row in ((1.0, 2 * c), (-1.0, 2 * c + 1)):
            W1[row, :d] = sign * direction[c]
            W1[row, d + c] = GATE
            b1[row] = -sign * midpoint[c] - GATE
            w2[row] = sign / spread[c]
    return W1, b1, w2


def _normalise_readout(Z, train_mask, W1, b1, w2):
    logit = np.maximum(Z @ W1.T + b1, 0.0) @ w2
    scale = 1.0 / max(float(np.std(logit[train_mask])), 1e-6)
    return (w2 * scale).astype(np.float32)


def _robust_spread(x) -> float:
    return max(float(1.4826 * np.median(np.abs(x - np.median(x)))), 1e-3)


def build_programmed(Z, y, train_mask, cat, cats):
    """Closed-form weights from fit rows only.

    Per category c: u = unit(mean_defect - mean_normal), t = midpoint of the
    class means projected on u, s = robust spread of normals along u.
    Hidden units are relu(+/-(u.z - t)) gated by the category, readout +/-1/s.
    """
    d = Z.shape[1] - len(cats)
    direction, midpoint, spread, info = [], [], [], {}
    for name in cats:
        m = train_mask & (cat == name)
        zf, yf = Z[m, :d], y[m]
        mu_n, mu_a = zf[yf == 0].mean(0), zf[yf == 1].mean(0)
        u = mu_a - mu_n
        norm = float(np.linalg.norm(u))
        if norm < 1e-9:
            raise SystemExit(f"degenerate direction for {name}")
        u = (u / norm).astype(np.float32)
        t = float(0.5 * (mu_a + mu_n) @ u)
        s = _robust_spread(zf[yf == 0] @ u)
        direction.append(u)
        midpoint.append(t)
        spread.append(s)
        info[name] = {"raw_separation_norm": round(norm, 3), "midpoint_projection": round(t, 3),
                      "normal_spread": round(s, 3), "fit_rows": int(m.sum()),
                      "fit_defects": int((yf == 1).sum())}
    W1, b1, w2 = _gated_units(Z, d, cats, direction, midpoint, spread)
    return W1, b1, _normalise_readout(Z, train_mask, W1, b1, w2), np.float32(0.0), info


def build_placebo(Z, train_mask, cat, cats, seed=SEED + 7):
    """Same construction with a random direction; only fit normals are used."""
    d = Z.shape[1] - len(cats)
    rng = np.random.default_rng(seed)
    direction, midpoint, spread = [], [], []
    for name in cats:
        zf = Z[train_mask & (cat == name), :d]
        u = rng.standard_normal(d).astype(np.float32)
        u /= max(float(np.linalg.norm(u)), 1e-9)
        direction.append(u)
        midpoint.append(float(zf.mean(0) @ u))
        spread.append(_robust_spread(zf @ u))
    W1, b1, w2 = _gated_units(Z, d, cats, direction, midpoint, spread)
    return W1, b1, _normalise_readout(Z, train_mask, W1, b1, w2), np.float32(0.0)


def build_zero(n_in, cats):
    n = 2 * len(cats)
    return np.zeros((n, n_in), np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32), np.float32(0.0)


def build_trained(Z, y, train_mask, cats, steps=TRAIN_STEPS, lr=TRAIN_LR, seed=SEED):
    torch.manual_seed(seed)
    Xt = torch.from_numpy(Z[train_mask])
    yt = torch.from_numpy(y[train_mask].astype(np.float32))
    model = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], 2 * len(cats)),
                                torch.nn.ReLU(), torch.nn.Linear(2 * len(cats), 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    final = None
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(Xt).squeeze(1), yt)
        loss.backward()
        opt.step()
        final = float(loss.item())
    return (model[0].weight.detach().numpy().copy(), model[0].bias.detach().numpy().copy(),
            model[2].weight.detach().numpy().reshape(-1).copy(),
            np.float32(float(model[2].bias.detach().numpy().reshape(-1)[0])), final)


def zscore_per_category(score, cat, base_mask, cats):
    out = np.zeros(len(score), float)
    for name in cats:
        base = score[base_mask & (cat == name)]
        med = float(np.median(base))
        mad = 1.4826 * float(np.median(np.abs(base - med)))
        mad = mad if mad > 1e-9 else max(float(np.std(base)), 1e-9)
        sel = cat == name
        out[sel] = (score[sel] - med) / mad
    return out


def run(cohort_dir: Path, protocol: Path | None) -> dict:
    t0, c0 = time.perf_counter(), time.process_time()
    X, y, role, cat, cats, shas, patch, clip_score = load_cohort(cohort_dir)
    n_cont = X.shape[1] - len(cats)
    train_mask = np.isin(role, ["fit", "support_anomaly"])
    cal_mask = role == "calibration"
    test_mask = np.isin(role, ["test_good", "test_anomaly"])
    if (cal_mask & (y == 1)).any():
        raise SystemExit("calibration role must be good-only")
    Z, _, _ = standardise(X, train_mask, n_cont)

    audit = None
    if protocol is not None:
        audit = leakage_audit(shas, protocol)
        if audit["final60_overlap"] or audit["gate1_overlap"]:
            raise SystemExit(f"cohort overlaps reserved rows: {audit}")

    counts = {"train_rows": int(train_mask.sum()),
              "train_good": int((train_mask & (y == 0)).sum()),
              "train_defect": int((train_mask & (y == 1)).sum()),
              "calibration_rows": int(cal_mask.sum()), "test_rows": int(test_mask.sum()),
              "test_good": int((test_mask & (y == 0)).sum()),
              "test_defect": int((test_mask & (y == 1)).sum())}

    def evaluate_arm(score, extra):
        started = time.process_time()
        cal = score[cal_mask]
        thr = float(np.percentile(np.sort(cal), 95))
        band = 0.1 * float(np.percentile(cal, 95) - np.percentile(cal, 5))
        s, yy = score[test_mask], y[test_mask]
        per_cat = {}
        for cname in cats:
            m = test_mask & (cat == cname)
            per_cat[cname] = {
                "auc": round(auc(y[m], score[m]), 4), "n": int(m.sum()),
                "defects": int((y[m] == 1).sum()), "goods": int((y[m] == 0).sum()),
                "wrong_releases": int(((score[m] < thr) & (y[m] == 1)).sum()),
                "false_holds": int(((score[m] >= thr) & (y[m] == 0)).sum())}
        return {"threshold_calibrated_q95": round(thr, 6), "escalation_band": round(band, 6),
                "test_auc": round(auc(yy, s), 4),
                "test_auc_ci95_bootstrap": bootstrap_auc(yy, s),
                "operating_point": operating_point(yy, s, thr, band),
                "per_category": per_cat,
                "worst_category_auc": round(min(v["auc"] for v in per_cat.values()), 4),
                "cpu_s_scoring": round(time.process_time() - started, 4), **extra}

    arms: dict[str, dict] = {}

    def add_net_arm(name, W1, b1, w2, b2, extra=None):
        started = time.process_time()
        score = np.maximum(Z @ W1.T + b1, 0.0) @ w2 + float(b2)
        build_cpu = time.process_time() - started
        w_all = np.concatenate([W1.reshape(-1), b1, w2, np.array([b2], np.float32)])
        meta = {"parameters": int(w_all.size),
                "nonzero_weight_fraction": round(float(np.count_nonzero(w_all) / w_all.size), 6),
                "weight_sha256": sha_array(w_all),
                "build_plus_forward_cpu_s": round(build_cpu, 4)}
        arms[name] = {**meta, **evaluate_arm(score, extra or {})}

    Wz, bz, wz, b2z = build_zero(Z.shape[1], cats)
    add_net_arm("ZERO_untrained", Wz, bz, wz, b2z, {"label_use": "none"})

    started = time.process_time()
    Wp, bp, wp, b2p, prog_info = build_programmed(Z, y, train_mask, cat, cats)
    add_net_arm("PROGRAMMED_closed_form", Wp, bp, wp, b2p, {
        "label_use": "fit-split class-conditional means and directions; no gradient, no optimizer",
        "program_build_cpu_s": round(time.process_time() - started, 4),
        "per_category_program": prog_info})

    Wr, br, wr, b2r = build_placebo(Z, train_mask, cat, cats)
    add_net_arm("PLACEBO_random_direction", Wr, br, wr, b2r,
                {"label_use": "fit-split normals only, for centering"})

    started = time.process_time()
    Wt, bt, wt, b2t, final_loss = build_trained(Z, y, train_mask, cats)
    add_net_arm("TRAINED_adam_mlp", Wt, bt, wt, b2t, {
        "label_use": "same labelled rows, Adam", "steps": TRAIN_STEPS, "lr": TRAIN_LR,
        "final_train_loss": round(final_loss, 5),
        "train_cpu_s": round(time.process_time() - started, 4)})

    for name, raw, note in (("PATCHCAL", patch, "existing PatchCore memory-bank score"),
                            ("CLIPCAL", clip_score, "existing CLIP RN50 contrast score")):
        started = time.process_time()
        z = zscore_per_category(raw, cat, train_mask | cal_mask, cats)
        cpu = time.process_time() - started
        arms[name] = {"parameters": 0, "nonzero_weight_fraction": 0.0,
                      "weight_sha256": sha_array(z.astype(np.float32)),
                      "score_source": note, "cpu_s_scoring": round(cpu, 4),
                      **evaluate_arm(z, {})}

    def compare(a, b):
        A, B = arms[a], arms[b]
        op_a, op_b = A["operating_point"], B["operating_point"]
        return {"test_auc": {"a": A["test_auc"], "b": B["test_auc"],
                             "delta_b_minus_a": round(B["test_auc"] - A["test_auc"], 4)},
                "wrong_releases": {"a": op_a["wrong_releases"], "b": op_b["wrong_releases"],
                                   "delta_b_minus_a": op_b["wrong_releases"] - op_a["wrong_releases"]},
                "false_holds": {"a": op_a["false_holds"], "b": op_b["false_holds"],
                                "delta_b_minus_a": op_b["false_holds"] - op_a["false_holds"]},
                "worst_category_auc": {"a": A["worst_category_auc"], "b": B["worst_category_auc"]}}

    return {
        "schema": "vision-weight-edit-pilot1-1",
        "task": "surface_anomaly_decision (ok/not_ok), MVTec AD, 9 categories",
        "cohort": {
            "rows_sha256": sha_file(cohort_dir / "heldout_ad_scores.jsonl"),
            "clip_embeddings_sha256": sha_file(cohort_dir / "clip_embeddings.npz"),
            "resnet_embeddings_sha256": sha_file(cohort_dir / "resnet_embeddings.npz"),
            "n": int(len(y)), "categories": cats, "feature_dim": int(X.shape[1]),
            "counts": counts, "leakage_audit": audit,
            "prior_exposure": "all rows were scored by an earlier experiment; the test rows "
                              "also overlap an earlier surface experiment, so this is not blind"},
        "architecture": {
            "input": "standardised CLIP(1024) + ResNet(2560) + category one-hot(9)",
            "hidden": "2 signed ReLU units per category, category-gated",
            "parameters_per_neural_arm": int(arms["PROGRAMMED_closed_form"]["parameters"])},
        "arms": arms,
        "comparisons": {
            "PROGRAMMED_vs_TRAINED": compare("PROGRAMMED_closed_form", "TRAINED_adam_mlp"),
            "PROGRAMMED_vs_ZERO": compare("ZERO_untrained", "PROGRAMMED_closed_form"),
            "PROGRAMMED_vs_PLACEBO": compare("PLACEBO_random_direction", "PROGRAMMED_closed_form"),
            "PROGRAMMED_vs_PATCHCAL": compare("PATCHCAL", "PROGRAMMED_closed_form"),
            "TRAINED_vs_PATCHCAL": compare("PATCHCAL", "TRAINED_adam_mlp")},
        "resources": {"cpu_s_total": round(time.process_time() - c0, 3),
                      "wall_s_total": round(time.perf_counter() - t0, 3), "threads": 1},
        "notes": [
            "The programmed arm is not zero-shot: it is built from fit-split label statistics.",
            "The trained arm gets the same labelled rows.",
            "ZERO and PLACEBO are controls, not methods.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort-dir", type=Path,
                        default=Path(os.environ.get("VISION_COHORT_DIR", "cohort")),
                        help="folder with heldout_ad_scores.jsonl, clip_embeddings.npz, "
                             "resnet_embeddings.npz (written by chapter 02 score_heldout_ad.py)")
    parser.add_argument("--protocol", type=Path, default=None,
                        help="optional role-labelled protocol JSON for the reserved-row overlap check")
    parser.add_argument("--out", type=Path, default=Path("results/summary.json"))
    args = parser.parse_args()

    result = run(args.cohort_dir, args.protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for name, arm in result["arms"].items():
        print(f"{name:26s} auc={arm['test_auc']:.4f} ci={arm['test_auc_ci95_bootstrap']} "
              f"{arm['operating_point']['wrong_releases']} wrong releases, "
              f"{arm['operating_point']['false_holds']} false holds")


if __name__ == "__main__":
    main()
