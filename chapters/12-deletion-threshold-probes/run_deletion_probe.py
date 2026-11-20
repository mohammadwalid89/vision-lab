"""Deletion-mask probe: which factor lets a weight edit remove the marker shortcut?

Factors: deletion-only versus signed change (F1), and how many weights may change
(F2). Heads: the Round 1 linear head A, and a small 1536-32-1 MLP. DEV only;
three seeds; scored with the Round 1 scorer and six-gate arithmetic.

    python run_deletion_probe.py --wsp-root <round-1 tree> --out-dir results_full/
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from probe_lib import (
    BUDGETS,
    MLP_L2,
    MLP_LR,
    MLP_STEPS,
    PRIMARY_BUDGET,
    RANDOM_DRAWS,
    RETRAIN_LINEAR_STEPS,
    SEEDS,
    Data,
    LinearMask,
    MLPMask,
    logits_np,
    mlp_flat,
    mlp_forward_np,
    retrain_linear_head,
    sha256_of,
    train_mlp,
)

from vision_lab.wsp.legacy.dev_bars import seed_bars
from vision_lab.wsp.legacy.scorer import freeze_threshold, score

torch.set_num_threads(2)

GATES = dict(
    marker_relative_reduction_min=0.5,
    placebo_margin_min=0.0,
    recall_margin_min=-0.02,
    FPR_margin_min=-0.02,
    conflict_relative_reduction_min=0.3,
    same_direction_all_seeds=1.0,
)
# Published Round 1 iteration-1 metrics for seed 20261112; the probe stops if the
# reproduced base head does not match them.
PUBLISHED_A_SEED1 = dict(
    marker_flip_rate=0.7185247986434931,
    conflict_set_FP=1989,
    AUROC=0.8473383609854197,
    AP=0.5266897910583762,
)


def log(msg):
    print(f'[{time.strftime("%Y-%m-%dT%H:%M:%S")}] {msg}', flush=True)


def metrics_block(rows, scores, frozen, reference):
    t0 = time.perf_counter()
    m = score(
        rows,
        scores,
        frozen,
        reference,
        dict(n_images=len(rows), elapsed_seconds=time.perf_counter() - t0),
    )
    # The scorer needs a timing record; latency is not measured in this probe.
    return dict(m, ms_per_image=None)


def gates_for(cand, a, best_placebo):
    values = seed_bars(cand, a, None, None, best_placebo, 'V1')
    bars = {k: ('PASS' if values[k] >= GATES[k] - 1e-12 else 'FAIL') for k in GATES}
    return values, bars


def recal_block(rows, scores):
    """Non-gating diagnostic: Youden-J refit on clean DEV, then flip/recall/FPR."""
    y = np.array([r['label'] for r in rows])
    clean = np.array([r['variant'] == 'clean' for r in rows])
    s = scores[clean]
    yy = y[clean]
    order = np.argsort(s, kind='stable')
    ss = s[order]
    y2 = yy[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    removed = np.r_[0, np.cumsum(y2)][starts]
    n1 = int(y2.sum())
    n0 = len(y2) - n1
    tp = n1 - removed
    fp = n0 - (starts - removed)
    thresholds = np.r_[ss[starts], np.nextafter(ss[-1], np.inf)]
    ba = np.r_[(tp / n1 + 1 - fp / n0) / 2, 0.5]
    th = float(max(zip(ba, thresholds))[1])
    pred = scores >= th
    pairs = {}
    for i, r in enumerate(rows):
        if r['variant'] in ('clean', 'marker'):
            pairs.setdefault(r['group_id'], {})[r['variant']] = i
    flip = float(
        np.mean([pred[g['clean']] != pred[g['marker']] for g in pairs.values()])
    )
    return dict(
        threshold=th,
        marker_flip=flip,
        recall=float(pred[clean & (y == 1)].mean()),
        FPR=float(pred[clean & (y == 0)].mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--wsp-root',
        type=Path,
        required=True,
        help='Round 1/2 experiment tree (r1/, revision2/, DATA_MANIFEST_R1.json)',
    )
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument(
        '--preflight',
        action='store_true',
        help='reproduce the base head and stop before any arm',
    )
    parser.add_argument(
        '--skip-base-check',
        action='store_true',
        help='do not compare the base head with the published metrics '
        '(only for runs on synthetic data)',
    )
    args = parser.parse_args()
    root = args.wsp_root
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log('probe start')

    d = Data(root)
    # The k=32 lists must equal the ones Round 2 used.
    pub = json.loads((root / 'revision2/artifacts/shared/SUBSETS.json').read_text())
    for s in SEEDS:
        if d.subsets[s] != pub[str(s)]['32']:
            raise SystemExit(f'BLOCKED k=32 list differs for {s}')
    log('k=32 source lists reproduce the published subset exactly')

    trusted = {
        r['id']: r
        for r in json.loads((root / 'DATA_MANIFEST_R1.json').read_text())['images']
        if r['split'] == 'dev'
    }
    for r in d.dr:
        for key in ('group_id', 'category', 'label', 'variant', 'sets', 'marker'):
            if r.get(key) != trusted[r['id']].get(key):
                raise SystemExit(f'BLOCKED DEV row mismatch {r["id"]} {key}')
    log(f'DEV rows match trusted manifest: {len(d.dr)} views')
    Xdev = d.scaled(d.di)
    dr = d.dr
    clean_idx = np.array([i for i, r in enumerate(dr) if r['variant'] == 'clean'])

    results = []  # one dict per (head, budget, seed, arm)
    ref_rows = {}  # head -> seed -> A metrics
    timing = {'start': time.time()}

    log('=== linear head (R1 i1 A) ===')
    for seed in SEEDS:
        ci, mi, y = d.k_pairs(seed)
        xc = d.scaled(ci)
        xm = d.scaled(mi)
        x = np.concatenate([xc, xm], 0)
        yy = np.concatenate([y, y])
        a = d.A[seed]
        w, b = a[:-1], a[-1]
        frozen = d.thresholds[seed]
        a_scores = logits_np(Xdev, a)
        a_metrics = metrics_block(
            dr, a_scores, frozen, {r['id']: v for r, v in zip(dr, a_scores)}
        )
        ref_rows.setdefault('linear', {})[str(seed)] = a_metrics
        if seed == SEEDS[0]:
            if not args.skip_base_check:
                for key, val in PUBLISHED_A_SEED1.items():
                    if abs(a_metrics[key] - val) > 1e-9:
                        raise SystemExit(
                            f'BLOCKED base reproduction {key}: {a_metrics[key]} != {val}'
                        )
                log('base A_linear reproduction matches published R1 i1 metrics')
            if args.preflight:
                log('preflight ok (no arm executed)')
                return
        # Marker response on the TRAIN sources not used for the k-source fit.
        seen = set(r['group_id'] for r in d.subsets[seed])
        others = d.all_train_pair_ids(seen)
        oc = d.scaled([o[0] for o in others])
        om = d.scaled([o[1] for o in others])
        a_train_resp = float(np.mean(logits_np(om, a) - logits_np(oc, a)))
        log(
            f'linear seed {seed}: A flip {a_metrics["marker_flip_rate"]:.6f} '
            f'conflict {a_metrics["conflict_set_FP"]} AUROC {a_metrics["AUROC"]:.6f}'
        )

        variants = []  # (arm, budget, r, theta, diag)
        variants.append(('A_NO_INTERVENTION', None, 0, a, dict(norm_ratio=1.0)))
        t0 = time.time()
        th_warm = retrain_linear_head(x, yy, a, seed)
        variants.append(
            (
                'RETRAINED-HEAD@k',
                None,
                None,
                th_warm,
                dict(
                    norm_ratio=float(np.linalg.norm(th_warm[:-1]) / np.linalg.norm(w)),
                    seconds=time.time() - t0,
                ),
            )
        )
        t0 = time.time()
        th_fresh = retrain_linear_head(
            x,
            yy,
            np.random.default_rng(seed + 900000).normal(0, 0.001, len(a)),
            seed,
            steps=RETRAIN_LINEAR_STEPS,
        )
        variants.append(
            (
                'FRESH-RETRAIN@k',
                None,
                None,
                th_fresh,
                dict(
                    norm_ratio=float(np.linalg.norm(th_fresh[:-1]) / np.linalg.norm(w)),
                    seconds=time.time() - t0,
                ),
            )
        )
        for frac in BUDGETS:
            r = int(round(frac * len(w)))
            for draw in range(1, RANDOM_DRAWS + 1):
                rng = np.random.default_rng(seed + 100000 * draw)
                keep = np.ones(len(w), dtype=bool)
                keep[rng.choice(len(w), size=r, replace=False)] = False
                variants.append(
                    (
                        f'RANDOM-DEL#{draw}',
                        frac,
                        r,
                        np.r_[w * keep, b],
                        dict(
                            norm_ratio=float(
                                np.linalg.norm(w * keep) / np.linalg.norm(w)
                            ),
                            deletions=int(r),
                        ),
                    )
                )
            for kind, name in (('DEL', 'DEL-MASK'), ('SIGNED', 'SIGNED-SPARSE')):
                t0 = time.time()
                m = LinearMask(w, b, kind, r, seed)
                m.fit(x, yy)
                wf = m.final_weights()
                variants.append(
                    (
                        name,
                        frac,
                        r,
                        np.r_[wf, b],
                        dict(
                            norm_ratio=float(np.linalg.norm(wf) / np.linalg.norm(w)),
                            deletions=int(r),
                            delta_norm=float(np.linalg.norm(m.delta.detach().numpy()))
                            if m.delta is not None
                            else 0.0,
                            seconds=time.time() - t0,
                        ),
                    )
                )
        for frac in BUDGETS:
            r = int(round(frac * len(w)))
            best_pl = 0.0
            for arm, fr, rr, theta, diag in variants:
                if fr != frac or not arm.startswith('RANDOM-DEL'):
                    continue
                s = logits_np(Xdev, theta)
                met = metrics_block(
                    dr, s, frozen, {row['id']: v for row, v in zip(dr, a_scores)}
                )
                best_pl = max(
                    best_pl, a_metrics['marker_flip_rate'] - met['marker_flip_rate']
                )
            best_pl = max(0.0, best_pl)
            for arm, fr, rr, theta, diag in variants:
                if fr != frac:
                    continue
                s = logits_np(Xdev, theta)
                met = metrics_block(
                    dr, s, frozen, {row['id']: v for row, v in zip(dr, a_scores)}
                )
                vals, bars = gates_for(met, a_metrics, best_pl)
                train_resp = float(np.mean(logits_np(om, theta) - logits_np(oc, theta)))
                results.append(
                    dict(
                        head='linear',
                        arm=arm,
                        budget=frac,
                        deletions=rr,
                        seed=seed,
                        metrics=met,
                        gates=vals,
                        bars=bars,
                        best_placebo=best_pl,
                        norm_ratio=diag.get('norm_ratio'),
                        delta_norm=diag.get('delta_norm'),
                        seconds=diag.get('seconds'),
                        train_marker_resp=train_resp,
                        baseline=a_metrics,
                        recal=recal_block(dr, s),
                    )
                )
        # Retrainers have no budget; their placebo bar is the primary budget's.
        for arm in ('RETRAINED-HEAD@k', 'FRESH-RETRAIN@k'):
            theta, diag = next((v[3], v[4]) for v in variants if v[0] == arm)
            s = logits_np(Xdev, theta)
            met = metrics_block(
                dr, s, frozen, {row['id']: v for row, v in zip(dr, a_scores)}
            )
            rng_best = 0.0
            for varm, vfr, vrr, vth, vdiag in variants:
                if vfr == PRIMARY_BUDGET and varm.startswith('RANDOM-DEL'):
                    rng_best = max(
                        rng_best,
                        a_metrics['marker_flip_rate']
                        - metrics_block(
                            dr,
                            logits_np(Xdev, vth),
                            frozen,
                            {row['id']: v for row, v in zip(dr, a_scores)},
                        )['marker_flip_rate'],
                    )
            rng_best = max(0.0, rng_best)
            vals, bars = gates_for(met, a_metrics, rng_best)
            results.append(
                dict(
                    head='linear',
                    arm=arm,
                    budget=None,
                    deletions=None,
                    seed=seed,
                    metrics=met,
                    gates=vals,
                    bars=bars,
                    best_placebo=rng_best,
                    norm_ratio=diag.get('norm_ratio'),
                    delta_norm=None,
                    seconds=diag.get('seconds'),
                    train_marker_resp=float(
                        np.mean(logits_np(om, theta) - logits_np(oc, theta))
                    ),
                    baseline=a_metrics,
                    recal=recal_block(dr, s),
                )
            )
        # A itself
        vals, bars = gates_for(a_metrics, a_metrics, 0.0)
        results.append(
            dict(
                head='linear',
                arm='A_NO_INTERVENTION',
                budget=None,
                deletions=0,
                seed=seed,
                metrics=a_metrics,
                gates=vals,
                bars=bars,
                best_placebo=0.0,
                norm_ratio=1.0,
                delta_norm=0.0,
                seconds=0.0,
                train_marker_resp=a_train_resp,
                baseline=a_metrics,
                recal=recal_block(dr, a_scores),
            )
        )

    log('=== hidden-layer MLP head (1536-32-1) ===')
    cidx = np.array([i for i in d.ti if d.rows[i]['confounded_train']])
    conf_x = d.scaled(cidx)
    conf_y = np.array([d.rows[i]['label'] for i in cidx], dtype=np.float64)
    log(f'MLP baseline training rows: {len(cidx)} confounded TRAIN views')
    mlp = {}
    for seed in SEEDS:
        t0 = time.time()
        prm = train_mlp(conf_x, conf_y, seed, steps=MLP_STEPS, lr=MLP_LR, l2=MLP_L2)
        th = mlp_flat(prm)
        acc = float(np.mean((mlp_forward_np(th, conf_x) >= 0) == conf_y))
        mlp[seed] = (prm, th)
        log(f'MLP A seed {seed}: TRAIN acc {acc:.4f}, {time.time() - t0:.1f}s')
    for seed in SEEDS:
        prm, th = mlp[seed]
        W1, b1, W2, b2 = prm
        a_scores = mlp_forward_np(th, Xdev)
        frozen = freeze_threshold(
            [dr[i] for i in clean_idx], a_scores[clean_idx], sha256_of(th.tolist())
        )
        frozen.update(
            seed=seed, rule='clean-DEV Youden-J on A_mlp, frozen for all MLP arms'
        )
        a_metrics = metrics_block(
            dr, a_scores, frozen, {r['id']: v for r, v in zip(dr, a_scores)}
        )
        ref_rows.setdefault('mlp', {})[str(seed)] = a_metrics
        ci, mi, y = d.k_pairs(seed)
        xc = d.scaled(ci)
        xm = d.scaled(mi)
        x = np.concatenate([xc, xm], 0)
        yy = np.concatenate([y, y])
        seen = set(r['group_id'] for r in d.subsets[seed])
        others = d.all_train_pair_ids(seen)
        oc = d.scaled([o[0] for o in others])
        om = d.scaled([o[1] for o in others])
        a_train_resp = float(np.mean(mlp_forward_np(th, om) - mlp_forward_np(th, oc)))
        log(
            f'MLP seed {seed}: A flip {a_metrics["marker_flip_rate"]:.6f} '
            f'conflict {a_metrics["conflict_set_FP"]} AUROC {a_metrics["AUROC"]:.6f} '
            f'train_resp {a_train_resp:.3f}'
        )

        variants = [
            (
                'A_NO_INTERVENTION',
                None,
                0,
                (W1.copy(), b1.copy(), W2.copy(), b2.copy()),
                dict(norm_ratio=1.0),
            )
        ]
        t0 = time.time()
        prm_w = train_mlp(x, yy, seed, steps=MLP_STEPS, lr=MLP_LR, l2=MLP_L2, warm=prm)
        variants.append(
            (
                'RETRAINED-HEAD@k',
                None,
                None,
                prm_w,
                dict(
                    norm_ratio=float(np.linalg.norm(prm_w[0]) / np.linalg.norm(W1)),
                    seconds=time.time() - t0,
                ),
            )
        )
        t0 = time.time()
        prm_f = train_mlp(x, yy, seed, steps=MLP_STEPS, lr=MLP_LR, l2=MLP_L2)
        variants.append(
            (
                'FRESH-RETRAIN@k',
                None,
                None,
                prm_f,
                dict(
                    norm_ratio=float(np.linalg.norm(prm_f[0]) / np.linalg.norm(W1)),
                    seconds=time.time() - t0,
                ),
            )
        )
        for frac in BUDGETS:
            r = int(round(frac * W1.size))
            for draw in range(1, RANDOM_DRAWS + 1):
                rng = np.random.default_rng(seed + 100000 * draw)
                keep = np.ones(W1.size, dtype=bool)
                keep[rng.choice(W1.size, size=r, replace=False)] = False
                variants.append(
                    (
                        f'RANDOM-DEL#{draw}',
                        frac,
                        r,
                        (W1 * keep.reshape(W1.shape), b1, W2, b2),
                        dict(
                            norm_ratio=float(
                                np.linalg.norm((W1 * keep.reshape(W1.shape)))
                                / np.linalg.norm(W1)
                            ),
                            deletions=int(r),
                        ),
                    )
                )
            for kind, name in (('DEL', 'DEL-MASK'), ('SIGNED', 'SIGNED-SPARSE')):
                t0 = time.time()
                m = MLPMask(W1, b1, W2, b2, kind, r, seed)
                m.fit(x, yy)
                keep = m.final_keep()
                if kind == 'DEL':
                    W1e = W1 * keep.reshape(W1.shape)
                else:
                    W1e = W1 + m.delta.detach().numpy() * keep.reshape(W1.shape)
                variants.append(
                    (
                        name,
                        frac,
                        r,
                        (W1e, b1.copy(), W2.copy(), b2.copy()),
                        dict(
                            norm_ratio=float(np.linalg.norm(W1e) / np.linalg.norm(W1)),
                            deletions=int(r),
                            delta_norm=float(np.linalg.norm(m.delta.detach().numpy()))
                            if m.delta is not None
                            else 0.0,
                            seconds=time.time() - t0,
                        ),
                    )
                )
        for frac in BUDGETS:
            best_pl = 0.0
            for arm, fr, rr, p, diag in variants:
                if fr == frac and arm.startswith('RANDOM-DEL'):
                    met = metrics_block(
                        dr,
                        mlp_forward_np(mlp_flat(p), Xdev),
                        frozen,
                        {row['id']: v for row, v in zip(dr, a_scores)},
                    )
                    best_pl = max(
                        best_pl, a_metrics['marker_flip_rate'] - met['marker_flip_rate']
                    )
            best_pl = max(0.0, best_pl)
            for arm, fr, rr, p, diag in variants:
                if fr != frac:
                    continue
                s = mlp_forward_np(mlp_flat(p), Xdev)
                met = metrics_block(
                    dr, s, frozen, {row['id']: v for row, v in zip(dr, a_scores)}
                )
                vals, bars = gates_for(met, a_metrics, best_pl)
                thp = mlp_flat(p)
                train_resp = float(
                    np.mean(mlp_forward_np(thp, om) - mlp_forward_np(thp, oc))
                )
                results.append(
                    dict(
                        head='mlp',
                        arm=arm,
                        budget=frac,
                        deletions=rr,
                        seed=seed,
                        metrics=met,
                        gates=vals,
                        bars=bars,
                        best_placebo=best_pl,
                        norm_ratio=diag.get('norm_ratio'),
                        delta_norm=diag.get('delta_norm'),
                        seconds=diag.get('seconds'),
                        train_marker_resp=train_resp,
                        baseline=a_metrics,
                        recal=recal_block(dr, s),
                    )
                )
        for arm in ('RETRAINED-HEAD@k', 'FRESH-RETRAIN@k'):
            p, diag = next((v[3], v[4]) for v in variants if v[0] == arm)
            s = mlp_forward_np(mlp_flat(p), Xdev)
            met = metrics_block(
                dr, s, frozen, {row['id']: v for row, v in zip(dr, a_scores)}
            )
            best_pl = 0.0
            for varm, vfr, vrr, vp, vdiag in variants:
                if vfr == PRIMARY_BUDGET and varm.startswith('RANDOM-DEL'):
                    mmet = metrics_block(
                        dr,
                        mlp_forward_np(mlp_flat(vp), Xdev),
                        frozen,
                        {row['id']: v for row, v in zip(dr, a_scores)},
                    )
                    best_pl = max(
                        best_pl,
                        a_metrics['marker_flip_rate'] - mmet['marker_flip_rate'],
                    )
            best_pl = max(0.0, best_pl)
            vals, bars = gates_for(met, a_metrics, best_pl)
            thp = mlp_flat(p)
            results.append(
                dict(
                    head='mlp',
                    arm=arm,
                    budget=None,
                    deletions=None,
                    seed=seed,
                    metrics=met,
                    gates=vals,
                    bars=bars,
                    best_placebo=best_pl,
                    norm_ratio=diag.get('norm_ratio'),
                    delta_norm=None,
                    seconds=diag.get('seconds'),
                    train_marker_resp=float(
                        np.mean(mlp_forward_np(thp, om) - mlp_forward_np(thp, oc))
                    ),
                    baseline=a_metrics,
                    recal=recal_block(dr, s),
                )
            )
        vals, bars = gates_for(a_metrics, a_metrics, 0.0)
        results.append(
            dict(
                head='mlp',
                arm='A_NO_INTERVENTION',
                budget=None,
                deletions=0,
                seed=seed,
                metrics=a_metrics,
                gates=vals,
                bars=bars,
                best_placebo=0.0,
                norm_ratio=1.0,
                delta_norm=0.0,
                seconds=0.0,
                train_marker_resp=a_train_resp,
                baseline=a_metrics,
                recal=recal_block(dr, a_scores),
            )
        )

    _write_outputs(results, ref_rows, timing, args.out_dir)
    log('probe done')


def _write_outputs(results, ref_rows, timing, out_dir):
    wide = out_dir / 'RESULTS.csv'
    fields = [
        'head',
        'arm',
        'budget',
        'deletions',
        'seed',
        'threshold',
        'marker_flip',
        'marker_rel_reduction',
        'conflict_FP',
        'conflict_rel_reduction',
        'AUROC',
        'AP',
        'FP',
        'FN',
        'recall',
        'FPR',
        'recall_margin',
        'FPR_margin',
        'placebo_margin',
        'best_placebo_abs_reduction',
        'same_direction',
        'gate_pass_all',
        'failed_gates',
        'norm_ratio',
        'delta_norm',
        'train_marker_resp',
        'clean_newly_wrong_rate',
        'seconds',
        'recal_threshold',
        'recal_flip',
        'recal_recall',
        'recal_FPR',
    ]
    rowsout = []
    for it in results:
        m = it['metrics']
        g = it['gates']
        failed = [k for k, v in it['bars'].items() if v == 'FAIL']
        cd = m.get('clean_damage') or {}
        rowsout.append(
            dict(
                head=it['head'],
                arm=it['arm'],
                budget=it['budget'],
                deletions=it['deletions'],
                seed=it['seed'],
                threshold=m['threshold'],
                marker_flip=m['marker_flip_rate'],
                marker_rel_reduction=g['marker_relative_reduction_min'],
                conflict_FP=m['conflict_set_FP'],
                conflict_rel_reduction=g['conflict_relative_reduction_min'],
                AUROC=m['AUROC'],
                AP=m['AP'],
                FP=m['FP'],
                FN=m['FN'],
                recall=m['recall'],
                FPR=m['FPR'],
                recall_margin=g['recall_margin_min'],
                FPR_margin=g['FPR_margin_min'],
                placebo_margin=g['placebo_margin_min'],
                best_placebo_abs_reduction=it['best_placebo'],
                same_direction=g['same_direction_all_seeds'],
                gate_pass_all=('PASS' if not failed else 'FAIL'),
                failed_gates=('|'.join(sorted(failed))),
                norm_ratio=it['norm_ratio'],
                delta_norm=it['delta_norm'],
                train_marker_resp=it['train_marker_resp'],
                clean_newly_wrong_rate=cd.get('newly_wrong_rate'),
                seconds=it['seconds'],
                recal_threshold=it['recal']['threshold'],
                recal_flip=it['recal']['marker_flip'],
                recal_recall=it['recal']['recall'],
                recal_FPR=it['recal']['FPR'],
            )
        )
    with wide.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rowsout:
            w.writerow(
                {
                    k: (
                        ''
                        if row[k] is None
                        else (f'{row[k]:.12g}' if isinstance(row[k], float) else row[k])
                    )
                    for k in fields
                }
            )
    # long gate table
    with (out_dir / 'RESULTS_gates.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(
            [
                'head',
                'arm',
                'budget',
                'deletions',
                'seed',
                'gate',
                'value',
                'bar',
                'pass',
            ]
        )
        for it in results:
            for k in GATES:
                w.writerow(
                    [
                        it['head'],
                        it['arm'],
                        it['budget'],
                        it['deletions'],
                        it['seed'],
                        k,
                        f'{it["gates"][k]:.12g}',
                        GATES[k],
                        it['bars'][k],
                    ]
                )
    # aggregate (worst seed) per head/arm/budget
    agg = []
    keys = sorted(
        {(it['head'], it['arm'], it['budget']) for it in results},
        key=lambda k: (k[0], str(k[2]), k[1]),
    )
    for head, arm, budget in keys:
        items = [
            it
            for it in results
            if it['head'] == head and it['arm'] == arm and it['budget'] == budget
        ]
        row = dict(
            head=head,
            arm=arm,
            budget=budget,
            n_seeds=len(items),
            flip_mean=float(
                np.mean([it['metrics']['marker_flip_rate'] for it in items])
            ),
            AUROC_mean=float(np.mean([it['metrics']['AUROC'] for it in items])),
            FPR_mean=float(np.mean([it['metrics']['FPR'] for it in items])),
            recall_mean=float(np.mean([it['metrics']['recall'] for it in items])),
            conflict_FP_mean=float(
                np.mean([it['metrics']['conflict_set_FP'] for it in items])
            ),
        )
        for k in GATES:
            row[k] = min(it['gates'][k] for it in items)
        row['gate_pass_all'] = (
            'PASS' if all(row[k] >= GATES[k] - 1e-12 for k in GATES) else 'FAIL'
        )
        agg.append(row)
    with (out_dir / 'RESULTS_agg.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0]))
        w.writeheader()
        for row in agg:
            w.writerow(
                {
                    k: (f'{v:.12g}' if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            )
    with (out_dir / 'results_raw.json').open('w') as f:
        json.dump(
            dict(
                total_seconds=time.time() - timing['start'],
                references=ref_rows,
                split='dev',
                holdout_reads=0,
                results=results,
            ),
            f,
            indent=1,
            default=float,
            allow_nan=False,
        )
    print(json.dumps(dict(n_results=len(results), agg=len(agg)), indent=1))


if __name__ == '__main__':
    main()
