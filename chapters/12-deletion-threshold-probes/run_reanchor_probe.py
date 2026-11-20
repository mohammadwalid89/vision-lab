"""Threshold re-anchor probe: does the single frozen threshold explain the failures?

Every arm is scored at three operating points on DEV:
  a     the arm family's original frozen threshold;
  b     a threshold re-fit on one shared TRAIN-only calibration slice (the
        Round 1 maximum-balanced-accuracy rule);
  dfit  a threshold re-fit on clean DEV (diagnostic, never used for pass/fail).
It also reports two threshold-free checks of whether gates 2 and 3 can be met
together at any threshold.

    python run_reanchor_probe.py --wsp-root <round-1/2 tree> \
        --deletion-results <deletion probe results_raw.json> --out-dir out/
"""

import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from probe_lib import (
    MLP_L2,
    MLP_LR,
    MLP_STEPS,
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
from vision_lab.wsp.legacy.experiments import ablate
from vision_lab.wsp.legacy.r2_methods import controls, subspace
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
CAL_N_GOOD = 384
CAL_N_DEFECT = 128
R2_V1_K32 = 'revision2/artifacts/vision-wsp-r2-v1-i2-20261113'

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


def youden(scores, y):
    """The Round 1 threshold scan (`freeze_threshold`), usable on TRAIN scores."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(s, kind='stable')
    ss = s[order]
    y2 = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    removed = np.r_[0, np.cumsum(y2)][starts]
    n1 = int(y2.sum())
    n0 = len(y2) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError('Both classes required for threshold')
    tp = n1 - removed
    fp = n0 - (starts - removed)
    thresholds = np.r_[ss[starts], np.nextafter(ss[-1], np.inf)]
    ba = np.r_[(tp / n1 + 1 - fp / n0) / 2, 0.5]
    th = float(max(zip(ba, thresholds))[1])
    pred = s >= th
    rec = float(pred[y == 1].mean())
    fpr = float(pred[y == 0].mean())
    return dict(
        threshold=th,
        balanced_accuracy=(rec + 1 - fpr) / 2,
        youden=rec - fpr,
        recall=rec,
        FPR=fpr,
    )


def half_up(x):
    return int(np.floor(float(x) + 0.5))


def build_calibration(d):
    union = set()
    for s in SEEDS:
        union |= {r['group_id'] for r in d.subsets[s]}
    intake = json.loads((d.root / 'r1/INTAKE.json').read_text())['rows']
    elig = [
        r
        for r in intake
        if r['split'] == 'train'
        and r['group_id'] not in union
        and r['group_id'] in d._train_pairs
    ]

    def h(r):
        return hashlib.sha256(('reanchor|' + r['source_id']).encode()).hexdigest()

    pos = sorted([r for r in elig if r['label'] == 1], key=h)[:CAL_N_DEFECT]
    neg = sorted([r for r in elig if r['label'] == 0], key=h)[:CAL_N_GOOD]
    if len(pos) < CAL_N_DEFECT or len(neg) < CAL_N_GOOD:
        raise RuntimeError('Insufficient calibration capacity')
    cal = pos + neg
    ids = np.array([d._train_pairs[r['group_id']]['clean'] for r in cal])
    y = np.array([r['label'] for r in cal], dtype=int)
    return cal, ids, y


def metrics(dr, scores, thr, ref_scores):
    m = score(
        dr,
        np.asarray(scores, dtype=float),
        dict(threshold=float(thr), frozen=True),
        {r['id']: float(v) for r, v in zip(dr, ref_scores)},
        dict(n_images=len(dr), elapsed_seconds=0.0),
    )
    return m


def gate_values(cand, base, bar):
    return seed_bars(cand, base, None, None, float(bar), 'V1')


def bars_of(vals):
    return {k: ('PASS' if vals[k] >= GATES[k] - 1e-12 else 'FAIL') for k in GATES}


def roc_match(cand_scores, base_scores, clean_idx, y_clean, base_thr):
    """Threshold-free gates-2/3 feasibility.

    (i)  minimum clean FPR at >= baseline clean recall;
    (ii) maximum clean recall at clean FPR <= baseline FPR + 0.02.
    """
    cs = np.asarray(cand_scores)[clean_idx]
    bs = np.asarray(base_scores)[clean_idx]
    pos = np.sort(cs[y_clean == 1])[::-1]
    neg = np.sort(cs[y_clean == 0])
    bpos = bs[y_clean == 1]
    bneg = bs[y_clean == 0]
    base_rec = float((bpos >= base_thr).mean())
    base_fpr = float((bneg >= base_thr).mean())
    n1, n0 = len(pos), len(neg)
    k = max(1, min(n1, int(np.ceil(base_rec * n1))))
    t_rec = float(pos[k - 1])
    fpr_at_rec = float((neg >= t_rec).mean())
    rec_at_rec = float((pos >= t_rec).mean())
    limit = base_fpr + 0.02
    m = int(np.floor(limit * n0))
    m = max(0, min(n0, m))
    if m <= 0:
        t_fpr = float(np.nextafter(neg[-1], np.inf))
    elif m >= n0:
        t_fpr = float(np.nextafter(neg[0], -np.inf))
    else:
        t_fpr = float(neg[n0 - m])
    rec_at_fpr = float((pos >= t_fpr).mean())
    fpr_at_fpr = float((neg >= t_fpr).mean())
    return dict(
        base_recall=base_rec,
        base_FPR=base_fpr,
        fpr_at_base_recall=fpr_at_rec,
        recall_at_base_recall=rec_at_rec,
        fpr_margin_at_base_recall=base_fpr - fpr_at_rec,
        recall_at_base_fpr=rec_at_fpr,
        fpr_at_base_fpr=fpr_at_fpr,
        recall_margin_at_base_fpr=rec_at_fpr - base_rec,
    )


# ------------------------------------------------------------------ arm build
def linear_items(d, Xdev, Xcal, Xdev32, Xcal32):
    """One entry per seed: items of the linear/R1 cell."""
    out = []
    for seed in SEEDS:
        ci, mi, y = d.k_pairs(seed)
        x = np.concatenate([d.scaled(ci), d.scaled(mi)], 0)
        yy = np.concatenate([y, y])
        a = d.A[seed]
        w, b = a[:-1], a[-1]
        frozen = float(d.thresholds[seed]['threshold'])
        base_dev = logits_np(Xdev, a)
        base_cal = logits_np(Xcal, a)
        items = [
            dict(
                id='A_NO_INTERVENTION',
                arm='A_NO_INTERVENTION',
                budget=None,
                role='baseline',
                group='none',
                frozen=frozen,
                scores_dev=base_dev,
                scores_cal=base_cal,
                theta=a,
            )
        ]

        th = retrain_linear_head(x, yy, a, seed)
        items.append(
            dict(
                id='RETRAINED-HEAD@k',
                arm='RETRAINED-HEAD@k',
                budget=None,
                role='candidate',
                group='budget:0.1',
                frozen=frozen,
                scores_dev=logits_np(Xdev, th),
                scores_cal=logits_np(Xcal, th),
                theta=th,
            )
        )
        for frac, arm, kind in (
            (0.01, 'SIGNED-SPARSE', 'SIGNED'),
            (0.25, 'DEL-MASK', 'DEL'),
        ):
            r = half_up(frac * len(w))
            m = LinearMask(w, b, kind, r, seed)
            m.fit(x, yy)
            wf = m.final_weights()
            th = np.r_[wf, b]
            items.append(
                dict(
                    id=arm,
                    arm=arm,
                    budget=frac,
                    role='candidate',
                    group=f'budget:{frac:g}',
                    frozen=frozen,
                    scores_dev=logits_np(Xdev, th),
                    scores_cal=logits_np(Xcal, th),
                    theta=th,
                    r=r,
                )
            )
        for frac in sorted({0.01, 0.1, 0.25}):
            r = half_up(frac * len(w))
            for draw in (1, 2, 3):
                rng = np.random.default_rng(seed + 100000 * draw)
                keep = np.ones(len(w), dtype=bool)
                keep[rng.choice(len(w), size=r, replace=False)] = False
                th = np.r_[w * keep, b]
                items.append(
                    dict(
                        id=f'RANDOM-DEL#{draw}@{frac:g}',
                        arm=f'RANDOM-DEL#{draw}',
                        budget=frac,
                        role='placebo',
                        group=f'budget:{frac:g}',
                        frozen=frozen,
                        scores_dev=logits_np(Xdev, th),
                        scores_cal=logits_np(Xcal, th),
                        theta=th,
                        r=r,
                    )
                )
        # R1 i1 V1 saved EDITS: these are feature ablations, so the editable
        # artifact is the saved per-seed direction + centre in
        # r1/fits/i1/linear/{seed}_fit.npz; the per-arm *_weights.npy is A.
        f = np.load(d.root / f'r1/fits/i1/linear/{seed}_fit.npz')
        center = f['center']

        def abl_scores(X32, control, dose):
            sham = control == 'sham'
            x = ablate(X32, f[control], dose, center, sham=sham)
            return logits_np(x, a)

        items.append(
            dict(
                id='R1:marker_dose_1',
                arm='V1_marker_dose_1',
                budget=None,
                role='candidate',
                group='r1placebo',
                frozen=frozen,
                scores_dev=abl_scores(Xdev32, 'marker', 1.0),
                scores_cal=abl_scores(Xcal32, 'marker', 1.0),
                theta=None,
                source=f'r1/fits/i1/linear/{seed}_fit.npz:marker,center',
            )
        )
        for ctrl in ('irrelevant', 'random', 'sham'):
            for dose in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
                items.append(
                    dict(
                        id=f'R1:{ctrl}_dose_{dose:g}',
                        arm=f'{ctrl}_dose_{dose:g}',
                        budget=None,
                        role='placebo',
                        group='r1placebo',
                        frozen=frozen,
                        scores_dev=abl_scores(Xdev32, ctrl, dose),
                        scores_cal=abl_scores(Xcal32, ctrl, dose),
                        theta=None,
                    )
                )
        out.append(dict(cell='linear', seed=seed, items=items))
    return out


def mlp_items(d, Xdev, Xcal, conf_x, conf_y):
    out = []
    a_mlp = {}
    for seed in SEEDS:
        prm = train_mlp(conf_x, conf_y, seed, steps=MLP_STEPS, lr=MLP_LR, l2=MLP_L2)
        th = mlp_flat(prm)
        a_mlp[seed] = (prm, th)
        acc = float(np.mean((mlp_forward_np(th, conf_x) >= 0) == conf_y))
        log(f'MLP A seed {seed}: TRAIN acc {acc:.4f}')
    for seed in SEEDS:
        prm, thA = a_mlp[seed]
        W1, b1, W2, b2 = prm
        base_dev = mlp_forward_np(thA, Xdev)
        base_cal = mlp_forward_np(thA, Xcal)
        frozen = freeze_threshold(
            [d.dr[i] for i in d._clean_idx],
            base_dev[d._clean_idx],
            sha256_of(thA.tolist()),
        )
        frozen_thr = float(frozen['threshold'])
        ci, mi, y = d.k_pairs(seed)
        x = np.concatenate([d.scaled(ci), d.scaled(mi)], 0)
        yy = np.concatenate([y, y])
        items = [
            dict(
                id='A_NO_INTERVENTION',
                arm='A_NO_INTERVENTION',
                budget=None,
                role='baseline',
                group='none',
                frozen=frozen_thr,
                scores_dev=base_dev,
                scores_cal=base_cal,
                theta=thA,
            )
        ]
        prm_w = train_mlp(x, yy, seed, steps=MLP_STEPS, lr=MLP_LR, l2=MLP_L2, warm=prm)
        thw = mlp_flat(prm_w)
        items.append(
            dict(
                id='RETRAINED-HEAD@k',
                arm='RETRAINED-HEAD@k',
                budget=None,
                role='candidate',
                group='budget:0.1',
                frozen=frozen_thr,
                scores_dev=mlp_forward_np(thw, Xdev),
                scores_cal=mlp_forward_np(thw, Xcal),
                theta=thw,
            )
        )
        for frac, arm, kind in (
            (0.1, 'SIGNED-SPARSE', 'SIGNED'),
            (0.05, 'DEL-MASK', 'DEL'),
        ):
            r = half_up(frac * W1.size)
            m = MLPMask(W1, b1, W2, b2, kind, r, seed)
            m.fit(x, yy)
            keep = m.final_keep()
            if kind == 'DEL':
                W1e = W1 * keep.reshape(W1.shape)
            else:
                W1e = W1 + m.delta.detach().numpy() * keep.reshape(W1.shape)
            thp = mlp_flat((W1e, b1.copy(), W2.copy(), b2.copy()))
            items.append(
                dict(
                    id=arm,
                    arm=arm,
                    budget=frac,
                    role='candidate',
                    group=f'budget:{frac:g}',
                    frozen=frozen_thr,
                    scores_dev=mlp_forward_np(thp, Xdev),
                    scores_cal=mlp_forward_np(thp, Xcal),
                    theta=thp,
                    r=r,
                )
            )
        for frac in sorted({0.05, 0.1}):
            r = half_up(frac * W1.size)
            for draw in (1, 2, 3):
                rng = np.random.default_rng(seed + 100000 * draw)
                keep = np.ones(W1.size, dtype=bool)
                keep[rng.choice(W1.size, size=r, replace=False)] = False
                W1e = W1 * keep.reshape(W1.shape)
                thp = mlp_flat((W1e, b1.copy(), W2.copy(), b2.copy()))
                items.append(
                    dict(
                        id=f'RANDOM-DEL#{draw}@{frac:g}',
                        arm=f'RANDOM-DEL#{draw}',
                        budget=frac,
                        role='placebo',
                        group=f'budget:{frac:g}',
                        frozen=frozen_thr,
                        scores_dev=mlp_forward_np(thp, Xdev),
                        scores_cal=mlp_forward_np(thp, Xcal),
                        theta=thp,
                        r=r,
                    )
                )
        out.append(dict(cell='mlp', seed=seed, items=items))
    return out


def r2_items(d, cal_ids, cal_y, X2dev):
    shared = d.root / 'revision2/artifacts/shared'
    sc = np.load(shared / 'scalers.npz')
    mean, scale = sc['mean'], sc['scale']
    pooled = np.load(d.root / 'r1/cache/siglip2_base/pooled.npy', mmap_mode='r')
    mp = np.load(d.root / 'r1/cache/siglip2_base/mean_patch.npy', mmap_mode='r')

    def raw(ids):
        ids = np.asarray(ids)
        return np.concatenate(
            [pooled[ids, -1].astype('float64'), mp[ids, -1].astype('float64')], 1
        )

    X2cal = ((raw(cal_ids) - mean) / scale).astype('float32').astype(float)
    pw = np.load(d.root / R2_V1_K32 / 'predictions_weights.npz')
    out = []
    for seed in SEEDS:
        z = np.load(shared / f'{seed}_k32_pairs.npz')
        c = ((z['c'] - mean) / scale).astype('float32').astype(float)
        m = ((z['m'] - mean) / scale).astype('float32').astype(float)
        q, mu, g, diag = subspace(c, m, z['y'])
        a2 = np.load(shared / f'{seed}_A.npy')
        frozen = float(
            json.loads((shared / f'{seed}_threshold.json').read_text())['threshold']
        )

        def proj_scores(X, qq, mu_, dose=1.0):
            X = np.asarray(X, dtype=float)
            return logits_np(X - dose * (((X - mu_) @ qq) @ qq.T), a2)

        base_dev = logits_np(np.asarray(X2dev, dtype=float), a2)
        base_cal = logits_np(X2cal, a2)
        items = [
            dict(
                id='R2_A_BASELINE',
                arm='R2_A_BASELINE',
                budget=None,
                role='baseline',
                group='none',
                frozen=frozen,
                scores_dev=base_dev,
                scores_cal=base_cal,
                theta=a2,
            )
        ]
        cand_dev = proj_scores(X2dev, q, mu)
        cand_cal = proj_scores(X2cal, q, mu)
        items.append(
            dict(
                id='R2_BEST_EDIT',
                arm='R2_BEST_EDIT',
                budget=None,
                role='candidate',
                group='r2placebo',
                frozen=frozen,
                scores_dev=cand_dev,
                scores_cal=cand_cal,
                theta=None,
                rank=int(q.shape[1]),
                source='R2 V1 i2 candidate@1',
            )
        )
        # published R2 V1 control set: 3 random subspaces at dose 1 + zero-dose sham
        ct = controls(dict(q=q, mu=mu, g=g, delta=np.zeros_like(a2)), a2, seed, 'V1')
        # R2 bar is the best non-negative gain over the controls' FULL dose grid.
        for name in ('random1', 'random2', 'random3'):
            qq = ct[name]
            for dose in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
                items.append(
                    dict(
                        id=f'R2_PLACEBO:{name}@{dose:g}',
                        arm=f'R2_PLACEBO:{name}',
                        budget=None,
                        role='placebo',
                        group='r2placebo',
                        frozen=frozen,
                        scores_dev=proj_scores(X2dev, qq, mu, dose),
                        scores_cal=proj_scores(X2cal, qq, mu, dose),
                        theta=None,
                    )
                )
        items.append(
            dict(
                id='R2_PLACEBO:sham@0',
                arm='R2_PLACEBO:sham',
                budget=None,
                role='placebo',
                group='r2placebo',
                frozen=frozen,
                scores_dev=proj_scores(X2dev, q, mu, 0.0),
                scores_cal=proj_scores(X2cal, q, mu, 0.0),
                theta=None,
            )
        )
        chk = float(np.abs(cand_dev - pw[f'{seed}_score_candidate@1']).max())
        out.append(dict(cell='r2', seed=seed, items=items, r2_repro_maxdiff=chk))
    return out


# ------------------------------------------------------------------ evaluate
def evaluate(cells, cal_y, dr, clean_idx, y_clean, dp_recal):
    rows, gate_rows = [], []
    for cell in cells:
        seed = cell['seed']
        items = cell['items']
        base = next(it for it in items if it['role'] == 'baseline')
        # thresholds per condition
        for cond in ('a', 'b', 'dfit'):
            thr, calstat = {}, {}
            for it in items:
                if cond == 'a':
                    thr[it['id']] = it['frozen']
                elif cond == 'dfit':
                    thr[it['id']] = youden(
                        np.asarray(it['scores_dev'])[clean_idx], y_clean
                    )['threshold']
                else:
                    u = youden(it['scores_cal'], cal_y)
                    thr[it['id']] = u['threshold']
                    calstat[it['id']] = u
            bm = metrics(dr, base['scores_dev'], thr[base['id']], base['scores_dev'])
            # placebo bars per group
            bars = {}
            groups = {it['group'] for it in items if it['role'] == 'placebo'}
            for g in groups:
                pls = [
                    it for it in items if it['role'] == 'placebo' and it['group'] == g
                ]
                best = 0.0
                for it in pls:
                    pm = metrics(
                        dr, it['scores_dev'], thr[it['id']], base['scores_dev']
                    )
                    best = max(best, bm['marker_flip_rate'] - pm['marker_flip_rate'])
                bars[g] = max(0.0, best)
            for it in items:
                if it['role'] not in ('baseline', 'candidate'):
                    continue
                met = metrics(dr, it['scores_dev'], thr[it['id']], base['scores_dev'])
                bar = 0.0 if it['role'] == 'baseline' else bars[it['group']]
                vals = gate_values(met, bm, bar)
                gb = bars_of(vals)
                dfit = youden(np.asarray(it['scores_dev'])[clean_idx], y_clean)
                rm = roc_match(
                    it['scores_dev'],
                    base['scores_dev'],
                    clean_idx,
                    y_clean,
                    thr[base['id']],
                )
                dp = dp_recal.get(
                    (cell['cell'], it['arm'], str(it['budget']), int(seed)), {}
                )
                u = calstat.get(it['id'], youden(it['scores_cal'], cal_y))
                rthr = thr[it['id']] if cond == 'dfit' else u['threshold']
                rows.append(
                    dict(
                        cell=cell['cell'],
                        seed=seed,
                        condition=cond,
                        arm=it['arm'],
                        budget=it['budget'],
                        role=it['role'],
                        threshold=thr[it['id']],
                        frozen_threshold=it['frozen'],
                        reanchor_threshold=rthr,
                        threshold_delta=rthr - it['frozen'],
                        cal_balanced_accuracy=u['balanced_accuracy'],
                        cal_youden=u['youden'],
                        cal_recall=u['recall'],
                        cal_FPR=u['FPR'],
                        devfit_threshold=dfit['threshold'],
                        cal_minus_devfit=u['threshold'] - dfit['threshold'],
                        reanchor_minus_devfit=rthr - dfit['threshold'],
                        dp_devfit_threshold=dp.get('threshold'),
                        cal_minus_dp=(u['threshold'] - dp['threshold'])
                        if dp.get('threshold') is not None
                        else None,
                        marker_flip=met['marker_flip_rate'],
                        marker_rel_reduction=vals['marker_relative_reduction_min'],
                        conflict_FP=met['conflict_set_FP'],
                        conflict_rel_reduction=vals['conflict_relative_reduction_min'],
                        recall=met['recall'],
                        FPR=met['FPR'],
                        recall_margin=vals['recall_margin_min'],
                        FPR_margin=vals['FPR_margin_min'],
                        placebo_margin=vals['placebo_margin_min'],
                        best_placebo_abs_reduction=bar,
                        same_direction=vals['same_direction_all_seeds'],
                        gate_pass_all=(
                            'PASS' if all(v == 'PASS' for v in gb.values()) else 'FAIL'
                        ),
                        failed_gates='|'.join(
                            sorted(k for k, v in gb.items() if v == 'FAIL')
                        ),
                        AUROC=met['AUROC'],
                        AP=met['AP'],
                        marker_relative_reduction_min=vals[
                            'marker_relative_reduction_min'
                        ],
                        conflict_relative_reduction_min=vals[
                            'conflict_relative_reduction_min'
                        ],
                        recall_margin_min=vals['recall_margin_min'],
                        FPR_margin_min=vals['FPR_margin_min'],
                        placebo_margin_min=vals['placebo_margin_min'],
                        same_direction_all_seeds=vals['same_direction_all_seeds'],
                        **rm,
                    )
                )
                for gname in GATES:
                    gate_rows.append(
                        dict(
                            cell=cell['cell'],
                            seed=seed,
                            condition=cond,
                            arm=it['arm'],
                            budget=it['budget'],
                            gate=gname,
                            value=vals[gname],
                            bar=GATES[gname],
                            passfail=gb[gname],
                        )
                    )
    return rows, gate_rows


def aggregate(rows):
    keys = sorted(
        {(r['cell'], r['arm'], str(r['budget']), r['condition']) for r in rows},
        key=lambda k: (k[0], k[3], k[1], k[2]),
    )
    out = []
    for cell, arm, budget, cond in keys:
        its = [
            r
            for r in rows
            if r['cell'] == cell
            and r['arm'] == arm
            and str(r['budget']) == budget
            and r['condition'] == cond
        ]
        row = dict(
            cell=cell,
            arm=arm,
            budget=budget,
            condition=cond,
            n_seeds=len(its),
            threshold_mean=float(np.mean([r['threshold'] for r in its])),
            frozen_mean=float(np.mean([r['frozen_threshold'] for r in its])),
            threshold_delta_mean=float(np.mean([r['threshold_delta'] for r in its])),
            devfit_mean=float(np.mean([r['devfit_threshold'] for r in its])),
            cal_minus_devfit_mean=float(np.mean([r['cal_minus_devfit'] for r in its])),
            reanchor_minus_devfit_mean=float(
                np.mean([r['reanchor_minus_devfit'] for r in its])
            ),
            flip_mean=float(np.mean([r['marker_flip'] for r in its])),
            FPR_mean=float(np.mean([r['FPR'] for r in its])),
            recall_mean=float(np.mean([r['recall'] for r in its])),
            AUROC_mean=float(np.mean([r['AUROC'] for r in its])),
            AP_mean=float(np.mean([r['AP'] for r in its])),
            fpr_at_base_recall=float(np.mean([r['fpr_at_base_recall'] for r in its])),
            fpr_margin_at_base_recall=float(
                np.mean([r['fpr_margin_at_base_recall'] for r in its])
            ),
            recall_at_base_fpr=float(np.mean([r['recall_at_base_fpr'] for r in its])),
            recall_margin_at_base_fpr=float(
                np.mean([r['recall_margin_at_base_fpr'] for r in its])
            ),
            conflict_FP_mean=float(np.mean([r['conflict_FP'] for r in its])),
        )
        for g in GATES:
            row[g] = min(r[g] for r in its)
        row['gate_pass_all'] = (
            'PASS' if all(row[g] >= GATES[g] - 1e-12 for g in GATES) else 'FAIL'
        )
        row['n_pass'] = int(sum(row[g] >= GATES[g] - 1e-12 for g in GATES))
        row['failed_gates'] = '|'.join(
            sorted(g for g in GATES if row[g] < GATES[g] - 1e-12)
        )
        out.append(row)
    return out


def fmt(v):
    if v is None:
        return ''
    if isinstance(v, float) or isinstance(v, np.floating):
        return f'{float(v):.12g}'
    return str(v)


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--wsp-root',
        type=Path,
        required=True,
        help='Round 1/2 experiment tree (r1/, revision2/, DATA_MANIFEST_R1.json)',
    )
    parser.add_argument(
        '--deletion-results',
        type=Path,
        required=True,
        help='results_raw.json written by run_deletion_probe.py',
    )
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.wsp_root
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log('reanchor probe start')

    d = Data(root)
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

    dr = d.dr
    Xdev = d.scaled(d.di)
    clean_idx = np.array([i for i, r in enumerate(dr) if r['variant'] == 'clean'])
    y_clean = np.array([dr[i]['label'] for i in clean_idx], dtype=int)
    d._clean_idx = clean_idx

    # base reproduction + threshold scan equality
    a0 = d.A[SEEDS[0]]
    a0_scores = logits_np(Xdev, a0)
    frozen0 = float(d.thresholds[SEEDS[0]]['threshold'])
    m0 = metrics(dr, a0_scores, frozen0, a0_scores)
    for key, val in PUBLISHED_A_SEED1.items():
        if abs(m0[key] - val) > 1e-9:
            raise SystemExit(f'BLOCKED base reproduction {key}: {m0[key]} != {val}')
    fr = freeze_threshold([dr[i] for i in clean_idx], a0_scores[clean_idx], 'check')
    if abs(fr['threshold'] - frozen0) > 1e-12:
        raise SystemExit('BLOCKED threshold scan differs from R1 freeze rule')
    log(
        'A_linear reproduction matches published R1 i1 metrics; Youden scan equals R1 freeze rule'
    )

    # R1 saved-weights reproduction (candidate + placebo bar)
    r1_pub = json.loads(
        (root / 'r1/scores/vision-wsp-r1-v1-i1-20261113.json').read_text()
    )
    r1_checks = {}
    Xdev32 = d.scaled(d.di).astype('float32')
    for seed in SEEDS:
        f = np.load(root / f'r1/fits/i1/linear/{seed}_fit.npz')
        x = ablate(Xdev32, f['marker'], 1.0, f['center'])
        met = metrics(
            dr,
            logits_np(x, d.A[seed]),
            float(d.thresholds[seed]['threshold']),
            a0_scores,
        )
        ref = r1_pub['metrics']['marker_dose_1'][str(seed)]
        for k in ('marker_flip_rate', 'conflict_set_FP', 'AUROC'):
            if abs(met[k] - ref[k]) > 1e-9:
                raise SystemExit(
                    f'BLOCKED R1 saved-direction reproduction {seed} {k}: '
                    f'{met[k]} vs {ref[k]}'
                )
        r1_checks[str(seed)] = 'ok'
    log('R1 marker_dose_1 saved direction+centre reproduce published R1 i1 metrics')

    # calibration slice
    cal, cal_ids, cal_y = build_calibration(d)
    Xcal = d.scaled(cal_ids)
    Xcal32 = Xcal.astype('float32')
    log(
        f'calibration slice: {len(cal)} TRAIN sources ({int(cal_y.sum())} defect / '
        f'{len(cal_y) - int(cal_y.sum())} good), sha256 {sha256_of([c["group_id"] for c in cal])[:16]}'
    )
    log(
        f'TRAIN A threshold (calibration, A_linear): {youden(logits_np(Xcal, a0), cal_y)}'
    )

    # MLP training data (confounded TRAIN views) for the secondary head
    cidx = np.array([i for i in d.ti if d.rows[i]['confounded_train']])
    conf_x = d.scaled(cidx)
    conf_y = np.array([d.rows[i]['label'] for i in cidx], dtype=np.float64)
    log(f'MLP baseline training rows: {len(cidx)} confounded TRAIN views')

    # deletion-probe DEV-fit recal thresholds for identical arms
    dp = json.loads(args.deletion_results.read_text())
    dp_recal = {}
    for it in dp['results']:
        dp_recal[(it['head'], it['arm'], str(it['budget']), int(it['seed']))] = it[
            'recal'
        ]
    log(f'loaded {len(dp_recal)} deletion-probe recal threshold records')

    t0 = time.time()
    cells = []
    log('=== linear head + R1 saved edits ===')
    cells += linear_items(d, Xdev, Xcal, Xdev32, Xcal32)
    log('=== R2 V1 k=32 projection edit (own representation) ===')
    r2cells = r2_items(
        d, cal_ids, cal_y, np.load(root / 'revision2/artifacts/shared/linear_dev.npy')
    )
    for c in r2cells:
        log(
            f'R2 reproduction seed {c["seed"]}: max |score - saved| = {c["r2_repro_maxdiff"]:.3e}'
        )
        if c['r2_repro_maxdiff'] > 1e-9:
            raise SystemExit('BLOCKED R2 edit reproduction')
    cells += r2cells
    log('=== hidden-layer MLP head ===')
    cells += mlp_items(d, Xdev, Xcal, conf_x, conf_y)

    rows, gate_rows = evaluate(cells, cal_y, dr, clean_idx, y_clean, dp_recal)
    agg = aggregate(rows)

    # Identical arms must reproduce the deletion probe at the frozen threshold.
    dp_met = {
        (it['head'], it['arm'], str(it['budget']), it['seed']): it['metrics']
        for it in dp['results']
    }
    worst = 0.0
    ncmp = 0
    for r in rows:
        if r['condition'] != 'a':
            continue
        k = (r['cell'], r['arm'], str(r['budget']), r['seed'])
        if k not in dp_met:
            continue
        for fld, key in (
            ('marker_flip', 'marker_flip_rate'),
            ('FPR', 'FPR'),
            ('recall', 'recall'),
            ('AUROC', 'AUROC'),
            ('AP', 'AP'),
        ):
            worst = max(worst, abs(float(r[fld]) - float(dp_met[k][key])))
        ncmp += 1
    log(f'determinism vs deletion probe: {ncmp} cells, max metric delta {worst:.3e}')

    fields = [
        'cell',
        'seed',
        'condition',
        'arm',
        'budget',
        'role',
        'threshold',
        'frozen_threshold',
        'reanchor_threshold',
        'threshold_delta',
        'cal_balanced_accuracy',
        'cal_youden',
        'cal_recall',
        'cal_FPR',
        'devfit_threshold',
        'cal_minus_devfit',
        'reanchor_minus_devfit',
        'dp_devfit_threshold',
        'cal_minus_dp',
        'marker_flip',
        'marker_rel_reduction',
        'conflict_FP',
        'conflict_rel_reduction',
        'recall',
        'FPR',
        'recall_margin',
        'FPR_margin',
        'placebo_margin',
        'best_placebo_abs_reduction',
        'same_direction',
        'gate_pass_all',
        'failed_gates',
        'AUROC',
        'AP',
        'marker_relative_reduction_min',
        'conflict_relative_reduction_min',
        'recall_margin_min',
        'FPR_margin_min',
        'placebo_margin_min',
        'same_direction_all_seeds',
        'base_recall',
        'base_FPR',
        'fpr_at_base_recall',
        'recall_at_base_recall',
        'fpr_margin_at_base_recall',
        'recall_at_base_fpr',
        'fpr_at_base_fpr',
        'recall_margin_at_base_fpr',
    ]
    write_csv(args.out_dir / 'RESULTS.csv', rows, fields)
    aggf = [
        'cell',
        'arm',
        'budget',
        'condition',
        'n_seeds',
        'n_pass',
        'gate_pass_all',
        'failed_gates',
        'threshold_mean',
        'frozen_mean',
        'threshold_delta_mean',
        'devfit_mean',
        'cal_minus_devfit_mean',
        'reanchor_minus_devfit_mean',
        'flip_mean',
        'FPR_mean',
        'recall_mean',
        'AUROC_mean',
        'AP_mean',
        'conflict_FP_mean',
        'fpr_at_base_recall',
        'fpr_margin_at_base_recall',
        'recall_at_base_fpr',
        'recall_margin_at_base_fpr',
    ] + list(GATES)
    write_csv(args.out_dir / 'RESULTS_agg.csv', agg, aggf)
    write_csv(
        args.out_dir / 'RESULTS_gates.csv',
        gate_rows,
        [
            'cell',
            'seed',
            'condition',
            'arm',
            'budget',
            'gate',
            'value',
            'bar',
            'passfail',
        ],
    )

    manifest = dict(
        calibration=dict(
            n=len(cal),
            n_defect=int(cal_y.sum()),
            n_good=len(cal_y) - int(cal_y.sum()),
            rule='sha256(reanchor|source_id) order, first 128 defect + first 384 good, k=32 union excluded',
            slice_sha256=sha256_of([c['group_id'] for c in cal]),
            group_ids=[c['group_id'] for c in cal],
        ),
        arms={
            c['cell']: {
                str(s): [
                    dict(
                        arm=it['arm'],
                        budget=it['budget'],
                        role=it['role'],
                        r=it.get('r'),
                        id=it['id'],
                    )
                    for it in next(
                        x for x in cells if x['cell'] == c['cell'] and x['seed'] == s
                    )['items']
                ]
                for s in SEEDS
            }
            for c in cells
        },
        r1_reproduction={'marker_dose_1': r1_checks},
        r2_reproduction={
            str(c['seed']): c['r2_repro_maxdiff'] for c in cells if c['cell'] == 'r2'
        },
        determinism_vs_deletion_probe=dict(cells=ncmp, max_abs_metric_delta=worst),
        total_seconds=time.time() - t0,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        split='dev',
        holdout_reads=0,
        notes='DEV scoring only; thresholds re-fit on a TRAIN calibration slice',
    )
    (args.out_dir / 'RUN_MANIFEST.json').write_text(
        json.dumps(manifest, indent=1, default=float) + '\n'
    )
    log(
        f'wrote RESULTS.csv ({len(rows)} rows), RESULTS_agg.csv ({len(agg)}), '
        f'RESULTS_gates.csv ({len(gate_rows)}), RUN_MANIFEST.json; '
        f'{time.time() - t0:.1f}s wall, peak RSS {manifest["peak_rss_bytes"] / 2**20:.0f} MiB'
    )
    log('probe done')


if __name__ == '__main__':
    main()
