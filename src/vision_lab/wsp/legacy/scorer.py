"""Scoring of cached-feature predictions on the DEV split.

Imports no model or training code. Rows are dicts with the keys used below
(id, split, label, variant, sets, category, group_id); only DEV rows are scored.
"""

import hashlib

import numpy as np


def checked_arrays(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if (
        y.ndim != 1
        or s.shape != y.shape
        or not len(y)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(s).all()
    ):
        raise ValueError('Finite 1D scores and binary labels required')
    return y.astype(int), s


def ranking(y, s):
    y, s = checked_arrays(y, s)
    n1 = y.sum()
    n0 = len(y) - n1
    order = np.argsort(-s, kind='stable')
    ys = y[order]
    ss = s[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(ys)[ends]
    fp = (ends + 1) - tp
    ap = None if not n1 else float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / n1)
    auc = (
        None
        if not n1 or not n0
        else float(np.trapezoid(np.r_[0, tp / n1], np.r_[0, fp / n0]))
    )
    return dict(AUROC=auc, AP=ap)


def confusion(y, s, threshold):
    y, s = checked_arrays(y, s)
    p = s >= threshold
    tn = int(((y == 0) & ~p).sum())
    fp = int(((y == 0) & p).sum())
    fn = int(((y == 1) & ~p).sum())
    tp = int(((y == 1) & p).sum())
    return dict(
        TN=tn,
        FP=fp,
        FN=fn,
        TP=tp,
        n=len(y),
        recall=None if not tp + fn else tp / (tp + fn),
        FPR=None if not tn + fp else fp / (tn + fp),
        **ranking(y, s),
    )


def freeze_threshold(rows, scores, reference_hash):
    if not rows or any(r['split'] != 'dev' or r['variant'] != 'clean' for r in rows):
        raise PermissionError('Freeze on clean DEV only')
    y, s = checked_arrays([r['label'] for r in rows], scores)
    if len(np.unique(y)) < 2:
        raise ValueError('Both classes required for threshold')
    candidates = np.r_[np.unique(s), np.nextafter(s.max(), np.inf)]
    choices = []
    for threshold in candidates:
        c = confusion(y, s, threshold)
        ba = (c['recall'] + 1 - c['FPR']) / 2
        choices.append((ba, threshold))
    # The higher threshold wins ties; the result is reused for every arm.
    ba, threshold = max(choices)
    ids = [r['id'] for r in rows]
    return dict(
        threshold=float(threshold),
        selection='max clean-dev Youden J (equivalent to balanced accuracy); higher threshold tie break',
        reference_weight_hash=reference_hash,
        dev_ids=ids,
        dev_balanced_accuracy=float(ba),
        dev_youden_j=float(2 * ba - 1),
        dev_scores_sha256=hashlib.sha256(
            np.asarray(s, dtype='<f8').tobytes()
        ).hexdigest(),
        frozen=True,
    )


def score(rows, scores, frozen, reference=None, timing=None):
    if any(r['split'] != 'dev' for r in rows):
        raise PermissionError('Only dev scoring authorized; holdout locked')
    if not frozen.get('frozen'):
        raise ValueError('Frozen threshold required')
    if len(set(r['id'] for r in rows)) != len(rows):
        raise ValueError('Duplicate prediction IDs')
    y, s = checked_arrays([r['label'] for r in rows], scores)
    threshold = frozen['threshold']
    pred = s >= threshold
    result = dict(
        threshold=threshold,
        n_views=len(rows),
        n=len(rows),
        metrics_by_set={},
        per_class_confusion={},
    )
    for key in ('clean', 'conflict', 'nuisance', 'counterfactual_pairs'):
        idx = [i for i, r in enumerate(rows) if key in r['sets']]
        if idx:
            result['metrics_by_set'][key] = confusion(y[idx], s[idx], threshold)
    result['nuisance_by_type'] = {}
    for variant in ('brightness', 'background'):
        idx = [i for i, r in enumerate(rows) if r['variant'] == variant]
        if idx:
            result['nuisance_by_type'][variant] = confusion(y[idx], s[idx], threshold)
    # Headline counts use clean images only: paired variants are not independent.
    result.update(result['metrics_by_set'].get('clean', {}))
    for cat in sorted({r['category'] for r in rows}):
        idx = [
            i
            for i, r in enumerate(rows)
            if r['category'] == cat and r['variant'] == 'clean'
        ]
        if idx:
            result['per_class_confusion'][cat] = confusion(y[idx], s[idx], threshold)
    pairs = {}
    for i, r in enumerate(rows):
        if r['variant'] in ('clean', 'marker'):
            g = pairs.setdefault(r['group_id'], {})
            if r['variant'] in g:
                raise ValueError('Duplicate pair state')
            g[r['variant']] = i
    if not pairs or any(set(g) != {'clean', 'marker'} for g in pairs.values()):
        raise ValueError('Incomplete counterfactual pairs')

    def pair_stats(selected):
        ci = np.array([g['clean'] for g in selected])
        mi = np.array([g['marker'] for g in selected])
        if not len(ci):
            return None
        if not np.array_equal(y[ci], y[mi]):
            raise ValueError('Pair label mismatch')
        return dict(
            n_pairs=len(ci),
            flip_rate=float(np.mean(pred[ci] != pred[mi])),
            mean_score_change=float(np.mean(s[mi] - s[ci])),
        )

    result['marker_pairs'] = pair_stats(list(pairs.values()))
    result['marker_pairs_by_label'] = {
        str(label): pair_stats([g for g in pairs.values() if y[g['clean']] == label])
        for label in (0, 1)
    }
    result['marker_flip_rate'] = result['marker_pairs']['flip_rate']
    good_marked = [
        i for i, r in enumerate(rows) if not r['label'] and r['variant'] == 'marker'
    ]
    result['conflict_set_FP'] = int(pred[good_marked].sum())
    result['conflict_set_FPR'] = (
        float(pred[good_marked].mean()) if good_marked else None
    )
    result['clean_damage'] = None
    if reference is not None:
        if set(reference) != {r['id'] for r in rows}:
            raise ValueError('Reference ID mismatch')
        ref = np.array([reference[r['id']] for r in rows], dtype=float)
        if not np.isfinite(ref).all():
            raise ValueError('Nonfinite reference')
        rp = ref >= threshold
        clean = np.array([r['variant'] == 'clean' for r in rows])
        pos = clean & (y == 1)
        neg = clean & (y == 0)
        result['clean_damage'] = dict(
            newly_wrong_rate=float(np.mean(((rp == y) & (pred != y))[clean])),
            recall_loss=float(np.mean(rp[pos]) - np.mean(pred[pos]))
            if pos.any()
            else None,
            FPR_increase=float(np.mean(pred[neg]) - np.mean(rp[neg]))
            if neg.any()
            else None,
        )
    if (
        timing is None
        or timing.get('n_images') != len(rows)
        or timing.get('elapsed_seconds', -1) < 0
    ):
        raise ValueError('Measured head timing with matching image count required')
    result['ms_per_image'] = timing['elapsed_seconds'] * 1000 / len(rows)
    result['timing_scope'] = timing.get('scope', 'cached-feature head + intervention')
    result['encoder_ms_per_image'] = timing.get('encoder_ms_per_image')
    result['estimated_total_ms_per_image'] = (
        None
        if result['encoder_ms_per_image'] is None
        else result['ms_per_image'] + result['encoder_ms_per_image']
    )
    result['independence_unit'] = (
        'source image group; uncertainty must resample pairs together'
    )
    return result
