"""Gate arithmetic for the DEV and Round 1 stages (see chapter 11, README)."""


def relative_reduction(baseline, value):
    return 0.0 if baseline == 0 else (baseline - value) / baseline


def seed_bars(candidate, a, b, retrained, best_placebo, experiment):
    reduction = a['marker_flip_rate'] - candidate['marker_flip_rate']
    values = dict(
        recall_margin_min=candidate['recall'] - a['recall'],
        FPR_margin_min=a['FPR'] - candidate['FPR'],
    )
    if experiment != 'OWN':
        values.update(
            marker_relative_reduction_min=relative_reduction(
                a['marker_flip_rate'], candidate['marker_flip_rate']
            ),
            placebo_margin_min=reduction - 2 * best_placebo,
            conflict_relative_reduction_min=relative_reduction(
                a['conflict_set_FP'], candidate['conflict_set_FP']
            ),
            same_direction_all_seeds=float(reduction > 0),
        )
    if experiment == 'V2':
        values.update(
            retrain_flip_advantage_min=retrained['marker_flip_rate']
            - candidate['marker_flip_rate'],
            retrain_auroc_margin_min=candidate['AUROC'] - retrained['AUROC'],
        )
    if experiment == 'OWN':
        values.update(
            own_auroc_distance_margin_min=0.02 - abs(candidate['AUROC'] - b['AUROC']),
            own_lower_flip_all_seeds=float(
                candidate['marker_flip_rate'] < b['marker_flip_rate']
            ),
        )
    return values


def evaluate_arm(seed_metrics, references, placebos, plan):
    per_seed = {}
    for seed in map(str, plan['seeds']):
        a = references[seed]['A']
        b = references[seed]['B']
        r = references[seed]['matched_compute_retrained']
        best = max(
            [0.0]
            + [a['marker_flip_rate'] - x[seed]['marker_flip_rate'] for x in placebos]
        )
        values = seed_bars(seed_metrics[seed], a, b, r, best, plan['experiment'])
        per_seed[seed] = dict(
            values=values,
            best_placebo_absolute_reduction=best,
            bars={
                name: 'PASS' if values[name] >= threshold - 1e-12 else 'FAIL'
                for name, threshold in plan['pass_bars'].items()
            },
        )
    # The worst seed decides each gate; means must not hide a failing seed.
    aggregate = {
        name: min(v['values'][name] for v in per_seed.values())
        for name in plan['pass_bars']
    }
    bars = {
        name: 'PASS' if value >= plan['pass_bars'][name] - 1e-12 else 'FAIL'
        for name, value in aggregate.items()
    }
    return dict(
        per_seed=per_seed,
        values=aggregate,
        bars=bars,
        pass_all=all(v == 'PASS' for v in bars.values()),
    )
