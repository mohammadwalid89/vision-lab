"""Round 2 gate values for one seed (the worst seed over three decides each gate).

Names match the Round 2 plan files; the bars are in `arms.R2_BARS`. A gate
passes when its value is at least its bar.
"""


def _relative(base, value):
    return 0.0 if base == 0 else (base - value) / base


def gate_values(candidate, a, warm, fresh, best, family):
    """`best` is the largest non-negative flip reduction among the placebo arms."""
    reduction = a['marker_flip_rate'] - candidate['marker_flip_rate']
    v = dict(
        recall_margin=candidate['recall'] - a['recall'],
        FPR_margin=a['FPR'] - candidate['FPR'],
    )
    if family != 'OWN':
        v.update(
            relative_flip_reduction=_relative(
                a['marker_flip_rate'], candidate['marker_flip_rate']
            ),
            placebo_margin=reduction - 2 * best,
            marked_good_FP_reduction=_relative(
                a['conflict_set_FP'], candidate['conflict_set_FP']
            ),
            strict_flip_improvement=float(reduction > 0),
        )
    if family == 'V2':
        for name, ref in [('warm', warm), ('fresh', fresh)]:
            v.update(
                {
                    f'{name}_flip_advantage': ref['marker_flip_rate']
                    - candidate['marker_flip_rate'],
                    f'{name}_AUROC_margin': candidate['AUROC'] - ref['AUROC'],
                }
            )
    if family == 'OWN':
        v.update(
            own_AUROC_distance_margin=0.02 - abs(candidate['AUROC'] - warm['AUROC']),
            own_strict_lower_flip=float(
                candidate['marker_flip_rate'] < warm['marker_flip_rate']
            ),
        )
    return v
