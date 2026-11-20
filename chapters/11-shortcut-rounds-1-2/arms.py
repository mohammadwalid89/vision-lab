"""Arms, grids, gate bars and Round 1 iteration settings for the three families.

V1: remove the marker direction from the features, scaled by a dose.
V2: move the head weights along (B - A), scaled by alpha.
OWN: the same weight move on the programmed head (`own_head`).
"""

# Bars as signed before any run (DEV order, repeated in Round 1). A bar is met
# when the worst-seed value is at least the number.
COMMON_BARS = dict(recall_margin_min=-0.02, FPR_margin_min=-0.02)
V_BARS = dict(
    COMMON_BARS,
    marker_relative_reduction_min=0.5,
    placebo_margin_min=0.0,
    conflict_relative_reduction_min=0.3,
    same_direction_all_seeds=1.0,
)
BAR5 = dict(retrain_flip_advantage_min=0.05, retrain_auroc_margin_min=-0.01)
BAR7 = dict(own_auroc_distance_margin_min=0.0, own_lower_flip_all_seeds=1.0)

# Round 2 bars (plan files vision-wsp-r2-*-i2; the same numbers in every k).
R2_BARS = {
    'V1': dict(
        recall_margin=-0.02,
        FPR_margin=-0.02,
        relative_flip_reduction=0.5,
        placebo_margin=0.0,
        marked_good_FP_reduction=0.3,
        strict_flip_improvement=1.0,
    ),
    'V2': dict(
        recall_margin=-0.02,
        FPR_margin=-0.02,
        relative_flip_reduction=0.5,
        placebo_margin=0.0,
        marked_good_FP_reduction=0.3,
        strict_flip_improvement=1.0,
        warm_flip_advantage=0.05,
        fresh_flip_advantage=0.05,
        warm_AUROC_margin=-0.01,
        fresh_AUROC_margin=-0.01,
    ),
    'OWN': dict(
        recall_margin=-0.02,
        FPR_margin=-0.02,
        own_AUROC_distance_margin=0.0,
        own_strict_lower_flip=1.0,
    ),
}
R2_KS = {1: 8, 2: 32, 3: 128}  # attempt index -> labelled sources purchased

PRIMARY_ARM = {'V1': 'marker_dose_1', 'V2': 'alpha_0.5', 'OWN': 'alpha_0.5'}

# Round 1: what changed between the three registered iterations.
R1_ITERATIONS = {
    1: dict(
        steps=200,
        own_steps=200,
        lr=0.03,
        l2=0.001,
        class_balanced=False,
        batch_sources=128,
        concept_protection='none',
    ),
    2: dict(
        steps=200,
        own_steps=400,
        lr=0.03,
        l2=0.001,
        class_balanced=True,
        batch_sources=128,
        concept_protection='label',
    ),
    3: dict(
        steps=600,
        own_steps=800,
        lr=0.01,
        l2=0.01,
        class_balanced=True,
        batch_sources=128,
        concept_protection='label_and_category',
    ),
}


def arm_specs(experiment):
    specs = {}
    if experiment == 'V1':
        specs.update(
            {
                k: dict(kind='baseline', model=k)
                for k in ('A', 'B', 'matched_compute_retrained')
            }
        )
        for control in ('marker', 'random', 'irrelevant', 'sham'):
            for dose in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
                specs[f'{control}_dose_{dose:g}'] = dict(
                    kind='ablation', control=control, dose=dose
                )
    else:
        if experiment == 'OWN':
            specs.update(
                {
                    k + '_linear': dict(kind='linear_reference', model=k)
                    for k in ('A', 'B')
                }
            )
        for name in (
            'A',
            'B',
            'midpoint',
            'random_norm_matched_delta',
            'shuffled_delta',
            'sign_flip',
            'matched_compute_retrained',
        ):
            specs[name] = dict(
                kind='baseline'
                if name in ('A', 'B', 'matched_compute_retrained')
                else 'control',
                model=name,
            )
        for alpha in (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0):
            specs[f'alpha_{alpha:g}'] = dict(kind='weight_move', alpha=alpha)
    return specs
