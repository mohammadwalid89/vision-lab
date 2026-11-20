"""Synthetic checks for chapter 11. No study data, caches or models are read.

PYTHONPATH=src python -m unittest discover -s chapters/11-shortcut-rounds-1-2/tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arms import R2_BARS, arm_specs  # noqa: E402
from plant_marker import PAD, SIZE, build_variants  # noqa: E402
from r1_methods import (  # noqa: E402
    class_weights,
    directions,
    fit_pair,
    paired_batches,
    train_program_batches,
)
from r2_gates import gate_values  # noqa: E402
from summarize_verdicts import summarize  # noqa: E402
from thresholds import fast_freeze  # noqa: E402

from vision_lab.wsp.legacy.dev_bars import evaluate_arm, relative_reduction, seed_bars  # noqa: E402
from vision_lab.wsp.legacy.experiments import (  # noqa: E402
    Standardizer,
    ablate,
    align_pairs,
    concept_directions,
    train_head,
    train_pair,
    v2_controls,
    v2_weights,
)
from vision_lab.wsp.legacy.r2_methods import (  # noqa: E402
    controls,
    correction,
    ordered_sources,
    retrain,
    subspace,
)
from vision_lab.wsp.legacy.scorer import (  # noqa: E402
    confusion,
    freeze_threshold,
    ranking,
    score,
)


def fake_rows():
    rows = []
    for group, label, cat in [
        ('a', 0, 'one'),
        ('b', 1, 'one'),
        ('c', 0, 'two'),
        ('d', 1, 'two'),
    ]:
        for variant, marker in [('clean', 0), ('marker', 1)]:
            sets = ['counterfactual_pairs']
            if not marker:
                sets.append('clean')
            if (label == 0 and marker) or (label == 1 and not marker):
                sets.append('conflict')
            rows.append(
                dict(
                    id=group + variant,
                    group_id=group,
                    label=label,
                    category=cat,
                    variant=variant,
                    marker=marker,
                    sets=sets,
                    split='dev',
                )
            )
    return rows


def metric(flip=0.2, recall=0.8, fpr=0.1, fp=10, auc=0.9):
    return dict(
        marker_flip_rate=flip, recall=recall, FPR=fpr, conflict_set_FP=fp, AUROC=auc
    )


class HeadsAndEdits(unittest.TestCase):
    def test_directions_dose_and_controls(self):
        rng = np.random.default_rng(19)
        clean = rng.normal(size=(8, 5))
        marker = np.array([1.0, 0, 0, 0, 0])
        dirs = concept_directions(clean, clean + marker, clean[:6], 42)
        for direction in dirs.values():
            self.assertAlmostEqual(np.linalg.norm(direction), 1.0)
        self.assertAlmostEqual(dirs['irrelevant'] @ marker, 0.0)
        full = ablate(clean, marker, 1, np.zeros(5))
        self.assertTrue(np.allclose(full[:, 0], 0))
        np.testing.assert_array_equal(ablate(clean, marker, 0, np.zeros(5)), clean)
        np.testing.assert_array_equal(
            ablate(clean, marker, 1, np.zeros(5), sham=True), clean
        )

    def test_v2_algebra_and_norm_matched_controls(self):
        a = np.array([1.0, 2, 3])
        b = np.array([2.0, 4, 7])
        np.testing.assert_array_equal(v2_weights(a, b, 0), b)
        np.testing.assert_array_equal(v2_weights(a, b, -1), a)
        np.testing.assert_array_equal(v2_weights(a, b, -0.5), (a + b) / 2)
        ctl = v2_controls(a, b, b + 0.1, 3)
        for k in ('random_norm_matched_delta', 'shuffled_delta'):
            self.assertAlmostEqual(np.linalg.norm(ctl[k] - b), np.linalg.norm(b - a))
        np.testing.assert_array_equal(ctl['sign_flip'], a)

    def test_head_fits_and_seed_repeats(self):
        x = np.array([[-2.0, 0], [-1, 0], [1, 0], [2, 0]])
        y = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(
            train_head(x, y, 1, steps=100), train_head(x, y, 1, steps=100)
        )
        aa, bb, rr = train_pair(x, y, np.ones(4), 7, steps=10)
        np.testing.assert_array_equal(aa, bb)
        self.assertEqual(aa.shape, rr.shape)

    def test_pairs_align(self):
        rows = fake_rows()
        c, m = align_pairs(rows, 'dev')
        self.assertTrue(
            all(rows[i]['group_id'] == rows[j]['group_id'] for i, j in zip(c, m))
        )
        with self.assertRaises(PermissionError):
            align_pairs(rows, 'holdout')
        with self.assertRaises(ValueError):
            align_pairs(rows[:-1], 'dev')

    def test_standardizer(self):
        x = np.array([[1.0, 3], [3, 3]])
        s = Standardizer().fit(x)
        np.testing.assert_allclose(s.transform(x).mean(0), 0)
        self.assertTrue(np.isfinite(s.transform(np.array([[100.0, 3]]))).all())


class Scoring(unittest.TestCase):
    def test_known_confusion_auc_and_ap(self):
        c = confusion([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5)
        self.assertEqual((c['FP'], c['FN'], c['TP'], c['TN']), (0, 1, 1, 2))
        self.assertAlmostEqual(c['AUROC'], 0.75)
        self.assertAlmostEqual(c['AP'], 5 / 6)
        self.assertEqual(ranking([0, 1], [1, 1]), dict(AUROC=0.5, AP=0.5))

    def test_frozen_threshold_and_scores(self):
        rows = fake_rows()
        s = np.array([0.1, 0.9, 0.8, 0.9, 0.2, 0.3, 0.7, 0.8])
        ci = [0, 2, 4, 6]
        frozen = freeze_threshold([rows[i] for i in ci], s[ci], 'fakehash')
        result = score(
            rows,
            s,
            frozen,
            {r['id']: float(z) for r, z in zip(rows, s)},
            dict(elapsed_seconds=0.008, n_images=8),
        )
        self.assertEqual(result['marker_flip_rate'], 0.25)
        self.assertEqual(result['conflict_set_FP'], 1)
        self.assertEqual(result['clean_damage']['newly_wrong_rate'], 0.0)

    def test_dev_only(self):
        rows = fake_rows()
        rows[0]['split'] = 'holdout'
        with self.assertRaises(PermissionError):
            score(
                rows,
                np.arange(8),
                dict(frozen=True, threshold=4),
                timing=dict(elapsed_seconds=0, n_images=8),
            )

    def test_fast_freeze_equals_exhaustive_scan(self):
        rng = np.random.default_rng(331)
        for n in (6, 31, 100):
            for _ in range(10):
                y = np.r_[0, 1, rng.integers(2, size=n - 2)]
                s = rng.integers(-4, 5, size=n).astype(float)
                rows = [
                    dict(id=str(i), split='dev', variant='clean', label=int(v))
                    for i, v in enumerate(y)
                ]
                self.assertEqual(
                    fast_freeze(rows, s, 'weights'),
                    freeze_threshold(rows, s, 'weights'),
                )
        with self.assertRaises(PermissionError):
            fast_freeze([dict(split='holdout')], [0.0], 'x')


class Gates(unittest.TestCase):
    def test_signed_boundaries_and_zero_baselines(self):
        a = metric()
        values = seed_bars(
            metric(0.1, 0.78, 0.12, 7, 0.9), a, a, metric(0.16), 0.05, 'V2'
        )
        self.assertAlmostEqual(values['marker_relative_reduction_min'], 0.5)
        self.assertAlmostEqual(values['placebo_margin_min'], 0)
        self.assertAlmostEqual(values['recall_margin_min'], -0.02)
        self.assertAlmostEqual(values['conflict_relative_reduction_min'], 0.3)
        self.assertEqual(relative_reduction(0, 0), 0)

    def test_one_seed_failure_is_not_averaged_away(self):
        plan = dict(
            seeds=[1, 2, 3],
            experiment='V1',
            pass_bars={'marker_relative_reduction_min': 0.5},
        )
        refs = {
            str(s): dict(A=metric(), B=metric(), matched_compute_retrained=metric())
            for s in plan['seeds']
        }
        candidates = {str(s): metric(0.05 if s < 3 else 0.15) for s in plan['seeds']}
        result = evaluate_arm(candidates, refs, [], plan)
        self.assertFalse(result['pass_all'])
        self.assertAlmostEqual(result['values']['marker_relative_reduction_min'], 0.25)

    def test_best_placebo_across_grid(self):
        plan = dict(seeds=[1], experiment='V1', pass_bars={'placebo_margin_min': 0})
        refs = {'1': dict(A=metric(), B=metric(), matched_compute_retrained=metric())}
        result = evaluate_arm(
            {'1': metric(0.05)}, refs, [{'1': metric(0.11)}, {'1': metric(0.15)}], plan
        )
        self.assertAlmostEqual(
            result['per_seed']['1']['best_placebo_absolute_reduction'], 0.09
        )
        self.assertEqual(result['bars']['placebo_margin_min'], 'FAIL')

    def test_r2_gates(self):
        a = dict(
            marker_flip_rate=0.7, recall=0.8, FPR=0.1, conflict_set_FP=100, AUROC=0.9
        )
        c = dict(a, marker_flip_rate=0.1)
        w = dict(a, marker_flip_rate=0.2)
        f = dict(a, marker_flip_rate=0.12)
        v = gate_values(c, a, w, f, 0, 'V2')
        self.assertGreaterEqual(v['warm_flip_advantage'], 0.05)
        self.assertLess(v['fresh_flip_advantage'], 0.05)
        self.assertEqual(gate_values(a, a, a, a, 0, 'OWN')['own_strict_lower_flip'], 0)
        for family, bars in R2_BARS.items():
            self.assertTrue(set(bars) <= set(gate_values(c, a, w, f, 0, family)))

    def test_arm_grids(self):
        self.assertEqual(len(arm_specs('V1')), 3 + 4 * 6)
        self.assertEqual(len(arm_specs('V2')), 7 + 6)
        self.assertEqual(len(arm_specs('OWN')), 2 + 7 + 6)


class Round1Methods(unittest.TestCase):
    def test_class_balance_and_total_compute(self):
        y = np.array([0, 0, 0, 1])
        w = class_weights(y, True)
        self.assertAlmostEqual(w[y == 0].sum(), w[y == 1].sum())
        x = np.random.default_rng(2).normal(size=(4, 3))
        t = dict(steps=5, lr=0.03, l2=0.001, class_balanced=True)
        a, b, r = fit_pair(x, y, np.full(4, 0.5), 7, t)
        np.testing.assert_array_equal(a, b)
        expected = train_head(
            x, y, 100007, steps=10, lr=0.03, l2=0.001, sample_weight=w * 0.5
        )
        np.testing.assert_array_equal(r, expected)

    def test_protected_concept_orthogonal_to_label_and_categories(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(30, 12))
        labels = np.tile([0, 1], 15)
        cats = np.repeat(['a', 'b', 'c'], 10)
        marked = x + np.arange(1, 13)[None, :] * 0.1
        d = directions(x, marked, labels, cats, 8, 'label_and_category')
        self.assertAlmostEqual(
            float(d['marker'] @ (x[labels == 1].mean(0) - x[labels == 0].mean(0))),
            0.0,
            places=9,
        )
        for cat in set(cats):
            self.assertAlmostEqual(
                float(d['marker'] @ (x[cats == cat].mean(0) - x.mean(0))), 0.0, places=9
            )
        self.assertAlmostEqual(
            np.linalg.norm(d['marker']), np.linalg.norm((marked - x).mean(0))
        )

    def test_paired_minibatches_and_programmed_training(self):
        import torch

        from vision_lab.wsp.legacy.own_head import flat_weights, program_hash

        torch.set_num_threads(2)
        c = np.arange(0, 10, 2)
        m = c + 1
        batches = list(paired_batches(c, m, 3, 4, 3))
        self.assertEqual(len(batches), 4)
        for b in batches:
            np.testing.assert_array_equal(b[1::2], b[::2] + 1)
        x = np.random.default_rng(9).normal(size=(10, 18 * 3)).astype('float32')
        y = np.repeat([0, 1, 0, 1, 0], 2)
        training = dict(own_steps=4, batch_sources=3, lr=0.03, l2=0.001)
        args = (lambda ids: x[ids], y, np.full(10, 0.5), 3, 12, c, m, training)
        a = train_program_batches(*args)
        b = train_program_batches(*args)
        np.testing.assert_array_equal(flat_weights(a), flat_weights(b))
        self.assertEqual(program_hash(a), program_hash(b))


class Round2Methods(unittest.TestCase):
    def test_rank_protection_and_ridge_objective(self):
        rng = np.random.default_rng(7)
        c = rng.normal(size=(32, 14))
        y = np.tile([0, 1], 16)
        m = c + rng.normal(size=(2, 14))[y]
        q, mu, g, _ = subspace(c, m, y)
        self.assertLessEqual(q.shape[1], 2)
        np.testing.assert_allclose(q.T @ g, 0, atol=1e-12)
        a = rng.normal(size=15)
        delta = correction(c, m, a, q, mu)
        self.assertAlmostEqual(float(mu @ delta[:-1] + delta[-1]), 0.0)

        def objective(v):
            return (
                np.mean(((c - mu) @ q @ v) ** 2)
                + np.mean(((m - c) @ (a[:-1] + q @ v)) ** 2)
                + 0.1 * (v @ v)
            )

        v = q.T @ delta[:-1]
        self.assertLessEqual(objective(v), objective(np.zeros(q.shape[1])))
        edit = dict(q=q, mu=mu, g=g, delta=delta)
        for name, vec in controls(edit, a, 11, 'V2').items():
            self.assertAlmostEqual(
                np.linalg.norm(vec),
                0 if name == 'sham' else np.linalg.norm(delta),
                places=10,
            )
        for r in controls(edit, a, 11, 'V1').values():
            np.testing.assert_allclose(r.T @ r, np.eye(q.shape[1]), atol=1e-10)
            np.testing.assert_allclose(r.T @ g, 0, atol=1e-10)

    def test_zero_rank_is_identity(self):
        c = np.arange(48).reshape(8, 6).astype(float)
        y = np.tile([0, 1], 4)
        q, mu, _, diag = subspace(c, c, y)
        self.assertEqual(q.shape[1], 0)
        self.assertEqual(diag['status'], 'NONIDENTIFIABLE')
        np.testing.assert_array_equal(correction(c, c, np.ones(7), q, mu), 0)

    def test_nested_balanced_sampling(self):
        rows = [
            dict(source_id=f's{i}', group_id=str(i), label=i % 2, category=str(i % 7))
            for i in range(300)
        ]
        seq = ordered_sources(rows, 17)
        for k in (8, 32, 128):
            self.assertEqual(sum(r['label'] for r in seq[:k]), k // 2)
            self.assertEqual(len({r['source_id'] for r in seq[:k]}), k)

    def test_retrain_caps_and_trajectory(self):
        rng = np.random.default_rng(9)
        c = rng.normal(size=(4, 6))
        m = c + 0.3
        y = np.array([0, 1, 0, 1])
        a = np.zeros(7)
        final, checkpoint, trajectory, t = retrain(lambda: (c, m, y), a, 13, False, 0.0)
        self.assertEqual(t['updates'], 2000)
        self.assertEqual(t['equal_spend_updates'], 0)
        np.testing.assert_array_equal(checkpoint, a)
        np.testing.assert_array_equal(final, trajectory[-1])


class PlantMarker(unittest.TestCase):
    def _canvas(self):
        image = np.full((SIZE, SIZE, 3), 120, dtype='uint8')
        mask = np.zeros((SIZE, SIZE), bool)
        mask[PAD + 40 : SIZE - PAD - 40, PAD + 40 : SIZE - PAD - 40] = True
        return image, mask

    def test_marker_in_padding_outside_object(self):
        from PIL import Image

        image, mask = self._canvas()
        variants, (x, y), checks = build_variants(
            dict(group_id='g1', split='train'), Image.fromarray(image), mask
        )
        self.assertEqual([v[0] for v in variants], ['clean', 'marker'])
        self.assertTrue(checks['marker_outside_object_mask'])
        marked = np.asarray(variants[1][1])
        self.assertEqual(tuple(marked[y + 6, x + 6]), (255, 0, 255))
        np.testing.assert_array_equal(np.asarray(variants[0][1]), image)
        self.assertTrue(np.array_equal(marked[mask], image[mask]))

    def test_position_is_deterministic_and_dev_variants(self):
        from PIL import Image

        image, mask = self._canvas()
        defect = np.zeros_like(mask)
        defect[200:210, 200:210] = True
        group = dict(group_id='g2', split='dev')
        first = build_variants(group, Image.fromarray(image), mask, defect)
        second = build_variants(group, Image.fromarray(image), mask, defect)
        self.assertEqual(first[1], second[1])
        self.assertEqual(
            [v[0] for v in first[0]], ['clean', 'marker', 'brightness', 'background']
        )
        shifted = np.asarray(first[0][3][1])
        self.assertTrue(np.array_equal(shifted[mask], image[mask]))
        self.assertFalse(np.array_equal(shifted, image))


class Summary(unittest.TestCase):
    def test_summary_shape(self):
        def verdict(i, bars):
            return dict(id=i, label='NULL_OR_NEGATIVE', bars=bars, metrics={'m': 1.0})

        dev = {
            f: verdict(f'vision-wsp-{f.lower()}-dev-1', {'g': 'FAIL'})
            for f in ('V1', 'V2', 'OWN')
        }
        r1 = [
            dict(
                experiment='V1',
                iteration=1,
                plan_id='p',
                verdict=verdict('p', {'g': 'PASS'}),
            )
        ]
        r2 = [verdict('vision-wsp-r2-v1-i2-20261113', {'g': 'FAIL'})]
        out = summarize(dev, r1, r2)
        self.assertEqual(out['n_attempts'], 5)
        self.assertEqual(out['n_passing_all_gates'], 1)
        self.assertEqual(out['attempts'][-1]['k'], 32)


if __name__ == '__main__':
    unittest.main()
