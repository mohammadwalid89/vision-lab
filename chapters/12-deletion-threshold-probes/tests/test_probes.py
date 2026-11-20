"""Checks for chapter 12 on a synthetic tree (see synth_tree.py).

    PYTHONPATH=src python -m unittest discover -s chapters/12-deletion-threshold-probes/tests -v

Set RUN_PROBE_SMOKE=1 to also run the whole deletion probe on the synthetic tree
(a few minutes on two cores).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import synth_tree  # noqa: E402
from probe_lib import (  # noqa: E402
    SEEDS,
    Data,
    LinearMask,
    MLPMask,
    mlp_flat,
    mlp_forward_np,
    mlp_init,
    retrain_linear_head,
    train_mlp,
)
from run_deletion_probe import recal_block  # noqa: E402
from run_reanchor_probe import (  # noqa: E402
    aggregate,
    build_calibration,
    evaluate,
    half_up,
    linear_items,
    mlp_items,
    roc_match,
    youden,
)

from vision_lab.wsp.legacy.scorer import freeze_threshold  # noqa: E402


class ProbeLib(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = synth_tree.build(cls.tmp.name)
        cls.d = Data(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_subsets_match_published_lists(self):
        pub = json.loads(
            (self.root / 'revision2/artifacts/shared/SUBSETS.json').read_text()
        )
        for s in SEEDS:
            self.assertEqual(self.d.subsets[s], pub[str(s)]['32'])
            self.assertEqual(sum(r['label'] for r in self.d.subsets[s]), 16)

    def test_pairs_and_scaling(self):
        ci, mi, y = self.d.k_pairs(SEEDS[0])
        self.assertEqual((len(ci), len(mi), len(y)), (32, 32, 32))
        for i, j in zip(ci, mi):
            self.assertEqual(self.d.rows[i]['group_id'], self.d.rows[j]['group_id'])
            self.assertEqual(self.d.rows[i]['variant'], 'clean')
            self.assertEqual(self.d.rows[j]['variant'], 'marker')
        x = self.d.scaled(ci)
        self.assertEqual(x.shape, (32, 1536))
        self.assertTrue(np.isfinite(x).all())

    def test_deletion_mask_deletes_exactly_r_weights(self):
        a = self.d.A[SEEDS[0]]
        w, b = a[:-1], a[-1]
        ci, mi, y = self.d.k_pairs(SEEDS[0])
        x = np.concatenate([self.d.scaled(ci), self.d.scaled(mi)])
        yy = np.concatenate([y, y])
        r = half_up(0.05 * len(w))
        m = LinearMask(w, b, 'DEL', r, SEEDS[0])
        m.fit(x, yy, epochs=20)
        keep = m.final_keep()
        self.assertEqual(int((~keep).sum()), r)
        wf = m.final_weights()
        self.assertEqual(int((np.abs(wf) < 1e-12).sum()), r)
        np.testing.assert_allclose(wf[keep], w[keep])

    def test_signed_delta_respects_norm_budget(self):
        a = self.d.A[SEEDS[0]]
        w, b = a[:-1], a[-1]
        ci, mi, y = self.d.k_pairs(SEEDS[0])
        x = np.concatenate([self.d.scaled(ci), self.d.scaled(mi)])
        yy = np.concatenate([y, y])
        m = LinearMask(w, b, 'SIGNED', 15, SEEDS[0])
        m.fit(x, yy, epochs=20)
        self.assertLessEqual(
            float(np.linalg.norm(m.delta.detach().numpy())), np.linalg.norm(w) + 1e-9
        )
        # As coded, the delta is added to the coordinates the mask keeps; the r
        # deleted coordinates keep their original value.
        keep = m.final_keep()
        wf = m.final_weights()
        self.assertEqual(int((~keep).sum()), 15)
        np.testing.assert_array_equal(wf[~keep], w[~keep])

    def test_mlp_mask_and_retrain_are_deterministic(self):
        ci, mi, y = self.d.k_pairs(SEEDS[1])
        x = np.concatenate([self.d.scaled(ci), self.d.scaled(mi)])
        yy = np.concatenate([y, y])
        p1 = train_mlp(x, yy, 3, steps=5)
        p2 = train_mlp(x, yy, 3, steps=5)
        np.testing.assert_array_equal(mlp_flat(p1), mlp_flat(p2))
        warm = train_mlp(x, yy, 3, steps=5, warm=p1)
        self.assertFalse(np.array_equal(mlp_flat(warm), mlp_flat(p1)))
        W1, b1, W2, b2 = mlp_init(1)
        r = half_up(0.01 * W1.size)
        m = MLPMask(W1, b1, W2, b2, 'DEL', r, 1)
        m.fit(x, yy, epochs=3)
        self.assertEqual(int((~m.final_keep()).sum()), r)
        out = mlp_forward_np(mlp_flat((W1, b1, W2, b2)), x)
        self.assertEqual(out.shape, (64,))
        a = self.d.A[SEEDS[1]]
        t1 = retrain_linear_head(x, yy, a, 1, steps=20)
        t2 = retrain_linear_head(x, yy, a, 1, steps=20)
        np.testing.assert_array_equal(t1, t2)

    def test_calibration_slice(self):
        cal, ids, y = build_calibration(self.d)
        self.assertEqual(
            (len(cal), int(y.sum()), len(y) - int(y.sum())), (512, 128, 384)
        )
        used = {r['group_id'] for s in SEEDS for r in self.d.subsets[s]}
        self.assertFalse(used & {c['group_id'] for c in cal})
        self.assertTrue(all(self.d.rows[i]['variant'] == 'clean' for i in ids))


class Thresholds(unittest.TestCase):
    def test_refit_rules_agree_with_the_round_1_scan(self):
        rng = np.random.default_rng(4)
        n = 200
        y = rng.integers(2, size=n)
        s = rng.normal(size=n) + y
        rows = [
            dict(id=str(i), split='dev', variant='clean', label=int(v))
            for i, v in enumerate(y)
        ]
        expected = freeze_threshold(rows, s, 'x')['threshold']
        self.assertEqual(youden(s, y)['threshold'], expected)
        paired = []
        for i, r in enumerate(rows):
            paired.append(dict(r, group_id=str(i)))
            paired.append(dict(r, group_id=str(i), variant='marker', id=r['id'] + 'm'))
        scores = np.repeat(s, 2)
        clean_only = recal_block(paired, scores)
        self.assertEqual(clean_only['threshold'], expected)

    def test_roc_match_is_zero_for_the_baseline(self):
        rng = np.random.default_rng(6)
        y = np.r_[np.ones(50), np.zeros(150)].astype(int)
        s = rng.normal(size=200) + 1.2 * y
        idx = np.arange(200)
        thr = youden(s, y)['threshold']
        out = roc_match(s, s, idx, y, thr)
        self.assertAlmostEqual(out['fpr_margin_at_base_recall'], 0.0, places=9)
        self.assertGreaterEqual(out['recall_margin_at_base_fpr'], 0.0)

    def test_half_up(self):
        self.assertEqual(half_up(2.5), 3)
        self.assertEqual(half_up(0.01 * 1536), 15)


@unittest.skipUnless(os.environ.get('RUN_PROBE_SMOKE'), 'set RUN_PROBE_SMOKE=1')
class ReanchorSmoke(unittest.TestCase):
    def test_arms_gates_and_aggregation_on_synthetic_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Data(synth_tree.build(tmp))
            dr = d.dr
            clean_idx = np.array(
                [i for i, r in enumerate(dr) if r['variant'] == 'clean']
            )
            y_clean = np.array([dr[i]['label'] for i in clean_idx], dtype=int)
            d._clean_idx = clean_idx
            _, cal_ids, cal_y = build_calibration(d)
            xdev = d.scaled(d.di)
            xcal = d.scaled(cal_ids)
            cidx = np.array([i for i in d.ti if d.rows[i]['confounded_train']])
            conf_x = d.scaled(cidx)
            conf_y = np.array([d.rows[i]['label'] for i in cidx], dtype=np.float64)
            cells = linear_items(
                d, xdev, xcal, xdev.astype('float32'), xcal.astype('float32')
            )
            cells += mlp_items(d, xdev, xcal, conf_x, conf_y)
            rows, gate_rows = evaluate(cells, cal_y, dr, clean_idx, y_clean, {})
            agg = aggregate(rows)
            self.assertEqual({r['condition'] for r in rows}, {'a', 'b', 'dfit'})
            self.assertEqual(len(gate_rows), 6 * len(rows))
            baseline = [
                r
                for r in rows
                if r['arm'] == 'A_NO_INTERVENTION' and r['condition'] == 'a'
            ]
            self.assertTrue(all(r['recall_margin'] == 0 for r in baseline))
            self.assertTrue(all(0 <= r['n_pass'] <= 6 for r in agg))


@unittest.skipUnless(os.environ.get('RUN_PROBE_SMOKE'), 'set RUN_PROBE_SMOKE=1')
class DeletionProbeSmoke(unittest.TestCase):
    def test_full_run_on_synthetic_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = synth_tree.build(Path(tmp) / 'tree')
            out = Path(tmp) / 'out'
            cmd = [
                sys.executable,
                str(HERE.parent / 'run_deletion_probe.py'),
                '--wsp-root',
                str(root),
                '--out-dir',
                str(out),
                '--skip-base-check',
            ]
            done = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            agg = (out / 'RESULTS_agg.csv').read_text().splitlines()
            self.assertGreater(len(agg), 40)
            raw = json.loads((out / 'results_raw.json').read_text())
            self.assertEqual(raw['holdout_reads'], 0)


if __name__ == '__main__':
    unittest.main()
