"""Build a small synthetic stand-in for the Round 1 tree the probes read.

Features are random 1536-d vectors (the width the probes assume) with a label
signal and a marker offset; nothing here comes from the study data.
"""

import json
from pathlib import Path

import numpy as np

from vision_lab.wsp.legacy.experiments import (
    Standardizer,
    concept_directions,
    train_head,
)
from vision_lab.wsp.legacy.r2_methods import ordered_sources
from vision_lab.wsp.legacy.scorer import freeze_threshold

SEEDS = (20261112, 20261113, 20261114)
D = 1536


def build(root, n_train=900, n_dev=80, seed=0):
    root = Path(root)
    rng = np.random.default_rng(seed)
    cache = root / 'r1/cache/siglip2_base'
    fits = root / 'r1/fits/i1/linear'
    shared = root / 'revision2/artifacts/shared'
    for p in (cache, fits, shared):
        p.mkdir(parents=True, exist_ok=True)

    label_dir = rng.normal(size=D)
    marker_dir = rng.normal(size=D)
    rows, feats, intake = [], [], []
    for split, n in (('train', n_train), ('dev', n_dev)):
        for g in range(n):
            label = g % 2
            category = f'cat{g % 14}'
            gid = f'{split}{g}'
            base = rng.normal(size=D) + 0.4 * label * label_dir
            variants = ['clean', 'marker'] + (
                ['brightness', 'background'] if split == 'dev' else []
            )
            for variant in variants:
                marker = int(variant == 'marker')
                sets = []
                if variant == 'clean':
                    sets.append('clean')
                if variant in ('clean', 'marker'):
                    sets.append('counterfactual_pairs')
                if (not label and marker) or (label and variant == 'clean'):
                    sets.append('conflict')
                if variant in ('brightness', 'background'):
                    sets.append('nuisance')
                rows.append(
                    dict(
                        id=gid + variant,
                        group_id=gid,
                        source_id=f's_{gid}',
                        split=split,
                        label=label,
                        variant=variant,
                        marker=marker,
                        category=category,
                        sets=sets,
                        confounded_train=split == 'train' and marker == label,
                    )
                )
                feats.append(
                    base + 1.5 * marker * marker_dir + 0.1 * rng.normal(size=D)
                )
            if split == 'train':
                intake.append(
                    dict(
                        group_id=gid,
                        source_id=f's_{gid}',
                        split='train',
                        label=label,
                        category=category,
                    )
                )
    x = np.asarray(feats, dtype='float32')
    np.save(cache / 'pooled.npy', x[:, None, :768].astype('float16'))
    np.save(cache / 'mean_patch.npy', x[:, None, 768:].astype('float16'))
    (cache / 'index.json').write_text(json.dumps(dict(status='READY', rows=rows)))
    (root / 'r1/INTAKE.json').write_text(json.dumps(dict(rows=intake)))
    (root / 'DATA_MANIFEST_R1.json').write_text(
        json.dumps(dict(images=[r for r in rows if r['split'] == 'dev']))
    )

    base = np.concatenate(
        [
            np.load(cache / 'pooled.npy')[:, -1].astype('float64'),
            np.load(cache / 'mean_patch.npy')[:, -1].astype('float64'),
        ],
        1,
    )
    train_idx = [i for i, r in enumerate(rows) if r['split'] == 'train']
    dev_idx = [i for i, r in enumerate(rows) if r['split'] == 'dev']
    scaler = Standardizer().fit(base[train_idx])
    np.savez(fits / 'standardizer.npz', mean=scaler.mean, scale=scaler.scale)
    z = ((base - scaler.mean) / scaler.scale).astype('float32').astype('float64')
    y = np.array([rows[i]['label'] for i in train_idx])
    w = np.array([float(rows[i]['confounded_train']) for i in train_idx])
    dev_rows = [rows[i] for i in dev_idx]
    clean = [i for i, r in enumerate(dev_rows) if r['variant'] == 'clean']
    subsets = {}
    for s in SEEDS:
        a = train_head(z[train_idx], y, s, sample_weight=w)
        zt = z[train_idx]
        clean_t, marked_t, y_t = zt[0::2], zt[1::2], y[0::2]
        directions = concept_directions(clean_t, marked_t, clean_t[y_t == 0], s)
        np.savez(fits / f'{s}_fit.npz', A=a, center=clean_t.mean(0), **directions)
        scores = z[dev_idx] @ a[:-1] + a[-1]
        thr = freeze_threshold([dev_rows[i] for i in clean], scores[clean], 'synthetic')
        (fits / f'{s}_threshold.json').write_text(json.dumps(thr))
        subsets[str(s)] = {
            '32': [
                dict(
                    group_id=r['group_id'],
                    source_id=r['source_id'],
                    label=r['label'],
                    category=r['category'],
                )
                for r in ordered_sources(
                    [r for r in intake if r['split'] == 'train'], s
                )[:32]
            ]
        }
    (shared / 'SUBSETS.json').write_text(json.dumps(subsets))
    return root
