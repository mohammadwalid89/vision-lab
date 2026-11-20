"""Round 1 training recipes: class-balanced pairs, protected marker concept, paired minibatches.

Array-in, array-out; nothing here reads files or splits.
"""

import numpy as np

from vision_lab.wsp.legacy.experiments import (
    concept_directions,
    norm_match,
    train_head,
)


def class_weights(y, balanced):
    y = np.asarray(y, dtype=int)
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Two binary classes required')
    if not balanced:
        return np.ones(len(y))
    counts = np.bincount(y, minlength=2)
    return len(y) / (2 * counts[y])


def fit_pair(x, y, confounded, seed, training):
    cw = class_weights(y, training['class_balanced'])
    args = dict(steps=training['steps'], lr=training['lr'], l2=training['l2'])
    initial = np.random.default_rng(seed).normal(0, 0.001, x.shape[1] + 1)
    a = train_head(x, y, seed, sample_weight=cw * confounded, initial=initial, **args)
    b = train_head(x, y, seed, sample_weight=cw * 0.5, initial=initial, **args)
    args['steps'] = 2 * args['steps']
    r = train_head(x, y, seed + 100000, sample_weight=cw * 0.5, **args)
    return a, b, r


def directions(clean, marked, labels, categories, seed, protection):
    """Remove TRAIN clean-label/category directions from the paired marker mean."""
    labels = np.asarray(labels)
    categories = np.asarray(categories)
    result = concept_directions(clean, marked, clean[labels == 0], seed)
    if protection == 'none':
        return result
    rows = [clean[labels == 1].mean(0) - clean[labels == 0].mean(0)]
    if protection == 'label_and_category':
        center = clean.mean(0)
        rows.extend(
            clean[categories == cat].mean(0) - center for cat in sorted(set(categories))
        )
    elif protection != 'label':
        raise ValueError(protection)
    _, s, v = np.linalg.svd(np.asarray(rows, dtype=float), full_matrices=False)
    basis = v[s > max(s[0] * 1e-8, 1e-10)]
    original = result['marker']
    protected = original - basis.T @ (basis @ original)
    if np.linalg.norm(protected) < 1e-8:
        raise ValueError('Protected marker concept unidentifiable')
    marker = norm_match(protected, original)
    random = norm_match(np.random.default_rng(seed).normal(size=marker.shape), marker)
    irrelevant = result['irrelevant'] - marker * (result['irrelevant'] @ marker) / (
        marker @ marker
    )
    return dict(
        marker=marker,
        random=random,
        irrelevant=norm_match(irrelevant, marker),
        sham=marker.copy(),
    )


def paired_batches(clean_indices, marker_indices, seed, steps, batch_sources):
    """Same paired-source minibatch schedule for A and B, with replacement."""
    c = np.asarray(clean_indices)
    m = np.asarray(marker_indices)
    if c.shape != m.shape or not len(c):
        raise ValueError('Aligned pairs required')
    rng = np.random.default_rng(seed + 700000)
    for _ in range(steps):
        ids = rng.integers(len(c), size=batch_sources)
        yield np.stack([c[ids], m[ids]], 1).reshape(-1)


def train_program_batches(
    fetch, y, weights, width, seed, clean, marked, training, steps=None
):
    import torch

    from vision_lab.wsp.legacy.own_head import make_head

    model = make_head(width, seed)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=training['lr'], weight_decay=training['l2']
    )
    y = np.asarray(y)
    weights = np.asarray(weights)
    for ids in paired_batches(
        clean, marked, seed, steps or training['own_steps'], training['batch_sources']
    ):
        x = torch.as_tensor(fetch(ids), dtype=torch.float32)
        target = torch.as_tensor(y[ids], dtype=torch.float32)
        w = torch.as_tensor(weights[ids], dtype=torch.float32)
        optimizer.zero_grad()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                model(x), target, reduction='none'
            )
            * w
        ).sum() / w.sum()
        loss.backward()
        optimizer.step()
    return model
