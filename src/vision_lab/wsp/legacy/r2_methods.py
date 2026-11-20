"""Round 2 numerical recipes: source ordering, subspace edit, retrainers, controls.

Pure functions on arrays; no file or cache access.
"""

import hashlib
import time

import numpy as np

from .experiments import sigmoid


def _digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def ordered_sources(rows, seed):
    """Seeded order that alternates labels and cycles through categories."""
    h = _digest
    ordered = []
    for label in (0, 1):
        cats = sorted(
            {r['category'] for r in rows if r['label'] == label},
            key=lambda c: h(f'r2|{seed}|{c}'),
        )
        queues = {
            c: sorted(
                [r for r in rows if r['label'] == label and r['category'] == c],
                key=lambda r: h(f'r2|{seed}|{r["source_id"]}'),
            )
            for c in cats
        }
        seq = []
        while any(queues.values()):
            for c in cats:
                if queues[c]:
                    seq.append(queues[c].pop(0))
        ordered.append(seq)
    if min(map(len, ordered)) < 64:
        raise ValueError('Insufficient TRAIN capacity')
    return [r for pair in zip(*ordered) for r in pair]


def cosine(a, b):
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if d == 0 else float(a @ b / d)


def subspace(c, m, y):
    c = np.asarray(c, dtype=float)
    m = np.asarray(m, dtype=float)
    y = np.asarray(y)
    g = c[y == 1].mean(0) - c[y == 0].mean(0)
    gn = float(g @ g)
    means = np.stack([(m - c)[y == j].mean(0) for j in (0, 1)], 1)
    protected = means.copy()
    if gn >= 1e-20:
        protected -= np.outer(g, g @ means / gn)
    u, s, _ = np.linalg.svd(protected, full_matrices=False)
    q = u[:, s > max(1e-10, 1e-8 * s[0])]

    def detail(d, p):
        energy = float(d @ d)
        return dict(
            marker_defect_cosine=cosine(d, g),
            overlap_before=None
            if not energy or not gn
            else float((d @ g) ** 2 / gn / energy),
            overlap_after=None
            if not p @ p or not gn
            else float((p @ g) ** 2 / gn / (p @ p)),
            retained_marker_energy=None if not energy else float(p @ p / energy),
        )

    diag = dict(
        rank=q.shape[1],
        status='NONIDENTIFIABLE' if not q.shape[1] else 'IDENTIFIABLE',
        label_protection=gn >= 1e-20,
        singular_values=s.tolist(),
        good_defect_direction_cosine=cosine(means[:, 0], means[:, 1]),
        by_label={str(j): detail(means[:, j], protected[:, j]) for j in (0, 1)},
    )
    return q, c.mean(0), g, diag


def correction(c, m, a, q, mu):
    z = (c - mu) @ q
    d = (m - c) @ q
    mat = z.T @ z / len(c) + d.T @ d / len(c) + 0.1 * np.eye(q.shape[1])
    v = np.linalg.solve(mat, -d.T @ ((m - c) @ a[:-1]) / len(c))
    dw = q @ v
    return np.r_[dw, -mu @ dw]


def timed_edit(fetch, a, family):
    cpu = time.process_time()
    wall = time.perf_counter()
    c, m, y = fetch()
    q, mu, g, diag = subspace(c, m, y)
    delta = correction(c, m, a, q, mu) if family != 'V1' else np.zeros_like(a)
    # Building the primary edit is part of the timed (60 CPU-second) allowance.
    _primary = a + 1.5 * delta if family != 'V1' else (q.copy(), mu.copy(), a.copy())
    used = time.process_time() - cpu
    if used > 60:
        raise TimeoutError('Editor exceeded 60 process CPU seconds')
    return dict(q=q, mu=mu, g=g, delta=delta, diagnostics=diag), dict(
        cpu_seconds=used,
        wall_seconds=time.perf_counter() - wall,
        updates=0,
        allowance=60,
    )


def retrain(fetch, a, seed, fresh, spend):
    cpu = time.process_time()
    wall = time.perf_counter()
    c, m, y = fetch()
    x = np.concatenate([c, m]).astype(float)
    labels = np.tile(y, 2).astype(float)
    theta = (
        np.random.default_rng(seed + 900000).normal(0, 0.001, len(a))
        if fresh
        else a.copy()
    )
    checkpoint = theta.copy()
    checkpoint_updates = 0
    checkpoint_cost = 0.0
    used = time.process_time() - cpu
    updates = 0
    costs = []
    trajectory = [theta.copy()]
    if used <= spend:
        checkpoint_cost = used
    for i in range(2000):
        err = (sigmoid(x @ theta[:-1] + theta[-1]) - labels) / len(labels)
        nxt = theta.copy()
        nxt[:-1] -= 0.03 * (x.T @ err + 0.001 * theta[:-1])
        nxt[-1] -= 0.03 * err.sum()
        materialized = nxt.copy()
        elapsed = time.process_time() - cpu
        if elapsed > 60:
            break
        theta = nxt
        updates = i + 1
        used = elapsed
        costs.append(elapsed)
        trajectory.append(materialized)
        if elapsed <= spend:
            checkpoint = theta.copy()
            checkpoint_updates = updates
            checkpoint_cost = elapsed
    if not np.isfinite(theta).all():
        raise ArithmeticError('Nonfinite retraining')
    return (
        theta,
        checkpoint,
        np.asarray(trajectory),
        dict(
            cpu_seconds=time.process_time() - cpu,
            last_complete_cpu_seconds=used,
            wall_seconds=time.perf_counter() - wall,
            updates=updates,
            equal_spend_cpu_seconds=checkpoint_cost,
            equal_spend_updates=checkpoint_updates,
            edit_cpu_seconds=spend,
            update_cpu_seconds=costs,
            allowance=60,
            initialization_if_no_update=True,
        ),
    )


def controls(edit, a, seed, family):
    q, mu, g, delta = map(edit.get, ('q', 'mu', 'g', 'delta'))
    result = {}
    for j in (1, 2, 3):
        rng = np.random.Generator(np.random.PCG64(seed + 100000 * j))
        if family == 'V1':
            r = rng.normal(size=q.shape)
            if g @ g >= 1e-20:
                r -= np.outer(g, g @ r / (g @ g))
            result[f'random{j}'] = np.linalg.qr(r, mode='reduced')[0]
        else:
            r = rng.normal(size=delta.shape)
            result[f'random{j}'] = r * (np.linalg.norm(delta) / np.linalg.norm(r))
            result[f'shuffle{j}'] = rng.permutation(delta)
    result['sham'] = q if family == 'V1' else np.zeros_like(delta)
    if family != 'V1':
        result['reverse'] = -delta
    return result


def stability(fetch, a, family, categories):
    c, m, y = fetch()
    out = []
    for fold in (0, 1):
        train = np.concatenate([np.flatnonzero(y == j)[fold::2] for j in (0, 1)])
        test = np.setdiff1d(np.arange(len(y)), train)
        q, mu, g, diag = subspace(c[train], m[train], y[train])
        if family == 'V1':

            def score(x):
                return (x - (x - mu) @ q @ q.T) @ a[:-1] + a[-1]
        else:
            theta = a + 1.5 * correction(c[train], m[train], a, q, mu)

            def score(x):
                return x @ theta[:-1] + theta[-1]

        out.append(
            dict(
                fold=fold,
                n_train=len(train),
                n_test=len(test),
                rank=q.shape[1],
                held_marker_response_before=float(
                    np.mean((m[test] - c[test]) @ a[:-1])
                ),
                held_marker_response_after=float(
                    np.mean(score(m[test]) - score(c[test]))
                ),
                held_clean_score_change=float(
                    np.mean(score(c[test]) - (c[test] @ a[:-1] + a[-1]))
                ),
            )
        )
    category = {}
    for cat in sorted(set(categories)):
        ids = np.flatnonzero(np.asarray(categories) == cat)
        counts = {str(j): int(sum(y[ids] == j)) for j in (0, 1)}
        if min(counts.values()):
            _, _, _, diag = subspace(c[ids], m[ids], y[ids])
            category[cat] = dict(counts=counts, **diag)
    return dict(two_fold=out, categories_with_both_labels=category)
