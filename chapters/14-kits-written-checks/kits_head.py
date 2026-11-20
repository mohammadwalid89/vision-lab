"""Written part checks over SigLIP patch tokens, a small learned part, and baselines.

Every score is a function of one image's patch-token matrix X (P x D, P = GRID^2)
and of statistics fitted on normal images only. The exact algebra is in README.md.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

SEED = 20261112
GRID = 14
CHECK_ORDER = ("presence", "count", "position", "extent", "cooccurrence")

# Written before any development image was scored.
CHECK_WEIGHTS = {"presence": 1.0, "count": 1.0, "position": 0.75,
                 "extent": 1.0, "cooccurrence": 1.0}

OCC_MIN = 0.02          # a part must cover >= 2% of patches to count as expected
OCC_SLACK = 1.0         # one patch of occupancy jitter is free
OCC_MAX_LOCAL = 0.5     # a part covering half the frame is background, not countable
POS_GATE = 2.0          # minimum position/extent tolerance, in normal sigmas
COUNT_SLACK = 0.5       # minimum count tolerance: half a component
PAIR_RATE_MIN = 0.9
MAX_PAIRS = 32


def unit(x, axis=-1, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def grid_uv(P):
    g = int(round(np.sqrt(P)))
    if g * g != P:
        raise ValueError(f"patch count {P} is not a square grid")
    idx = np.arange(P)
    return (idx % g + .5) / g, (idx // g + .5) / g, g


def connected_count(mask, g, min_area=2):
    """Number of 4-connected active components with at least `min_area` patches."""
    if not mask.any():
        return 0
    lab, n = ndimage.label(mask.reshape(g, g),
                           structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel())[1:]
    return int(np.count_nonzero(sizes >= min_area))


@dataclass
class Standardizer:
    """Median / MAD location and scale, fitted on normal scores only."""
    median: float = 0.0
    scale: float = 1.0

    @classmethod
    def fit(cls, values):
        v = np.asarray(values, float)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        return cls(med, max(mad, 1e-3))

    def __call__(self, values):
        return (np.asarray(values, float) - self.median) / self.scale


@dataclass
class PartDictionary:
    """K unit-norm patch-token anchors fitted on normal images."""
    anchors: np.ndarray
    k: int = 0

    @classmethod
    def fit(cls, tokens, k=24, sample=48000, seed=SEED, n_init=3):
        X = unit(np.asarray(tokens, dtype=np.float32))
        if len(X) > sample:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(X), sample, replace=False))
            X = X[idx]
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
        a = unit(km.cluster_centers_.astype(np.float32))
        order = np.lexsort(np.round(a, 6).T[::-1])   # stable anchor ids
        return cls(a[order], k)


@dataclass
class CheckBank:
    """Written checks with normal-only reference statistics, envelopes and menus."""
    anchors: np.ndarray
    tau: np.ndarray
    mu: dict
    sd: dict
    env: dict
    menu: dict
    pairs: np.ndarray
    weights: dict = field(default_factory=lambda: dict(CHECK_WEIGHTS))
    grid: int = GRID

    @property
    def occ_floor(self):
        return 1.0 / float(self.grid ** 2)

    @classmethod
    def fit(cls, dictionary, tokens_factory, k_tau_quantile=0.995):
        """`tokens_factory()` returns a fresh iterator of (P, D) normal token
        arrays; it is called twice so only one image is held in memory."""
        A = dictionary.anchors
        K = len(A)
        g = None
        sims = [[] for _ in range(K)]
        for X in tokens_factory():
            S = unit(X) @ A.T
            if g is None:
                g = int(round(np.sqrt(S.shape[0])))
            for j in range(K):
                sims[j].append(S[:, j])
        tau = np.array([np.clip(np.quantile(np.concatenate(sims[j]), k_tau_quantile),
                                0.35, 0.85) for j in range(K)], dtype=np.float32)
        per = {n: [] for n in ("occ", "cnt", "u", "v", "h", "w")}
        for X in tokens_factory():
            st = _stats_from_sim(unit(X) @ A.T, tau, g)
            for n in per:
                per[n].append(st[n])
        P = {n: np.stack(per[n]) for n in per}
        mu = {n: _nan_stat(P[n], np.nanmean) for n in per}
        sd = {n: _nan_stat(P[n], np.nanstd) for n in per}
        occ_floor = 1.0 / float(int(g) ** 2)

        # Tolerance = 99.5th percentile of the check's own deviation on normals,
        # so a normal image has almost zero violation by construction.
        occ_dev = np.where(np.isfinite(P["occ"]), mu["occ"] - P["occ"], np.nan)
        cnt_dev = np.where(np.isfinite(P["cnt"]), np.abs(P["cnt"] - mu["cnt"]), np.nan)
        zu_n = (P["u"] - mu["u"]) / np.maximum(sd["u"], 0.01)
        zv_n = (P["v"] - mu["v"]) / np.maximum(sd["v"], 0.01)
        pos_dev = np.where(np.isfinite(P["u"]), np.hypot(zu_n, zv_n), np.nan)
        zh_n = np.abs(P["h"] - mu["h"]) / np.maximum(sd["h"], 0.01)
        zw_n = np.abs(P["w"] - mu["w"]) / np.maximum(sd["w"], 0.01)
        ext_dev = np.where(np.isfinite(P["h"]), np.maximum(zh_n, zw_n), np.nan)
        env = {
            "occ": np.maximum(_q995(occ_dev), OCC_SLACK * occ_floor),
            "cnt": np.maximum(_q995(cnt_dev), COUNT_SLACK),
            "pos": np.maximum(_q995(pos_dev), POS_GATE),
            "ext": np.maximum(_q995(ext_dev), POS_GATE),
        }

        localized = (mu["occ"] >= OCC_MIN) & (mu["occ"] < OCC_MAX_LOCAL)
        menu = {
            "presence": np.where(mu["occ"] >= OCC_MIN)[0],
            "count": np.where(localized)[0],
            "position": np.where(localized)[0],
            "extent": np.where(localized)[0],
        }
        present = P["occ"] >= np.maximum(0.5 * mu["occ"], occ_floor)[None, :]
        margin = present.mean(0)
        cand = []
        for i in range(K):
            for j in range(i + 1, K):
                if margin[i] >= PAIR_RATE_MIN and margin[j] >= PAIR_RATE_MIN:
                    rate = float(np.mean(present[:, i] & present[:, j]))
                    if rate >= PAIR_RATE_MIN:
                        cand.append((rate, i, j))
        cand.sort(key=lambda t: (-t[0], t[1], t[2]))
        pairs = np.array([[i, j] for _, i, j in cand[:MAX_PAIRS]],
                         dtype=int).reshape(-1, 2)
        return cls(A, tau,
                   {n: mu[n].astype(np.float32) for n in mu},
                   {n: sd[n].astype(np.float32) for n in sd},
                   {k: v.astype(np.float32) for k, v in env.items()},
                   {c: np.asarray(v, dtype=int) for c, v in menu.items()},
                   pairs, grid=int(g))

    def with_weights(self, weights):
        """Control: same bank, different check weights."""
        return CheckBank(self.anchors, self.tau, self.mu, self.sd, self.env,
                         self.menu, self.pairs, dict(weights), self.grid)

    def permuted(self, seed=SEED + 1):
        """Control: permute which normal reference each anchor is compared with.
        Keeps the score's shape, breaks the token-to-check mapping."""
        K = len(self.anchors)
        perm = np.random.default_rng(seed).permutation(K)
        inv = np.empty(K, int)
        inv[perm] = np.arange(K)
        return CheckBank(self.anchors, self.tau,
                         {k: v[inv] for k, v in self.mu.items()},
                         {k: v[inv] for k, v in self.sd.items()},
                         {k: v[inv] for k, v in self.env.items()},
                         {c: perm[v] for c, v in self.menu.items()},
                         perm[self.pairs] if self.pairs.size else self.pairs.copy(),
                         dict(self.weights), self.grid)

    def image_stats(self, X):
        return _stats_from_sim(unit(X) @ self.anchors.T, self.tau, self.grid)

    def _z(self, st):
        mu, sd = self.mu, self.sd
        return ((st["u"] - mu["u"]) / np.maximum(sd["u"], 0.01),
                (st["v"] - mu["v"]) / np.maximum(sd["v"], 0.01),
                (st["h"] - mu["h"]) / np.maximum(sd["h"], 0.01),
                (st["w"] - mu["w"]) / np.maximum(sd["w"], 0.01))

    def violations(self, st):
        mu, sd, env = self.mu, self.sd, self.env
        v = {}

        idx = self.menu["presence"]
        scale = np.maximum(sd["occ"][idx],
                           np.maximum(0.25 * mu["occ"][idx], self.occ_floor))
        v["presence"] = _max_over(idx, np.maximum(
            0.0, (mu["occ"][idx] - st["occ"][idx] - env["occ"][idx]) / scale))

        idx = self.menu["count"]
        v["count"] = _max_over(idx, np.maximum(
            0.0, (np.abs(st["cnt"][idx] - mu["cnt"][idx]) - env["cnt"][idx]) /
            np.maximum(sd["cnt"][idx], 0.25)))

        zu, zv, zh, zw = self._z(st)
        idx = self.menu["position"]
        v["position"] = _max_over(idx, np.maximum(
            0.0, np.hypot(zu[idx], zv[idx]) - env["pos"][idx]))

        idx = self.menu["extent"]
        v["extent"] = _max_over(idx, np.maximum(
            0.0, np.maximum(np.abs(zh[idx]), np.abs(zw[idx])) - env["ext"][idx]))

        p = self.pairs
        if len(p):
            present = st["occ"] >= np.maximum(0.5 * mu["occ"], self.occ_floor)
            v["cooccurrence"] = float((present[p[:, 0]] ^ present[p[:, 1]]).mean())
        else:
            v["cooccurrence"] = 0.0
        return v

    def written_score(self, X):
        v = self.violations(self.image_stats(X))
        return float(sum(self.weights[c] * v[c] for c in CHECK_ORDER))

    def compact(self, X, pooled=None, projector=None):
        """Check values, per-anchor residuals and optional pooled PCA coordinates."""
        st = self.image_stats(X)
        mu, sd = self.mu, self.sd
        v = self.violations(st)
        zu, zv, zh, zw = self._z(st)
        zc = _clipz((st["cnt"] - mu["cnt"]) / np.maximum(sd["cnt"], 0.25))
        zo = _clipz((st["occ"] - mu["occ"]) / np.maximum(
            sd["occ"], np.maximum(0.25 * mu["occ"], self.occ_floor)))
        parts = [np.array([v[c] for c in CHECK_ORDER], np.float32),
                 zo, zc, _clipz(zu), _clipz(zv), _clipz(zh), _clipz(zw)]
        if projector is not None and pooled is not None:
            parts.append(projector(pooled))
        return np.concatenate(parts).astype(np.float32)


def _max_over(idx, values):
    # Absent anchors are the presence check's job; non-finite terms are ignored here.
    if np.asarray(idx).size == 0 or np.size(values) == 0:
        return 0.0
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else 0.0


def _clipz(z):
    return np.clip(np.nan_to_num(np.asarray(z, np.float32), nan=0.0,
                                 posinf=8.0, neginf=-8.0), -8.0, 8.0)


def _q995(a, q=0.995):
    M = np.where(np.isfinite(a), a, np.nan)
    with np.errstate(invalid="ignore"):
        out = np.nanquantile(M, q, axis=0)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _nan_stat(P, fn):
    # Anchors never active in any normal image get 0.
    M = np.where(np.isfinite(P), P, np.nan)
    with np.errstate(invalid="ignore"):
        out = fn(M, axis=0)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _stats_from_sim(S, tau, g):
    P, K = S.shape
    A = S >= tau[None, :]
    u, v, _ = grid_uv(P)
    occ = A.mean(0).astype(np.float32)
    cnt = np.zeros(K, np.float32)
    cu = np.full(K, np.nan, np.float32)
    cv = np.full(K, np.nan, np.float32)
    hh = np.full(K, np.nan, np.float32)
    ww = np.full(K, np.nan, np.float32)
    for k in range(K):
        m = A[:, k]
        cnt[k] = connected_count(m, g)
        if m.any():
            uu, vv = u[m], v[m]
            cu[k], cv[k] = uu.mean(), vv.mean()
            hh[k] = vv.max() - vv.min()
            ww[k] = uu.max() - uu.min()
    return {"occ": occ, "cnt": cnt, "u": cu, "v": cv, "h": hh, "w": ww}


class PooledProjector:
    """PCA of pooled embeddings fitted on normals; returns `dim` clipped coordinates."""

    def __init__(self, dim=16):
        self.dim = dim
        self.pca = None

    def fit(self, pooled):
        d = min(self.dim, len(pooled) - 1, pooled.shape[1])
        self.pca = PCA(n_components=d, random_state=SEED).fit(np.asarray(pooled, np.float32))
        return self

    def __call__(self, pooled_row):
        z = self.pca.transform(np.asarray(pooled_row, np.float32).reshape(1, -1))[0]
        return _clipz(z / 3.0)


class LearnedPart:
    """L2 logistic head on the compact feature vector."""

    def __init__(self, C=0.5, seed=SEED):
        self.clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced",
                                      random_state=seed, solver="lbfgs")

    def fit(self, F, y):
        self.clf.fit(np.asarray(F, np.float64), np.asarray(y, int))
        return self

    def decision(self, f):
        return float(self.clf.decision_function(np.asarray(f, np.float64).reshape(1, -1))[0])


class ProgrammedHead:
    """alpha * written + (1 - alpha) * learned, both standardised on calibration normals."""

    def __init__(self, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha outside [0, 1]")
        self.alpha = alpha
        self.written_std = None
        self.learned_std = None

    def freeze(self, bank, learned, calib_pairs, projector=None):
        """calib_pairs: iterable of (patch tokens, pooled) for calibration normals."""
        wz, lz = [], []
        for X, pooled in calib_pairs:
            wz.append(bank.written_score(X))
            lz.append(learned.decision(bank.compact(X, pooled, projector)))
        self.written_std = Standardizer.fit(wz)
        self.learned_std = Standardizer.fit(lz)
        return self


class LinearHead:
    """Baseline: logistic regression on pooled embedding + last-layer mean patch."""

    def __init__(self, C=1.0, seed=SEED):
        self.clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced",
                                      random_state=seed, solver="lbfgs")

    def fit(self, F, y):
        self.clf.fit(np.asarray(F, np.float64), np.asarray(y, int))
        return self

    def score(self, f):
        return float(self.clf.decision_function(np.asarray(f, np.float64).reshape(1, -1))[0])


class MLPHead:
    """Baseline: matched-compute retrained MLP on the same compact features."""

    def __init__(self, hidden=64, seeds=(20261112, 20261113, 20261114), epochs=400, lr=1e-2):
        self.hidden, self.seeds, self.epochs, self.lr = hidden, seeds, epochs, lr
        self.nets = []

    def fit(self, F, y):
        import torch
        torch.set_num_threads(2)
        X = torch.from_numpy(np.asarray(F, np.float32))
        Y = torch.from_numpy(np.asarray(y, np.float32)).reshape(-1, 1)
        pos = float(max(Y.sum().item(), 1.0))
        neg = float(max(len(Y) - pos, 1.0))
        self.nets = []
        for s in self.seeds:
            torch.manual_seed(int(s))
            net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], self.hidden),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(self.hidden, 1))
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos))
            net.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = lossf(net(X), Y)
                loss.backward()
                opt.step()
            net.eval()
            self.nets.append(net)
        return self

    def score(self, f):
        import torch
        x = torch.from_numpy(np.asarray(f, np.float32).reshape(1, -1))
        with torch.inference_mode():
            return float(np.mean([n(x).item() for n in self.nets]))


class PatchKNN:
    """Baseline: PatchCore-style memory bank of SigLIP tokens (greedy cosine coreset)."""

    def __init__(self, k=512, seed=SEED):
        self.k, self.seed = k, seed
        self.bank = None

    def fit(self, patch_iter, max_candidates=20000):
        rng = np.random.default_rng(self.seed)
        pool = np.concatenate([unit(X) for X in patch_iter], axis=0)
        if len(pool) > max_candidates:
            pool = pool[np.sort(rng.choice(len(pool), max_candidates, replace=False))]
        k = min(self.k, len(pool))
        sel = [int(rng.integers(len(pool)))]
        sim = pool @ pool[sel[0]]
        for _ in range(k - 1):
            j = int(np.argmin(sim))
            sel.append(j)
            sim = np.maximum(sim, pool @ pool[j])
        self.bank = pool[np.asarray(sel)]
        return self

    def score(self, X):
        s = unit(X) @ self.bank.T
        return float(1.0 - np.max(s))
