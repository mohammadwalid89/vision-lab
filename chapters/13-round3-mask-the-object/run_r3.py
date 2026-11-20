"""Round 3 runner: plan, preflight, extract, attempt, report, and a synthetic dry run.

  plan      write R3_PLAN.json (all declared constants; its sha256 is what gets signed)
  preflight hashes, mask/view coverage, partition == published, extractor alignment and
            determinism; writes PREFLIGHT.json (PASS or BLOCKED; everything else refuses
            to start unless PASS)
  extract   resumable sharded feature extraction under the resource guard
  attempt   one slot x one k (all seeds, all controls) through vision_lab.wsp.harness
  report    R3_RESULTS[_arm].md from the attempt files
  dry-run   builds a synthetic R1-shaped tree and runs plan -> report end-to-end with a tiny
            encoder (used by tests/test_runner.py; seconds, no models)

Arms: every command takes --arm (default siglip, the primary arm).  The plan declares a second
arm, `dinov2`, added on 2026-11-17 with the same slots, recipes, gates, k schedule and seeds; it has
its own features, ledger, attempts and results (suffix _dinov2) and is reported separately.

Every subcommand applies the process limits (<= 3 cores, nice 10) and the resource guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from vision_lab.wsp import harness as H
from vision_lab.wsp.calibration import (
    AccessLedger,
    partition_sources,
    verify_against_published,
)
from vision_lab.wsp.ledger import AttemptLedger, ResourceGuard
from vision_lab.wsp.ordering import k_lists
from vision_lab.wsp.pipelines import SlotData, slot_pipelines
from vision_lab.wsp.r1tree import PluginViews, PngDirViews, R1Tree, sha256_file
from vision_lab.wsp.runtime import limit
from vision_lab.wsp.slots import input_mask as IM
from vision_lab.wsp.slots.hidden_deletion import DeletionRecipe

ARM_PRIMARY = "siglip"

DEFAULT_PLAN = dict(
    seeds=[20261112, 20261113, 20261114], ks=[8, 32, 128], primary_k=32,
    cal_n={"0": 384, "1": 128}, k_largest=128, min_per_label=64,
    slots=["slot1", "slot2", "slot3"],
    recipes=dict(
        slot1=dict(transform="sam_meanfill", fill="encoder_mean = round(255 x the arm encoder normalisation mean); see arms[*].fill_rgb",
                   mask="R1 SAM object mask, as stored",
                   head="A recipe refit on transformed fitting pool", placebos="random_area x3 (cyclic shift)",
                   references=["padding_only", "testtime_mask_A_head"]),
        slot2=dict(feature="object_only patch-grid (SigLIP 14x14, DINOv2 16x16) last-layer post-LN aggregate",
                   weights="SAM coverage, normalised",
                   head="A recipe", placebos="within-source permutation x3", references=["uniform", "pooled_concat"]),
        slot3=dict(deletion_recipe=None)),
    transforms=["identity", "sam_meanfill", "padding_only", "random_area_1", "random_area_2", "random_area_3"],
    gate_families=["CORE", "G5"], threshold_rules=["per_pipeline", "shared_A"], binding_rule="per_pipeline",
    feasibility_tol=0.02, alpha=0.01,
    resources=dict(max_cores=3, nice=10, free_floor_gib=4, footprint_cap_gb=5, rss_ceiling_gib=8),
    slot1_placebo="random_area_",          # the signed plan overrides this with "random_area_ps_" (padding-sparing)
    content_box=None, shard_size=512, alignment_views=12, determinism_views=512,
    alignment_tol_abs=0.05,
    # Encoder arms.  "siglip" is the primary arm (R1/R2's encoder).  "dinov2" is an ADDED ARM declared on
    # 2026-11-17 (owner approval 15:07 PT): every slot, recipe, gate, threshold
    # rule, seed, k schedule, partition and control is identical; only the frozen encoder (and therefore the
    # encoder-normalisation fill colour) differs.  Arms are run, ledgered and reported separately; no result
    # is pooled or selected across arms.  fill_rgb = round(255 * encoder normalisation mean) and is checked
    # against the instantiated encoder by preflight.
    arms=dict(
        siglip=dict(encoder="siglip", role="primary", suffix="", fill_rgb=[128, 128, 128], r1_cache_alignment=True,
                    weights_sha256="612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
                    input="SigLIP processor: 224 px, mean/std 0.5; 14x14 tokens; pooled = MAP head"),
        dinov2=dict(encoder="dinov2", role="ADDED ARM (owner approval 2026-11-17 15:07 PT)", suffix="_dinov2",
                    fill_rgb=[124, 116, 104], r1_cache_alignment=False,
                    weights_sha256="c24ecfb4a1d8ca79193f6b9efcffc461872a09ddac43a0931357e4802931a006",
                    input="timm vit_base_patch14_reg4_dinov2.lvd142m, whole 384 canvas -> 224 px bicubic, ImageNet "
                          "mean/std, 16x16 patch tokens, 4 registers dropped, pooled = CLS, mean_patch = token mean",
                    same_as_primary="slots, recipes, gates, k schedule, seeds, thresholds, calibration partition, controls",
                    reporting="separate attempt ledger and results; no pooling or selection across arms")))


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_plan(out):
    p = json.loads((out / "R3_PLAN.json").read_text())
    p["_sha256"] = _sha({k: v for k, v in p.items() if not k.startswith("_")})
    return p


def guard(out, plan):
    r = plan["resources"]
    return ResourceGuard(out, free_floor_bytes=int(r["free_floor_gib"] * 2**30),
                         footprint_cap_bytes=int(r["footprint_cap_gb"] * 10**9),
                         rss_ceiling_bytes=int(r["rss_ceiling_gib"] * 2**30), max_cores=r["max_cores"])


def make_encoder(name):
    if name == "siglip":
        from vision_lab.wsp.encoders import SiglipEncoder
        return SiglipEncoder()
    if name == "dinov2":
        from vision_lab.wsp.encoders import DinoV2Encoder
        return DinoV2Encoder()
    if name == "tiny":
        return TinyBatch()
    if name == "tiny_imagenet":          # dry run only: tiny encoder with DINOv2's normalisation mean
        return TinyBatch(mean=(0.485, 0.456, 0.406))
    raise ValueError(name)


def weights_file(name):
    from vision_lab.wsp.encoders import DINO_SNAP, SIGLIP_PATH
    siglip = None if SIGLIP_PATH is None else SIGLIP_PATH / "model.safetensors"
    return {"siglip": siglip, "dinov2": DINO_SNAP}.get(name)


def arm_cfg(plan, arm):
    if arm not in plan["arms"]:
        raise SystemExit(f"undeclared arm {arm!r}; plan declares {sorted(plan['arms'])}")
    return plan["arms"][arm]


def apaths(out, plan, arm):
    """Per-arm file layout.  The primary arm keeps the unsuffixed names."""
    sfx = arm_cfg(plan, arm)["suffix"]
    out = Path(out)
    return dict(features=out / f"features{sfx}", preflight=out / f"PREFLIGHT{sfx}.json",
                manifest=out / f"EXTRACT_MANIFEST{sfx}.json", ledger=out / f"ATTEMPT_LEDGER{sfx}.json",
                attempts=out / f"attempts{sfx}", results=out / f"R3_RESULTS{sfx}.md")


class TinyBatch:
    """Batch wrapper around synthetic.TinyPatchEncoder for the dry run."""
    def __init__(self, mean=(0.5, 0.5, 0.5)):
        from vision_lab.wsp.synthetic import TinyPatchEncoder
        self.e = TinyPatchEncoder(attention=True, mean=mean)
        self.grid, self.img, self.mean = self.e.grid, self.e.size, tuple(mean)

    def __call__(self, images):
        outs = [self.e(i) for i in images]
        return {k: np.stack([o[k] for o in outs]) for k in outs[0]}


def views_from(args):
    if args.views_plugin:
        return PluginViews(args.views_plugin)
    if args.views_dir:
        return PngDirViews(args.views_dir)
    raise SystemExit("supply --views-dir or --views-plugin")


def transforms_for(plan, enc):
    """Slot-1 transform family with the arm's encoder-mean fill (SigLIP 128,128,128; DINOv2 124,116,104)."""
    tf = IM.build_transforms(n_random=3, encoder_mean=tuple(enc.mean), padding_sparing=True)
    tf["keep_all"] = IM.InputTransform("keep_all", "noop", lambda im, meta: IM.mask_fill(im, np.ones(im.shape[:2], bool)))
    return tf


def cmd_plan(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(json.dumps(DEFAULT_PLAN))
    plan["recipes"]["slot3"]["deletion_recipe"] = dict(DeletionRecipe().__dict__, sha256=DeletionRecipe().digest())
    if args.content_box:
        plan["content_box"] = [int(v) for v in args.content_box.split(",")]
    if args.plan_overrides:
        plan.update(json.loads(Path(args.plan_overrides).read_text()))
    p = out / "R3_PLAN.json"
    if p.exists() and json.loads(p.read_text()) != plan:
        raise SystemExit("R3_PLAN.json exists and differs; a signed plan is immutable")
    p.write_text(json.dumps(plan, indent=1))
    print("plan sha256", _sha(plan))


def cmd_preflight(args):
    out = Path(args.out)
    plan = load_plan(out)
    arm = args.arm
    cfg, P = arm_cfg(plan, arm), apaths(out, plan, arm)
    g = guard(out, plan)
    tree, views = R1Tree(args.wsp_root), views_from(args)
    rep = dict(plan_sha256=plan["_sha256"], arm=arm, checks={}, started=time.time())
    ok = True

    def check(name, cond, detail=None):
        nonlocal ok
        rep["checks"][name] = dict(ok=bool(cond), detail=detail)
        ok &= bool(cond)

    rep["resources_start"] = g.check("preflight")
    rows, idx = tree.view_rows()
    check("active_rows", len(rows) > 0, dict(n=len(rows)))
    for name in ("pooled", "mean_patch"):
        p, h = tree.cached_array(idx, name)
        check(f"array_{name}_sha256", p.exists() and sha256_file(p) == h, str(p))
    groups = sorted({r["group_id"] for r in rows})
    man = tree.mask_manifest()
    missing = [gid for gid in groups if not (tree.r1 / "masks" / f"{gid}_object.png").exists()]
    bad_hash = [gid for gid in groups if man and f"{gid}_object.png" in man and
                gid not in missing and sha256_file(tree.r1 / "masks" / f"{gid}_object.png") != man[f"{gid}_object.png"]]
    check("masks_present", not missing, dict(missing=len(missing)))
    check("masks_hash", not bad_hash, dict(mismatch=len(bad_hash), manifest_entries=len(man)))
    nv = [r["id"] for r in rows if not views.exists(r)]
    check("views_resolvable", not nv, dict(missing=len(nv)))
    check("content_box_declared", plan["content_box"] is not None, plan["content_box"])
    if plan["content_box"] is not None:
        # label-free placebo-bar risk (non-blocking): how much padding do random-area masks keep?
        sample = groups[:200]
        kept = [IM.padding_kept_fraction(tree.mask(gid), tuple(plan["content_box"]),
                                         f"r3|slot1|random_area|{gid}|1") for gid in sample]
        rep["slot1_placebo_risk"] = dict(
            mean_padding_kept_by_random_area=float(np.mean(kept)), n_groups=len(sample),
            gate_1b_likely_impossible_with_random_area=bool(np.mean(kept) < 0.5),
            note="marker lives in the padding; if random-area placebos remove it more often than not, "
                 "2 x best placebo reduction exceeds A's flip rate. A plan override may set slot1_placebo='random_area_ps_'.")
    train = tree.intake_train()
    seeds = plan["seeds"]
    cal_n = {int(k): v for k, v in plan["cal_n"].items()}
    part = partition_sources(train, seeds, plan["k_largest"], cal_n, min_per_label=plan["min_per_label"])
    kl = k_lists(train, seeds, plan["ks"], plan["min_per_label"])
    if args.no_published_partition:
        check("partition_equals_published", True, "skipped by --no-published-partition (dry run only)")
    elif args.published_partition and Path(args.published_partition).exists():
        v = verify_against_published(part, json.loads(Path(args.published_partition).read_text()))
        check("partition_equals_published", v["equal"], v)
    else:
        check("partition_equals_published", False, "published partition missing")
    (out / "PARTITION.json").write_text(json.dumps(part, indent=1))
    (out / "K_LISTS.json").write_text(json.dumps({str(s): {str(k): v for k, v in d.items()} for s, d in kl.items()}))
    # extractor alignment vs the cached R1 arrays (identity transform, last layer)
    enc = make_encoder(cfg["encoder"])
    wf = weights_file(cfg["encoder"])
    if wf is not None:
        check("weights_sha256", sha256_file(wf.resolve()) == cfg["weights_sha256"], str(wf))
    fill = [int(v) for v in np.round(np.asarray(enc.mean) * 255.0)]
    check("fill_rgb_equals_encoder_mean", fill == list(cfg["fill_rgb"]), dict(encoder=fill, plan=cfg["fill_rgb"]))
    pick = sorted(range(len(rows)), key=lambda i: hashlib.sha256(("align|" + rows[i]["id"]).encode()).hexdigest())
    a_idx = pick[:plan["alignment_views"]]
    imgs = [views(rows[i]) for i in a_idx]
    f = enc(imgs)
    if cfg["r1_cache_alignment"]:
        pooled = np.load(tree.cached_array(idx, "pooled")[0], mmap_mode="r")
        meanp = np.load(tree.cached_array(idx, "mean_patch")[0], mmap_mode="r")
        pos = {r["id"]: i for i, r in enumerate(idx["rows"])}
        cp = np.stack([np.asarray(pooled[pos[rows[i]["id"]], -1], dtype=np.float64) for i in a_idx])
        cm = np.stack([np.asarray(meanp[pos[rows[i]["id"]], -1], dtype=np.float64) for i in a_idx])
        dp, dm = float(np.abs(f["pooled"] - cp).max()), float(np.abs(f["mean_patch"] - cm).max())
        check("alignment_pooled", dp <= plan["alignment_tol_abs"], dp)
        check("alignment_mean_patch", dm <= plan["alignment_tol_abs"], dm)
    else:    # R1 cached SigLIP2 only; the views themselves are proven by the primary arm's alignment check
        check("encoder_features_finite", bool(np.isfinite(f["pooled"]).all() and np.isfinite(f["mean_patch"]).all()
                                               and np.isfinite(f["tokens"]).all()),
              "R1 cache alignment not applicable to this encoder")
    # determinism: keep_all transform must reproduce identity features bit-for-bit
    tf = transforms_for(plan, enc)
    d_idx = pick[:plan["determinism_views"]]
    di = [views(rows[i]) for i in d_idx]
    f1 = enc(di)
    f2 = enc([tf["keep_all"](im, {}) for im in di])
    check("determinism_keep_all", all(np.array_equal(f1[k], f2[k]) for k in ("pooled", "mean_patch")),
          dict(n=len(d_idx)))
    rep["resources_end"] = g.check("preflight_end")
    rep["status"] = "PASS" if ok else "BLOCKED"
    P["preflight"].write_text(json.dumps(rep, indent=1, default=str))
    print("preflight", arm, rep["status"])
    return 0 if ok else 2


def require_preflight(out, plan, arm):
    p = apaths(out, plan, arm)["preflight"]
    if not p.exists():
        raise SystemExit(f"run preflight first (arm {arm})")
    pf = json.loads(p.read_text())
    if pf["status"] != "PASS" or pf["plan_sha256"] != plan["_sha256"] or pf.get("arm") != arm:
        raise SystemExit(f"preflight not PASS for this plan and arm {arm}")


def cmd_extract(args):
    from vision_lab.wsp.encoders import extract
    out = Path(args.out)
    plan = load_plan(out)
    arm = args.arm
    cfg, P = arm_cfg(plan, arm), apaths(out, plan, arm)
    require_preflight(out, plan, arm)
    g = guard(out, plan)
    tree, views = R1Tree(args.wsp_root), views_from(args)
    rows, _ = tree.view_rows()
    enc = make_encoder(cfg["encoder"])
    tf_all = transforms_for(plan, enc)
    want = args.transforms.split(",") if args.transforms else plan["transforms"]
    cb = tuple(plan["content_box"])
    manp = P["manifest"]
    man = json.loads(manp.read_text()) if manp.exists() else {}
    S = plan["shard_size"]
    masks = {}
    for t in want:
        (P["features"] / t).mkdir(parents=True, exist_ok=True)
        for s0 in range(0, len(rows), S):
            key = f"{t}/{s0 // S:05d}"
            fp = P["features"] / f"{key}.npz"
            if key in man and fp.exists() and sha256_file(fp) == man[key]["sha256"]:
                continue                                     # identical work already done
            chunk = rows[s0:s0 + S]
            items = []
            for r in chunk:
                if r["group_id"] not in masks:
                    masks[r["group_id"]] = tree.mask(r["group_id"])
                items.append((views(r), dict(source_id=r["source_id"], object_mask=masks[r["group_id"]], content_box=cb)))
            g.check(key, pending_bytes=len(chunk) * 8 * 768 * 2)
            t0 = time.process_time()
            f = extract(enc, items, {t: tf_all[t]})[t]
            keep = ("pooled", "mean_patch", "object", "perm_1", "perm_2", "perm_3", "uniform") if t == "identity" \
                else ("pooled", "mean_patch")
            np.savez(fp, ids=np.array([r["id"] for r in chunk]), **{k: f[k].astype(np.float16) for k in keep})
            man[key] = dict(sha256=sha256_file(fp), n=len(chunk), cpu_s=round(time.process_time() - t0, 1),
                            plan_sha256=plan["_sha256"])
            manp.write_text(json.dumps(man, indent=1))
            if len(masks) > 4096:
                masks.clear()
    print("extract done:", arm, want)


def load_features(feat_dir, rows, plan):
    feats = {}
    ids = [r["id"] for r in rows]
    for t in plan["transforms"]:
        parts = sorted((feat_dir / t).glob("*.npz"))
        if not parts:
            raise SystemExit(f"features for {t} missing")
        acc = {}
        got = []
        for p in parts:
            z = np.load(p)
            got.extend(z["ids"].tolist())
            for k in z.files:
                if k != "ids":
                    acc.setdefault(k, []).append(z[k].astype(np.float32))   # ~1.3 GB for all R3 views
        if got != ids:
            raise SystemExit(f"features for {t} incomplete or out of order")
        feats[t] = {k: np.concatenate(v) for k, v in acc.items()}
    # keep_all no-op: realised as an alias after the preflight bit-for-bit determinism check
    feats["keep_all"] = feats["identity"]
    return feats


def cmd_attempt(args):
    out = Path(args.out)
    plan = load_plan(out)
    arm = args.arm
    P = apaths(out, plan, arm)
    require_preflight(out, plan, arm)
    g = guard(out, plan)
    slot, k = args.slot, int(args.k)
    if slot not in plan["slots"] or k not in plan["ks"]:
        raise SystemExit("undeclared slot or k")
    tree = R1Tree(args.wsp_root)
    rows, _ = tree.view_rows()
    part = json.loads((out / "PARTITION.json").read_text())
    kl = {int(s): {int(kk): v for kk, v in d.items()} for s, d in json.loads((out / "K_LISTS.json").read_text()).items()}
    feats = load_features(P["features"], rows, plan)
    D = SlotData(rows, frozenset(part["calibration_source_ids"]), kl, k)
    acc = AccessLedger(frozenset(part["calibration_source_ids"]), kl)
    ledger = AttemptLedger.load(P["ledger"])
    ledger.declared_k = tuple(plan["ks"])
    for s in plan["slots"]:
        ledger.register_recipe(s, plan["recipes"][s])
    rec3 = DeletionRecipe(**{kk: v for kk, v in plan["recipes"]["slot3"]["deletion_recipe"].items() if kk != "sha256"})
    ledger.begin(slot, k, dict(plan=plan["_sha256"], arm=arm, partition=part["partition_digest"],
                               extract=sha256_file(P["manifest"])))
    spec = H.AttemptSpec(slot=slot, k=k, seeds=tuple(plan["seeds"]), gate_families=tuple(plan["gate_families"]),
                         threshold_rules=tuple(plan["threshold_rules"]), binding_rule=plan["binding_rule"],
                         feasibility_tol=plan["feasibility_tol"], alpha=plan["alpha"], recipe_sha256=ledger.recipes[slot])
    per, compute = {}, {}
    try:
        for seed in plan["seeds"]:
            g.check(f"{slot}/{k}/{seed}")
            pipes, comp = slot_pipelines(slot, seed, feats, D, acc, rec3, slot1_placebo=plan["slot1_placebo"])
            compute[str(seed)] = comp
            per[seed] = H.evaluate_seed(spec, seed, D.cal_rows, D.dev_rows, pipes, acc)
        rep = H.evaluate_attempt(spec, per)
    except Exception as e:
        ledger.finish(slot, k, "INFRA_BLOCKED")
        raise SystemExit(f"attempt failed (INFRA_BLOCKED, not a scientific result): {e!r}")
    ledger.finish(slot, k, rep["status"])
    P["attempts"].mkdir(exist_ok=True)
    (P["attempts"] / f"{slot}_k{k}.json").write_text(json.dumps(
        dict(arm=arm, report=rep, per_seed={str(s): v for s, v in per.items()}, compute=compute,
             access=acc.report(), plan_sha256=plan["_sha256"]), indent=1, default=float))
    print(arm, slot, k, rep["status"])


def cmd_report(args):
    out = Path(args.out)
    plan = load_plan(out)
    arm = args.arm
    P = apaths(out, plan, arm)
    lines = [f"# R3 results, arm `{arm}` ({arm_cfg(plan, arm)['role']}; DEV only, exploratory; "
             "generated by run_r3.py report)", "",
             "Arms are reported separately; nothing here is pooled or selected across arms.", "",
             f"plan sha256 `{plan['_sha256']}`; binding threshold rule `{plan['binding_rule']}`; "
             f"k={plan['primary_k']} decides each slot's primary conclusion.", "",
             "| slot | k | status | candidate | gates (worst seed) | feasibility worst recall margin | no-op integrity |",
             "|---|---|---|---|---|---|---|"]
    for p in sorted(P["attempts"].glob("*.json")):
        d = json.loads(p.read_text())["report"]
        rr = d["rules"][d["binding_rule"]]
        for c, v in rr["candidates"].items():
            gates = ", ".join(f"{n}:{'P' if b == 'PASS' else 'F'}" for n, b in v["gates"]["bars"].items())
            lines.append(f"| {d['slot']} | {d['k']} | {d['status']} | {c} | {gates} | "
                         f"{v['feasibility_worst_recall_margin']:+.4f} | {rr['noop_integrity']} |")
    lines += ["", "Status meanings: PASS = every declared gate and the feasibility test on every seed; "
              "NULL_OR_NEGATIVE = completed, conjunction failed; INCONCLUSIVE = a no-op moved scores; "
              "INFRA_BLOCKED = resource/integrity stop. None of these authorises holdout access."]
    P["results"].write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def build_fake_tree(root: Path, n_train=320, n_dev=80):
    """Synthetic R1-shaped tree (tiny 32 px canvases) for an end-to-end runner test."""
    from PIL import Image

    from vision_lab.wsp.synthetic import image_world
    enc = TinyBatch()
    r1 = root / "r1"
    (r1 / "masks").mkdir(parents=True, exist_ok=True)
    (r1 / "cache/siglip2_base").mkdir(parents=True, exist_ok=True)
    vdir = root / "views"
    vdir.mkdir(exist_ok=True)
    world = image_world(n=n_train + n_dev, seed=5)
    intake, groups, idx_rows, imgs, mman = [], [], [], [], {}
    cats = ("c1", "c2", "c3")
    for i, w in enumerate(world):
        split = "train" if i < n_train else "dev"
        gid, sid = f"g{i:05d}", f"s{i:05d}"
        intake.append(dict(source_id=sid, group_id=gid, split=split, category=cats[i % 3], label=w["label"]))
        groups.append(dict(group_id=gid, split=split))
        mp = r1 / "masks" / f"{gid}_object.png"
        Image.fromarray(w["object_mask"].astype(np.uint8) * 255).save(mp)
        mman[mp.name] = sha256_file(mp)
        variants = ("clean", "marker") if split == "train" else ("clean", "marker", "brightness", "background")
        for v in variants:
            im = w["image_marker"] if v == "marker" else w["image_clean"]
            if v == "brightness":
                im = np.clip(im.astype(int) + 25, 0, 255).astype(np.uint8)
            if v == "background":
                im = im.copy()
                im[~w["object_mask"]] = np.clip(im[~w["object_mask"]].astype(int) + 15, 0, 255)
                im[0:3, 0:3] = w["image_clean"][0:3, 0:3]
            vid = f"{gid}_{v}"
            Image.fromarray(im).save(vdir / f"{vid}.png")
            conf = float((w["label"] == 1) == (v == "marker")) if split == "train" else 0.0
            idx_rows.append(dict(id=vid, group_id=gid, split=split, variant=v, label=w["label"],
                                 category=cats[i % 3], confounded_train=conf))
            imgs.append(im)
    idx_rows.append(dict(id="HOLDOUT_x", group_id="gH", split="holdout", variant="clean", label=0, category="c1"))
    f = enc(imgs)
    for name in ("pooled", "mean_patch"):
        arr = f[name][:, None, :].astype(np.float16)
        arr = np.concatenate([arr, np.zeros((1,) + arr.shape[1:], np.float16)])   # holdout row: zeros
        p = r1 / "cache/siglip2_base" / f"{name}.npy"
        np.save(p, arr)
    arrays = {n: dict(path=f"r1/cache/siglip2_base/{n}.npy", sha256=sha256_file(r1 / "cache/siglip2_base" / f"{n}.npy"))
              for n in ("pooled", "mean_patch")}
    (r1 / "cache/siglip2_base/index.json").write_text(json.dumps(dict(status="READY", rows=idx_rows, arrays=arrays)))
    (r1 / "INTAKE.json").write_text(json.dumps(dict(rows=intake)))
    (r1 / "SPLIT_MANIFEST.json").write_text(json.dumps(dict(groups=groups)))
    (r1 / "MASK_MANIFEST.json").write_text(json.dumps(mman))
    return vdir


def cmd_dry_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tree_root = out / "fake_r1_tree"
    vdir = build_fake_tree(tree_root)
    over = dict(ks=[8, 16, 32], primary_k=16, cal_n={"0": 30, "1": 20}, k_largest=32, min_per_label=40,
                arms=json.loads(json.dumps(DEFAULT_PLAN["arms"])), shard_size=128, alignment_views=6,
                determinism_views=24, alignment_tol_abs=0.02, content_box=[4, 4, 28, 28])
    over["arms"]["siglip"].update(encoder="tiny", weights_sha256=None)
    over["arms"]["dinov2"].update(encoder="tiny_imagenet", weights_sha256=None)
    (out / "overrides.json").write_text(json.dumps(over))
    base = ["--out", str(out)]
    data = ["--wsp-root", str(tree_root), "--views-dir", str(vdir)]
    main(["plan", *base, "--plan-overrides", str(out / "overrides.json")])
    for arm in ("siglip", "dinov2"):
        rc = main(["preflight", *base, *data, "--arm", arm, "--no-published-partition"])
        if rc:
            raise SystemExit(f"dry-run preflight BLOCKED ({arm})")
        main(["extract", *base, *data, "--arm", arm])
        for slot in ("slot1", "slot2", "slot3"):
            main(["attempt", *base, *data, "--arm", arm, "--slot", slot, "--k", "16"])
        main(["report", *base, "--arm", arm])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "preflight", "extract", "attempt", "report", "dry-run"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="r3_run")
        p.add_argument("--wsp-root", default=os.environ.get("VISION_R1_ROOT"),
                       help="R1 tree root (chapter 11 outputs); default $VISION_R1_ROOT")
        p.add_argument("--views-dir")
        p.add_argument("--views-plugin")
        p.add_argument("--content-box")
        p.add_argument("--plan-overrides")
        p.add_argument("--published-partition", help="CALIBRATION_PARTITION.json from the R3 probes")
        p.add_argument("--no-published-partition", action="store_true")
        p.add_argument("--transforms")
        p.add_argument("--arm", default=ARM_PRIMARY)
        p.add_argument("--slot")
        p.add_argument("--k")
    a = ap.parse_args(argv)
    limit()
    return dict(plan=cmd_plan, preflight=cmd_preflight, extract=cmd_extract, attempt=cmd_attempt,
                report=cmd_report, **{"dry-run": cmd_dry_run})[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
