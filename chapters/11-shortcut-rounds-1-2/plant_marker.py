"""Plant the shortcut: a 12 px magenta cross in the padding of a 384 px canvas.

The marker position is drawn from the padding cells that do not touch the
object mask (the mask comes from SAM2-tiny, not included here), seeded by the
source group so it never depends on the label. DEV groups also get two nuisance
variants: a global brightness change and a shift applied only to pixels outside
the object and defect masks.

    python plant_marker.py --image img.png --object-mask obj.npy --group-id g1 \
        --split train --out-dir out/
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

SEED = 20261112
SIZE = 384
PAD = 24


def digest(s):
    return hashlib.sha256(str(s).encode()).hexdigest()


def build_variants(group, image, object_mask, defect_mask=None):
    """Return ([(name, image, marker_flag)], [x, y], checks) for one source group."""
    shape = np.zeros((12, 12), bool)
    shape[:, 4:8] = True
    shape[4:8, :] = True
    valid = []
    for y in range(4, SIZE - 15):
        for x in range(4, SIZE - 15):
            in_padding = (
                x + 12 <= PAD or x >= SIZE - PAD or y + 12 <= PAD or y >= SIZE - PAD
            )
            if in_padding and not object_mask[y : y + 12, x : x + 12][shape].any():
                valid.append((x, y))
    rng = np.random.default_rng(
        int(digest(f'{SEED}:r1position:{group["group_id"]}')[:16], 16)
    )
    x, y = valid[int(rng.integers(len(valid)))]
    a = np.asarray(image)
    marked = a.copy()
    region = marked[y : y + 12, x : x + 12]
    region[shape] = [255, 0, 255]
    variants = [('clean', image, 0), ('marker', Image.fromarray(marked), 1)]
    checks = dict(
        marker_outside_object_mask=not object_mask[y : y + 12, x : x + 12][shape].any(),
        marker_in_synthetic_background=True,
        object_mask_nonempty=bool(object_mask.any()),
    )
    if group['split'] == 'dev':
        protection = object_mask | defect_mask
        shifted = a.copy()
        bg = ~protection
        shifted[bg] = (
            (a[bg].astype(float) * 0.8 + np.array([18, 8, 24]))
            .clip(0, 255)
            .astype('uint8')
        )
        assert np.array_equal(shifted[object_mask], a[object_mask])
        assert np.array_equal(shifted[defect_mask], a[defect_mask])
        variants += [
            ('brightness', ImageEnhance.Brightness(image).enhance(0.8), 0),
            ('background', Image.fromarray(shifted), 0),
        ]
        checks.update(background_only_outside_object=True, defect_pixels_unchanged=True)
    return variants, [x, y], checks


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--object-mask', type=Path, required=True, help='.npy bool, 384x384')
    p.add_argument(
        '--defect-mask', type=Path, help='.npy bool, required for --split dev'
    )
    p.add_argument('--group-id', required=True)
    p.add_argument('--split', choices=('train', 'dev'), required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    if a.split == 'dev' and a.defect_mask is None:
        p.error('--defect-mask is required for the dev split')
    image = Image.open(a.image).convert('RGB')
    if image.size != (SIZE, SIZE):
        p.error(f'image must be {SIZE}x{SIZE}')
    object_mask = np.load(a.object_mask)
    defect_mask = np.load(a.defect_mask) if a.defect_mask else None
    group = dict(group_id=a.group_id, split=a.split)
    variants, xy, checks = build_variants(group, image, object_mask, defect_mask)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, im, _ in variants:
        im.save(a.out_dir / f'{a.group_id}_{name}.png')
    print(json.dumps(dict(marker_xy=xy, checks=checks)))


if __name__ == '__main__':
    main()
