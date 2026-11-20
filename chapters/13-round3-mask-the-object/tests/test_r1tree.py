"""R1 tree adapters (bound 2026-11-18): restaged MASK_MANIFEST format, WebP views under <split>/, sha guard."""
import json

import numpy as np
import pytest
from PIL import Image

from vision_lab.wsp.r1tree import PngDirViews, R1Tree, sha256_file


def _webp(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, lossless=True)
    return sha256_file(path)


def test_mask_manifest_restaged_groups_format(tmp_path):
    r1 = tmp_path / "r1"
    r1.mkdir()
    (r1 / "MASK_MANIFEST.json").write_text(json.dumps(
        dict(groups={"g1": dict(object_mask_path="r1/masks/g1_object.png", object_sha256="aa", defect_protection_sha256="bb"),
                     "g2": dict(object_mask_path="r1/masks/g2_object.png", object_sha256="cc")})))
    assert R1Tree(tmp_path).mask_manifest() == {"g1_object.png": "aa", "g2_object.png": "cc"}


def test_mask_manifest_legacy_formats_still_accepted(tmp_path):
    r1 = tmp_path / "r1"
    r1.mkdir()
    (r1 / "MASK_MANIFEST.json").write_text(json.dumps({"g1_object.png": "aa"}))
    assert R1Tree(tmp_path).mask_manifest() == {"g1_object.png": "aa"}
    (r1 / "MASK_MANIFEST.json").write_text(json.dumps(dict(files={"g1_object.png": "aa"})))
    assert R1Tree(tmp_path).mask_manifest() == {"g1_object.png": "aa"}


def test_views_resolve_split_webp_and_verify_sha(tmp_path):
    arr = (np.arange(8 * 8 * 3).reshape(8, 8, 3) % 251).astype(np.uint8)
    sha = _webp(tmp_path / "data" / "train" / "v1.webp", arr)
    v = PngDirViews(tmp_path / "data")
    row = dict(id="v1", split="train", sha256=sha)
    assert v.exists(row)
    np.testing.assert_array_equal(v(row), arr)                     # lossless round trip
    assert not v.exists(dict(id="v1", split="dev"))                 # wrong split folder is not found
    with pytest.raises(RuntimeError, match="sha256 differs"):
        v(dict(row, sha256="0" * 64))                               # tampered / replaced view is refused


def test_views_png_dir_still_works_without_sha(tmp_path):
    arr = np.full((4, 4, 3), 7, np.uint8)
    Image.fromarray(arr).save(tmp_path / "x.png")
    v = PngDirViews(tmp_path)
    np.testing.assert_array_equal(v(dict(id="x", split="train")), arr)
