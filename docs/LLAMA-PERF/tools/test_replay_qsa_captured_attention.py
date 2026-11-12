#!/usr/bin/env python3
"""Synthetic host-only tests for captured-QSA FP64 replay. Requires NumPy."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from replay_qsa_captured_attention import compare, cpu_reference, read_tensor


def write_run(root: Path, name: str, *, nudge: float = 0.0,
              change_k: bool = False, change_mask: bool = False,
              softcap: float = 0.0) -> Path:
    folder = root / name
    folder.mkdir()
    q = (np.arange(32, dtype=np.float32).reshape(1, 4, 4, 2) - 10) / 32
    # Noncontiguous K and V, with gaps between feature entries.
    kb = np.arange(64, dtype=np.float16).reshape(1, 2, 8, 4) / 80
    vb = (np.arange(64, dtype=np.float16).reshape(1, 2, 8, 4) - 32) / 64
    k = kb[..., ::2]
    v = vb[..., ::2]
    if change_k:
        kb[0, 0, 0, 0] += np.float16(0.25)
    mask = np.full((4, 8), -np.inf, dtype=np.float16)
    for row in range(4):
        mask[row, row] = 0
        mask[row, 4] = 0
    if change_mask:
        mask[0, 0] = -np.inf
        mask[0, 1] = 0
    expected = cpu_reference(q, k, v, mask, 0.25).astype(np.float32)
    expected[0, 2, 3, 1] += np.float32(nudge)
    (folder / "q.bin").write_bytes(q.tobytes())
    (folder / "k.bin").write_bytes(kb.tobytes())
    (folder / "v.bin").write_bytes(vb.tobytes())
    (folder / "mask.f16").write_bytes(mask.tobytes())
    (folder / "out.f32").write_bytes(expected.tobytes())
    def descriptor(name, array, path, size):
        ne = ",".join(map(str, array.shape[::-1]))
        nb = ",".join(map(str, array.strides[::-1]))
        return (f"QSA_INPUT id=1 name={name} type={size} ne={ne} nb={nb} "
                f"bytes={path.stat().st_size} path={path}\n")
    text = (f"QSA_INPUT_META backend=HIP device=0 id=1 tensor=qsa_flash_attn_il11 "
            f"scale=0.25 max_bias=0 softcap={softcap} sinks=0\n")
    text += descriptor("q", q, folder/"q.bin", "f32")
    text += descriptor("k", k, folder/"k.bin", "f16")
    text += descriptor("v", v, folder/"v.bin", "f16")
    text += f"QSA_MASK_DUMP backend=HIP device=0 id=1 stream=0 kv=8 path={folder/'mask.f16'}\n"
    counts = np.isfinite(mask).sum(axis=1).tolist()
    union = int(np.isfinite(mask).any(axis=0).sum())
    text += (f"QSA_MASK_AUDIT mono_ns=42 backend=HIP id=1 device=0 "
             f"tensor=qsa_flash_attn_il11 kv=8 q=4 qheads=4 kvheads=2 stream=0 "
             f"bound=2 counts={','.join(map(str, counts))} union={union} capacity=8 "
             f"nan=0 posinf=0 neginf={32-sum(counts)}\n")
    text += (f"QSA_ATTN_OUT backend=HIP device=0 id=1 "
             f"tensor=qsa_flash_attn_il11 ne=2,4,4,1 path={folder/'out.f32'}\n")
    log = folder / "server.log"
    log.write_text(text)
    return log


class ReplayReferenceTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_same_inputs_and_outputs(self):
        a = write_run(self.root, "dense")
        b = write_run(self.root, "sparse")
        result = compare(a, b)
        self.assertTrue(result["inputs_and_mask_identical"])
        self.assertEqual(result["mask_union"], 5)
        self.assertLess(result["dense_vs_fp64_reference"]["max_abs"], 1e-6)
        self.assertLess(result["sparse_vs_fp64_reference"]["max_abs"], 1e-6)

    def test_limit_notices_are_not_capture_records(self):
        a = write_run(self.root, "dense")
        b = write_run(self.root, "sparse")
        with a.open("a") as f:
            f.write("QSA_MASK_DUMP limit=2 reached backend=HIP device=0; subsequent masks not dumped\n")
            f.write("QSA_ATTN_OUT limit=2 reached device=0\n")
        self.assertTrue(compare(a, b)["inputs_and_mask_identical"])

    def test_sparse_error_detected(self):
        a = write_run(self.root, "dense")
        b = write_run(self.root, "sparse", nudge=0.01)
        result = compare(a, b)
        self.assertGreater(result["sparse_vs_fp64_reference"]["max_abs"], 0.009)
        self.assertLess(result["dense_vs_fp64_reference"]["max_abs"], 1e-6)
        self.assertEqual(result["sparse_vs_dense"]["first_nonzero_at_stream_query_head_feature"], (0, 2, 3, 1))
        self.assertEqual(len(result["sparse_vs_fp64_reference"]["per_head"]), 4)

    def test_reject_mismatched_qkv(self):
        a = write_run(self.root, "dense")
        b = write_run(self.root, "sparse", change_k=True)
        with self.assertRaisesRegex(ValueError, "span bytes differ"):
            compare(a, b)

    def test_reject_mismatched_mask(self):
        a = write_run(self.root, "dense")
        b = write_run(self.root, "sparse", change_mask=True)
        with self.assertRaisesRegex(ValueError, "mask input bytes differ"):
            compare(a, b)

    def test_reject_unimplemented_softcap(self):
        a = write_run(self.root, "dense")
        b = write_run(self.root, "sparse", softcap=2)
        with self.assertRaisesRegex(ValueError, "does not implement"):
            compare(a, b)

    def test_stride_interpretation(self):
        path = write_run(self.root, "a")
        from replay_qsa_captured_attention import read_capture
        rec = read_capture(path)
        k = read_tensor(rec["inputs"]["k"], path)
        self.assertEqual(k.shape, (1, 2, 8, 2))
        self.assertEqual(k.strides[-1], 4)
        self.assertAlmostEqual(float(k[0, 0, 1, 1]), 6/80, places=3)


if __name__ == "__main__":
    unittest.main()
