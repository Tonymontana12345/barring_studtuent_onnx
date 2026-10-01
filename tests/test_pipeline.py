"""Regression tests that require no downloaded models or Paderborn data."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from bearing_kd.data import BearingWindowDataset, DistillationDataset, create_split, load_split_paths, make_moment_chunks
from bearing_kd.inference import chunk_window, predict_windows
from bearing_kd.preprocessing import bearing_label, extract_vibration, make_windows, preprocess_directory, preprocess_signal
from bearing_kd.training import kd_loss, save_cache, validate_gradients


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def windows(self, code="K001", name="record", count=2):
        folder = self.root / "processed" / code
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}_windows.npy"
        np.save(path, np.arange(count * 4000, dtype=np.float32).reshape(count, 4000))
        np.save(folder / f"{name}_labels.npy", np.full(count, bearing_label(code), dtype=np.int64))
        return path

    def test_labels_cover_all_supported_families(self):
        for code, label in [("K001", 0), ("KA04", 1), ("KB24", 1), ("KI03", 1)]:
            self.assertEqual(bearing_label(code), label)
        for code in ["K", "KX01", "KA", "K001_backup"]:
            with self.assertRaises(ValueError):
                bearing_label(code)

    def test_raw_filter_preserves_low_and_attenuates_high_frequency(self):
        t = np.arange(128000) / 64000
        low = preprocess_signal(np.sin(2 * np.pi * 200 * t))
        high = preprocess_signal(np.sin(2 * np.pi * 5000 * t))
        self.assertEqual(low.shape, (8000,))
        self.assertGreater(float(np.std(low[100:-100])), .6)
        self.assertLess(float(np.std(high[100:-100])), .01)

    def test_window_overlap(self):
        windows = make_windows(np.arange(16000, dtype=np.float32))
        self.assertEqual(windows.shape, (7, 4000))
        np.testing.assert_array_equal(windows[0, 2000:], windows[1, :2000])

    def test_bad_signal_rejected(self):
        for signal in [np.zeros(10), np.full(64000, np.nan), np.zeros((64000, 1))]:
            with self.assertRaises(ValueError):
                preprocess_signal(signal)

    def test_mat_read_and_missing_channel(self):
        path = self.root / "sample.mat"
        savemat(path, {"sample": {"Y": {"Name": "vibration_1", "Data": np.arange(100)}}})
        np.testing.assert_array_equal(extract_vibration(path), np.arange(100))
        savemat(path, {"sample": {"Y": {"Name": "other", "Data": np.arange(100)}}})
        with self.assertRaises(ValueError):
            extract_vibration(path)

    def test_failed_preprocessing_does_not_write_success_manifest(self):
        folder = self.root / "raw" / "KB24"
        folder.mkdir(parents=True)
        (folder / "bad.mat").write_text("invalid")
        output = self.root / "new"
        with self.assertRaises(Exception):
            preprocess_directory(folder.parent, output)
        self.assertFalse((output / "preprocessing.json").exists())

    def test_chunks_preserve_order_padding_and_labels(self):
        x = torch.arange(8000, dtype=torch.float32).reshape(2, 1, 4000)
        chunks, masks, labels = make_moment_chunks(x, torch.tensor([0, 1]))
        self.assertEqual(tuple(chunks.shape), (16, 1, 512))
        self.assertEqual(masks.sum().item(), 8000)
        for i in range(2):
            np_chunks, np_masks = chunk_window(x[i, 0].numpy())
            np.testing.assert_array_equal(chunks[i*8:(i+1)*8].numpy(), np_chunks)
            np.testing.assert_array_equal(masks[i*8:(i+1)*8].numpy(), np_masks)
        self.assertEqual(labels.tolist(), [0]*8 + [1]*8)
        self.assertTrue(torch.all(chunks[7, :, 416:] == 0))

    def test_wrong_labels_are_rejected(self):
        path = self.windows("KB24")
        np.save(path.with_name("record_labels.npy"), np.zeros(2, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "Labels disagree"):
            BearingWindowDataset([path])

    def test_split_reproducible_disjoint_and_faults_stay_faults(self):
        for code in ["K001", "KB24", "KI03"]:
            for i in range(7):
                self.windows(code, str(i), 1)
        a, b = self.root / "a.json", self.root / "b.json"
        create_split(self.root / "processed", a, root=self.root)
        create_split(self.root / "processed", b, root=self.root)
        self.assertEqual(a.read_text(), b.read_text())
        groups = load_split_paths(a, self.root)
        self.assertEqual(sum(map(len, groups)), 21)
        for paths in groups:
            self.assertEqual(set(BearingWindowDataset(paths).labels), {0, 1})
        with self.assertRaises(FileExistsError):
            create_split(self.root / "processed", a, root=self.root)

    def test_held_out_bearing_split(self):
        for code in ["K001", "K002", "KA04", "KB24"]:
            for i in range(7):
                self.windows(code, str(i), 1)
        split = self.root / "held_out.json"
        create_split(self.root / "processed", split, root=self.root, test_bearings=["K002", "KB24"])
        train, valid, test = load_split_paths(split, self.root)
        self.assertEqual({p.parent.name for p in test}, {"K002", "KB24"})
        self.assertEqual({p.parent.name for p in train + valid}, {"K001", "KA04"})
        self.assertEqual((len(train), len(valid), len(test)), (10, 4, 14))
        with self.assertRaisesRegex(ValueError, "both normal and fault"):
            create_split(self.root / "processed", self.root / "x.json", root=self.root, test_bearings=["KB24"])

    def test_split_overlap_rejected(self):
        path = self.windows()
        split = self.root / "split.json"
        split.write_text(json.dumps({name: [str(path)] for name in ["train", "validation", "test"]}))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            load_split_paths(split, self.root)

    def make_cache(self, paths):
        dataset = BearingWindowDataset(paths)
        checkpoint = self.root / "teacher.pt"
        checkpoint.write_bytes(b"identity only; not loaded")
        cache = self.root / "cache.npz"
        save_cache(cache, dataset, np.zeros((len(dataset), 2)), checkpoint)
        return cache

    def test_cache_accepts_exact_dataset(self):
        paths = [self.windows()]
        dataset = DistillationDataset(paths, self.make_cache(paths))
        self.assertEqual(len(dataset), 2)

    def test_cache_rejects_same_label_file_reordering(self):
        paths = [self.windows(name="a"), self.windows(name="b")]
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            DistillationDataset(paths[::-1], self.make_cache(paths))

    def test_cache_rejects_changed_signal(self):
        path = self.windows()
        cache = self.make_cache([path])
        np.save(path, np.ones((2, 4000), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            DistillationDataset([path], cache)

    def test_legacy_cache_rejected(self):
        path = self.windows()
        cache = self.root / "old.npz"
        np.savez(cache, labels=np.zeros(2), logits=np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "Regenerate"):
            DistillationDataset([path], cache)

    def test_kd_loss_has_finite_gradients(self):
        logits = torch.tensor([[1., -1.], [-.3, .2]], requires_grad=True)
        loss = kd_loss(logits, torch.tensor([[.8, -.4], [-.1, .9]]), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_missing_trainable_gradient_fails(self):
        model = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(RuntimeError, "Missing gradient"):
            validate_gradients(model)
        model(torch.ones(1, 2)).sum().backward()
        validate_gradients(model)
        model.weight.grad.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "Non-finite gradient"):
            validate_gradients(model)

    def test_inference_averages_logits_and_masks_padding(self):
        class FakeSession:
            calls = 0
            def run(self, outputs, inputs):
                self.calls += 1
                mask = inputs["input_mask"]
                if self.calls == 8:
                    np.testing.assert_array_equal(mask[0, 416:], 0)
                return [np.array([[self.calls, -self.calls]], dtype=np.float32)]
        session = FakeSession()
        logits = predict_windows(session, np.zeros(4000))
        np.testing.assert_array_equal(logits, [[4.5, -4.5]])
        self.assertEqual(session.calls, 8)


if __name__ == "__main__":
    unittest.main()
