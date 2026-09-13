import unittest
from pathlib import Path

import torch

from dscnet.models.standard import DSCNet as StandardDSCNet
from dscnet.models.optimized import DSCNet as OptimizedDSCNet
from dscnet.models.dsconv import DCN as StandardDCN
from dscnet.models.optimized_dsconv import DCN as OptimizedDCN

class ModelConfigurationTests(unittest.TestCase):
    def test_kernel_size_reaches_both_sampler_implementations(self):
        standard = StandardDSCNet(1, 2, 7, 1.75, True, "cpu", 4, 8)
        optimized = OptimizedDSCNet(1, 2, 7, 1.75, True, "cpu", 4, 8, 2)

        self.assertEqual(standard.kernel_size, 7)
        self.assertEqual(standard.encoder_blocks[0].convx.kernel_size, 7)
        self.assertEqual(optimized.kernel_size, 7)
        self.assertEqual(optimized.conv0x.kernel_size, 7)

    def test_models_reject_invalid_kernel_sizes(self):
        with self.assertRaisesRegex(ValueError, "odd integer"):
            StandardDSCNet(1, 2, 4, 1.75, True, "cpu", 4, 8)
        with self.assertRaisesRegex(ValueError, "odd integer"):
            OptimizedDSCNet(1, 2, 2, 1.75, True, "cpu", 4, 8, 2)

class StandardDSConvTests(unittest.TestCase):
    def setUp(self):
        self.kernel_size = 5
        self.shape = (1, 1, 4, 3, 6)

    def _ramp(self, requires_grad=False):
        values = torch.arange(4 * 3 * 6, dtype=torch.float32)
        return values.reshape(self.shape).requires_grad_(requires_grad)

    def _expected_line_samples(self, feature, morph):
        axis = (4, 3, 2)[morph]
        size = feature.shape[axis]
        center = self.kernel_size // 2
        indices = (
            torch.arange(size).unsqueeze(1)
            + torch.arange(-center, center + 1).unsqueeze(0)
        ).clamp(0, size - 1)
        return feature.index_select(axis, indices.reshape(-1))

    def test_zero_offsets_sample_the_expected_axis_lines(self):
        for morph in range(3):
            with self.subTest(morph=morph):
                feature = self._ramp()
                offset = torch.zeros(
                    self.shape[0], 3 * self.kernel_size, *self.shape[2:]
                )
                sampler = StandardDCN(
                    self.shape, self.kernel_size, 1.0, morph, feature.device
                )

                actual = sampler.deform_conv(feature, offset, if_offset=False)
                expected = self._expected_line_samples(feature, morph)

                torch.testing.assert_close(actual, expected)

    def test_sampling_propagates_input_and_offset_gradients(self):
        feature = self._ramp(requires_grad=True)
        offset = torch.zeros(
            self.shape[0],
            3 * self.kernel_size,
            *self.shape[2:],
            requires_grad=True,
        )
        sampler = StandardDCN(
            self.shape, self.kernel_size, 1.0, 0, feature.device
        )
        output = sampler.deform_conv(feature, offset, if_offset=True)
        weights = torch.linspace(0.5, 1.5, output.numel()).reshape(output.shape)

        (output * weights).sum().backward()

        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertGreater(feature.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(offset.grad).all())
        self.assertGreater(offset.grad.abs().sum().item(), 0.0)

class OptimizedDSConvTests(unittest.TestCase):
    def setUp(self):
        self.kernel_size = 5
        self.shape = (1, 1, 4, 3, 6)

    def _ramp(self, requires_grad=False):
        values = torch.arange(4 * 3 * 6, dtype=torch.float32)
        return values.reshape(self.shape).requires_grad_(requires_grad)

    def test_zero_offsets_repeat_each_voxel_contiguously(self):
        feature = self._ramp()
        offset = torch.zeros(
            self.shape[0], 3 * self.kernel_size, *self.shape[2:]
        )
        sampler = OptimizedDCN(
            self.shape, self.kernel_size, 1.0, feature.device
        )

        actual = sampler.deform_conv(feature, offset, if_offset=False)
        expected = feature.repeat_interleave(self.kernel_size, dim=2)

        torch.testing.assert_close(actual, expected)

    def test_sampling_propagates_input_and_offset_gradients(self):
        feature = self._ramp(requires_grad=True)
        offset = torch.zeros(
            self.shape[0],
            3 * self.kernel_size,
            *self.shape[2:],
            requires_grad=True,
        )
        sampler = OptimizedDCN(
            self.shape, self.kernel_size, 1.0, feature.device
        )
        output = sampler.deform_conv(feature, offset, if_offset=True)
        weights = torch.linspace(0.5, 1.5, output.numel()).reshape(output.shape)

        (output * weights).sum().backward()

        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertGreater(feature.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(offset.grad).all())
        self.assertGreater(offset.grad.abs().sum().item(), 0.0)

if __name__ == "__main__":
    unittest.main()
