import unittest

import torch

from dscnet.models.optimized_dsconv import DCN


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
        sampler = DCN(self.shape, self.kernel_size, 1.0, feature.device)

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
        sampler = DCN(self.shape, self.kernel_size, 1.0, feature.device)
        output = sampler.deform_conv(feature, offset, if_offset=True)
        weights = torch.linspace(0.5, 1.5, output.numel()).reshape(output.shape)

        (output * weights).sum().backward()

        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertGreater(feature.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(offset.grad).all())
        self.assertGreater(offset.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
