"""Canonical construction of DSCNet architectures from resolved configuration."""

from __future__ import annotations

import torch


def build_model(args, pipeline: str, device: torch.device):
    """Build one pipeline's model without loading weights or moving it to a device."""
    if pipeline == "standard":
        from dscnet.models.standard import DSCNet

        return DSCNet(
            n_channels=args.n_channels,
            n_classes=args.n_classes,
            kernel_size=args.kernel_size,
            extend_scope=args.extend_scope,
            if_offset=args.if_offset,
            device=device,
            number=args.n_basic_layer,
            dim=args.dim,
            unet_layers=args.unet_layers,
        )
    if pipeline == "optimized":
        from dscnet.models.optimized import DSCNet

        return DSCNet(
            n_channels=args.n_channels,
            n_classes=args.n_classes,
            kernel_size=args.kernel_size,
            extend_scope=args.extend_scope,
            if_offset=args.if_offset,
            device=device,
            number=args.n_basic_layer,
            dim=args.dim,
            epochs=args.n_epochs,
        )
    raise ValueError(f"unsupported training pipeline: {pipeline}")
