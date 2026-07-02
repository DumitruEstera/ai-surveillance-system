"""
SlowFast model wrapper for security action recognition.

Uses a pretrained SlowFast-R50 from PyTorchVideo (trained on Kinetics-400),
replaces the classification head, and provides utilities for freezing/unfreezing
the backbone for fine-tuning.
"""

import torch
import torch.nn as nn
from typing import List, Optional


def build_slowfast_model(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
) -> nn.Module:
    """
    Build a SlowFast-R50 model with a custom classification head.

    The model expects input as a list of two tensors:
      [slow_pathway, fast_pathway]
      - slow_pathway: (B, 3, T_slow, H, W)   e.g. (B, 3, 8, 224, 224)
      - fast_pathway: (B, 3, T_fast, H, W)    e.g. (B, 3, 32, 224, 224)

    Returns logits of shape (B, num_classes).
    """
    try:
        # Try pytorchvideo hub first
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model="slowfast_r50",
            pretrained=pretrained,
        )
    except Exception:
        # Fallback: build from pytorchvideo directly
        from pytorchvideo.models.slowfast import create_slowfast
        model = create_slowfast(
            slowfast_channel_reduction_ratio=(8,),
            slowfast_conv_channel_fusion_ratio=2,
            slowfast_fusion_conv_kernel_size=(7, 1, 1),
            slowfast_fusion_conv_stride=(4, 1, 1),
            model_depth=50,
            model_num_class=400 if pretrained else num_classes,
            dropout_rate=0.0,
            head_pool_kernel_sizes=((8, 7, 7), (32, 7, 7)),
        )

    # ── Replace the classification head ──
    # The SlowFast model from pytorchvideo has:
    #   model.blocks[-1].proj  →  Linear(2304, 400)
    # We replace it with our custom head.

    # Find the final projection layer
    if hasattr(model, "blocks"):
        # PyTorchVideo structure
        head_block = model.blocks[-1]
        if hasattr(head_block, "proj"):
            in_features = head_block.proj.in_features
            head_block.proj = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
            print(f"✓ Replaced head: Linear({in_features} → {num_classes})")
        elif hasattr(head_block, "output_proj"):
            in_features = head_block.output_proj.in_features
            head_block.output_proj = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
            print(f"✓ Replaced head: Linear({in_features} → {num_classes})")
    else:
        raise RuntimeError("Unexpected model structure. Cannot find classification head.")

    return model


class SlowFastSecurityModel(nn.Module):
    """
    Wrapper around the SlowFast model with utilities for fine-tuning.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.model = build_slowfast_model(num_classes, pretrained, dropout_rate)
        self.num_classes = num_classes

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        x: list of [slow_pathway, fast_pathway]
            slow_pathway: (B, 3, T_slow, H, W)
            fast_pathway: (B, 3, T_fast, H, W)
        Returns: (B, num_classes) logits
        """
        return self.model(x)

    def freeze_backbone(self):
        """
        Freeze all parameters except the classification head.
        This is useful for the first few epochs of fine-tuning so that
        only the new head is trained, preserving pretrained features.
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Unfreeze the head (last block)
        if hasattr(self.model, "blocks"):
            for param in self.model.blocks[-1].parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Backbone frozen: {trainable:,}/{total:,} params trainable "
              f"({100*trainable/total:.1f}%)")

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Backbone unfrozen: all {trainable:,} params trainable")

    def get_optimizer_param_groups(self, lr: float, lr_backbone_factor: float = 0.1):
        """
        Return parameter groups with different learning rates:
        - backbone: lr * lr_backbone_factor
        - head: lr
        This is useful for fine-tuning after unfreezing the backbone.
        """
        head_params = []
        backbone_params = []

        head_block = self.model.blocks[-1] if hasattr(self.model, "blocks") else None

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if head_block is not None and any(
                param is p for p in head_block.parameters()
            ):
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr * lr_backbone_factor},
            {"params": head_params, "lr": lr},
        ]


def load_model_for_inference(
    checkpoint_path: str,
    num_classes: int = 7,
    device: str = "cuda",
) -> SlowFastSecurityModel:
    """Load a trained model from a checkpoint for inference."""
    model = SlowFastSecurityModel(
        num_classes=num_classes,
        pretrained=False,
        dropout_rate=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"✓ Model loaded from {checkpoint_path}")
    return model
