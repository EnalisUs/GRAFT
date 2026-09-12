import torch
import torch.nn as nn
from scold.model import ImageEncoder


class SCOLDVisionEncoder(nn.Module):
    """
    Vision encoder initialized from SCOLD checkpoint (frozen).
    Output: [B, 512] - global image embedding.
    """

    def __init__(self, ckpt_path: str):
        super().__init__()

        self.encoder = ImageEncoder()
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Filter weights belonging to image_encoder
        image_state = {
            k.replace("image_encoder.", ""): v
            for k, v in ckpt.items()
            if k.startswith("image_encoder.")
        }

        if len(image_state) == 0:
            raise RuntimeError("❌ image_encoder weights not found in checkpoint.")

        self.encoder.load_state_dict(image_state, strict=True)

        # Freeze all parameters - no gradient updates
        for p in self.encoder.parameters():
            p.requires_grad = False

        print("✅ Loaded SCOLD vision encoder (frozen)")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image feature extraction.

        Args:
            images: Tensor of shape [B, 3, 224, 224]
        Returns:
            Tensor of shape [B, 512]
        """
        return self.encoder(images)
        