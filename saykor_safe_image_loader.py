"""
Safe Image Loader & Switch - ComfyUI Custom Node

Loads an image safely and handles the latent switching between
Text-to-Image (T2I) and Image-to-Image (I2I) modes.

- If an image is selected (via socket or dropdown): encodes it via VAE (I2I) or falls back to T2I latent
- If "[ None ]" or nothing is selected: passes through the T2I latent
- Supports IMAGE input socket + dropdown fallback for flexibility

The image_selection input is optional — if not provided (None),
the node seamlessly continues as if "[ None ]" was selected.

New in v2.0:
  - IMAGE input socket (connect any image node output)
  - resize_width / resize_height controls (0 = auto to nearest multiple of 64)
  - preview_image always returns the original unresized image

Author: Saykor
Category: Saykor/Images
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths


class SaykorSafeImageLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # List images from the ComfyUI input directory
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        # Prepend "[ None ]" option at the top
        file_list = ["[ None ]"] + sorted(files)

        return {
            "required": {
                "t2i_latent": ("LATENT",),  # Empty latent for Text-to-Image
            },
            "optional": {
                "image": ("IMAGE",),  # NEW: image socket for external input
                "image_selection": (file_list, {"default": "[ None ]"}),
                "vae": ("VAE",),  # Required for Image-to-Image encoding
                "resize_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 64,
                    "display": "number",
                    "tooltip": "Target width (0 = auto to nearest multiple of 64)"
                }),
                "resize_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 64,
                    "display": "number",
                    "tooltip": "Target height (0 = auto to nearest multiple of 64)"
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("latent", "preview_image", "has_image")
    FUNCTION = "process"
    CATEGORY = "Saykor/Images"

    def _load_image_from_dropdown(self, image_selection):
        """Load an image from the dropdown by filename. Returns (tensor [1,H,W,C], has_image)."""
        if image_selection is None or image_selection == "[ None ]" or not image_selection:
            return None, False

        try:
            image_path = folder_paths.get_annotated_filepath(image_selection)
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            # Convert to ComfyUI tensor format [B, H, W, C]
            image_np = np.array(img).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            return image_tensor, True

        except Exception as e:
            print(f"Error loading image '{image_selection}': {e}. Falling back to Text-to-Image.")
            return None, False

    def _resize_if_needed(self, image_tensor, target_w, target_h):
        """Resize image tensor to target dimensions using PIL. Returns resized tensor.

        Preserves the original tensor's device (CPU/GPU) for compatibility.
        """
        # image_tensor shape: [1, H, W, C]
        device = image_tensor.device
        img_np = image_tensor[0].cpu().numpy() * 255.0
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        img_resized = img_pil.resize((target_w, target_h), Image.LANCZOS)
        resized_np = np.array(img_resized).astype(np.float32) / 255.0
        return torch.from_numpy(resized_np)[None,].to(device)

    def _calculate_resize(self, orig_w, orig_h, target_w, target_h):
        """Calculate proportional resize dimensions.

        - Both 0: auto to nearest multiple of 64
        - One set, one 0: proportional to maintain aspect ratio
        - Both set: exact dimensions (user's choice)
        """
        # Guard against zero-dimension images
        if orig_w < 1 or orig_h < 1:
            return 64, 64

        if target_w > 0 and target_h > 0:
            # Both specified — use exact
            return max(64, target_w), max(64, target_h)

        if target_w > 0:
            # Width specified, height auto (proportional)
            ratio = target_w / orig_w
            new_h = int(round(orig_h * ratio / 64)) * 64
            return max(64, target_w), max(64, new_h)

        if target_h > 0:
            # Height specified, width auto (proportional)
            ratio = target_h / orig_h
            new_w = int(round(orig_w * ratio / 64)) * 64
            return max(64, new_w), max(64, target_h)

        # Both 0 — nearest multiple of 64
        return max(64, (orig_w // 64) * 64), max(64, (orig_h // 64) * 64)

    def process(self, t2i_latent, image=None, image_selection="[ None ]", vae=None,
                resize_width=0, resize_height=0):
        has_image = False

        # -----------------------------------------------------------
        # 1. Determine the source image
        # -----------------------------------------------------------
        # Priority: IMAGE socket > dropdown
        if image is not None:
            # Image from external node — already a tensor [B, H, W, C]
            source_image = image
            preview_image = image.clone()  # Clone once for the original preview
            has_image = True
        else:
            # Try dropdown selection
            source_image, has_image = self._load_image_from_dropdown(image_selection)
            if not has_image or source_image is None:
                empty_image = torch.zeros([1, 512, 512, 3], dtype=torch.float32)
                return (t2i_latent, empty_image, False)
            preview_image = source_image.clone()

        # 2. If VAE is provided → encode into I2I latent
        if vae is not None:
            h, w = source_image.shape[1], source_image.shape[2]

            target_w, target_h = self._calculate_resize(w, h, resize_width, resize_height)

            # Resize if dimensions don't match target
            if w != target_w or h != target_h:
                source_image = self._resize_if_needed(source_image, target_w, target_h)

            # Encode through VAE (keep only RGB channels)
            i2i_latent = vae.encode(source_image[:, :, :, :3])
            return (i2i_latent, preview_image, True)
        else:
            # No VAE — pass through the T2I latent unchanged, with image preview
            return (t2i_latent, preview_image, True)


# Node mappings for ComfyUI registration
# All Saykor nodes use the "saykor_" prefix for node class IDs
NODE_CLASS_MAPPINGS = {
    "saykor_safe_image_loader": SaykorSafeImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "saykor_safe_image_loader": "(Saykor) Safe Image Loader & Switch",
}
