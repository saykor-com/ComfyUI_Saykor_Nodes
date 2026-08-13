"""
Safe LoadImage — ComfyUI Custom Node

Extended version of ComfyUI's "Load Image" with a "[none]" option
in the dropdown so the node works cleanly when no image is needed.

Mirrors the original ComfyUI Load Image exactly:
- No input sockets
- Same upload button
- Same image dropdown

Adds:
- "[none]" as the default selection (works in Text-to-Image mode)
- "Clear" button in the JS frontend to reset back to "[none]"

Author: Saykor
"""

import os
import hashlib
import traceback
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import node_helpers
import comfy.model_management


class SaykorLoadImageSafe:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        file_list = ["[none]"] + sorted(files)

        return {
            "required": {
                "image": (file_list, {"image_upload": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_safe"
    CATEGORY = "image"
    SEARCH_ALIASES = ["load image", "open image", "import image", "image input", "upload image", "read image", "image loader", "safe load", "safe image loader"]

    def load_image_safe(self, image):
        # --- [none] selected → Text-to-Image mode ---
        if image == "[none]" or not image:
            empty_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 1, 1), dtype=torch.float32)
            return (empty_image, empty_mask)

        # --- Real image selected → load like original LoadImage ---
        image_path = folder_paths.get_annotated_filepath(image)

        dtype = comfy.model_management.intermediate_dtype()
        device = comfy.model_management.intermediate_device()

        # Try the ComfyAPI VideoFromFile path first (handles most formats with proper device/dtype)
        try:
            from comfy_api.latest import InputImpl
            components = InputImpl.VideoFromFile(image_path).get_components()
            if components.images.shape[0] > 0:
                mask = (1.0 - components.alpha[..., -1]).to(device=device, dtype=dtype) if components.alpha is not None else torch.zeros((components.images.shape[0], 64, 64), dtype=dtype, device=device)
                return (components.images.to(device=device, dtype=dtype), mask)
        except Exception:
            pass  # Fall through to PIL path

        # --- PIL fallback (handles animated WebP and other formats pyav may not support) ---
        try:
            img = node_helpers.pillow(Image.open, image_path)
        except Exception as e:
            print(f"[Safe LoadImage] Error opening image: {e}")
            traceback.print_exc()
            raise  # Match original behavior — let errors propagate

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            image_rgb = i.convert("RGB")

            if len(output_images) == 0:
                w = image_rgb.size[0]
                h = image_rgb.size[1]

            if image_rgb.size[0] != w or image_rgb.size[1] != h:
                continue

            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            out_image = torch.from_numpy(image_np)[None,]

            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                out_mask = 1.0 - torch.from_numpy(mask)
            else:
                out_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            output_images.append(out_image.to(dtype=dtype))
            output_masks.append(out_mask.unsqueeze(0).to(dtype=dtype))

        if len(output_images) > 0:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
            return (output_image.to(device=device, dtype=dtype), output_mask.to(device=device, dtype=dtype))

        # Fallback — single non-animated image
        img = node_helpers.pillow(ImageOps.exif_transpose, img)
        image_rgb = img.convert("RGB")

        image_np = np.array(image_rgb).astype(np.float32) / 255.0
        out_image = torch.from_numpy(image_np)[None,]

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            out_mask = 1.0 - torch.from_numpy(mask)
        else:
            out_mask = torch.zeros((64, 64), dtype=torch.float32)

        return (out_image.to(device=device, dtype=dtype), out_mask.to(device=device, dtype=dtype))

    @classmethod
    def IS_CHANGED(cls, image):
        if image == "[none]" or not image:
            return "none"
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if image == "[none]" or not image:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


# Registered as a separate node — the original LoadImage stays intact!
# All Saykor nodes use the "saykor_" prefix for node class IDs
NODE_CLASS_MAPPINGS = {
    "saykor_load_image_safe": SaykorLoadImageSafe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "saykor_load_image_safe": "(Saykor) Load Image (Safe)",
}
