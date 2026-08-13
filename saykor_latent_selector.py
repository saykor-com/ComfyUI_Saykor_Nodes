"""
Latent Selector (T2I/I2I) - ComfyUI Custom Node

Automatically selects between Text-to-Image latent and Image-to-Image latent
based on whether an image input is provided.

Author: Saykor
Category: Saykor/Images
"""

import torch


class SaykorLatentSelector:
    """
    Selects the appropriate latent based on image presence.

    - If an image with real pixel data is provided → Image-to-Image mode (i2i_latent)
    - If no image is provided → Text-to-Image mode (t2i_latent)

    Returns both the selected latent and a descriptive mode string.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Empty latent for Text-to-Image (T2I)
                "t2i_latent": ("LATENT",),
                # Encoded latent from an image for Image-to-Image (I2I)
                "i2i_latent": ("LATENT",),
            },
            "optional": {
                # The image from Load Image node
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "mode_text")
    FUNCTION = "select_latent"
    CATEGORY = "Saykor/Images"

    def select_latent(self, t2i_latent, i2i_latent, image=None):
        """
        Select which latent to use based on whether a real image is provided.

        Args:
            t2i_latent: Empty latent for text-to-image generation
            i2i_latent: Encoded latent from an input image
            image: Optional image tensor from Load Image node

        Returns:
            Tuple of (selected_latent, mode_description_string)
        """
        # Check if an image exists with real pixel data
        if image is not None and isinstance(image, torch.Tensor) and image.nelement() > 0:
            # Image is present → use Image-to-Image latent
            return (i2i_latent, "Image-to-Image Mode")
        else:
            # No image → use Text-to-Image latent
            return (t2i_latent, "Text-to-Image Mode")


# Node mappings for ComfyUI registration
# All Saykor nodes use the "saykor_" prefix for node class IDs
NODE_CLASS_MAPPINGS = {
    "saykor_latent_selector": SaykorLatentSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "saykor_latent_selector": "(Saykor) Latent Selector (T2I/I2I)",
}
