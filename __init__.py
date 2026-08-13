# ComfyUI_Saykor_Nodes - Custom nodes for ComfyUI
# Project by Saykor
#
# All node class keys use the "saykor_" prefix for consistent naming.
# To add a new node: create a .py file, then add its import below.

from .saykor_latent_selector import (
    NODE_CLASS_MAPPINGS as _mappings_latent,
    NODE_DISPLAY_NAME_MAPPINGS as _display_latent,
)
from .saykor_ltx_sequencer import (
    NODE_CLASS_MAPPINGS as _mappings_ltx,
    NODE_DISPLAY_NAME_MAPPINGS as _display_ltx,
)
from .saykor_is_image_valid import (
    NODE_CLASS_MAPPINGS as _mappings_is_image_valid,
    NODE_DISPLAY_NAME_MAPPINGS as _display_is_image_valid,
)
from .saykor_safe_image_loader import (
    NODE_CLASS_MAPPINGS as _mappings_safe_image_loader,
    NODE_DISPLAY_NAME_MAPPINGS as _display_safe_image_loader,
)
from .saykor_loadimage_override import (
    NODE_CLASS_MAPPINGS as _mappings_loadimage_override,
    NODE_DISPLAY_NAME_MAPPINGS as _display_loadimage_override,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(_mappings_latent)
NODE_CLASS_MAPPINGS.update(_mappings_ltx)
NODE_CLASS_MAPPINGS.update(_mappings_is_image_valid)
NODE_CLASS_MAPPINGS.update(_mappings_safe_image_loader)
NODE_CLASS_MAPPINGS.update(_mappings_loadimage_override)

NODE_DISPLAY_NAME_MAPPINGS.update(_display_latent)
NODE_DISPLAY_NAME_MAPPINGS.update(_display_ltx)
NODE_DISPLAY_NAME_MAPPINGS.update(_display_is_image_valid)
NODE_DISPLAY_NAME_MAPPINGS.update(_display_safe_image_loader)
NODE_DISPLAY_NAME_MAPPINGS.update(_display_loadimage_override)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
