"""
Is Image Valid ? - ComfyUI Custom Node

Checks whether an image input is valid (not None) and returns
both a BOOLEAN and INT representation for maximum compatibility.

Author: Saykor
Category: Saykor/Images
"""


class SaykorIsImageValid:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
            },
        }

    # Return two output types for maximum compatibility
    RETURN_TYPES = ("BOOLEAN", "INT")
    RETURN_NAMES = ("is_valid", "as_int")
    FUNCTION = "check_is_valid"
    CATEGORY = "Saykor/Images"

    def check_is_valid(self, image=None):
        # image is valid only if it is not None and has length (list of tensors)
        valid = image is not None

        # Return tuple: (True/False, 1/0)
        return (valid, 1 if valid else 0)


# Node mappings for ComfyUI registration
# All Saykor nodes use the "saykor_" prefix for node class IDs
NODE_CLASS_MAPPINGS = {
    "saykor_is_image_valid": SaykorIsImageValid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "saykor_is_image_valid": "(Saykor) Is Image Valid ?",
}
