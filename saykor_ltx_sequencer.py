"""
Saykor LTX Sequencer with IC-LoRA - ComfyUI Custom Node

High-performance LTX video sequencer with integrated IC-LoRA Video guidance.
Bypasses slow timeline managers for raw generation speeds.

Author: Saykor
Category: Saykor/LTX
"""

from comfy_extras.nodes_lt import LTXVAddGuide
import torch
import torch.nn.functional as F

try:
    from comfy_extras.nodes_lt import get_noise_mask
except ImportError:
    def get_noise_mask(latent):
        if isinstance(latent, dict):
            return latent.get("noise_mask")
        return None


class SaykorLTXSequencerWithICLora:
    DESCRIPTION = (
        "High-performance LTX sequencer with integrated IC-LoRA Video guidance.\n"
        "Bypasses slow timeline managers for raw generation speeds."
    )
    CATEGORY = "Saykor/LTX"
    FUNCTION = "execute"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "latent": ("LATENT",),
            "multi_input": ("IMAGE",),
            "num_images": ("INT", {"default": 1, "min": 0, "max": 50, "step": 1}),
            "insert_mode": (["frames", "seconds"], {"default": "frames"}),
            "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1}),
            "strength_sync": ("BOOLEAN", {"default": True}),
            "bypass": ("BOOLEAN", {"default": False}),
            # IC-LoRA video control input
            "control_video": ("IMAGE",),  # Your Canny/Depth/Pose video
            "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "video_start_frame": ("INT", {"default": 0, "min": 0, "max": 9999}),
        }

        optional = {}
        for index in range(1, 51):
            optional[f"insert_frame_{index}"] = ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1})
            optional[f"insert_second_{index}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 0.1})
            optional[f"strength_{index}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        return {"required": required, "optional": optional}

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        multi_input,
        num_images,
        insert_mode,
        frame_rate,
        strength_sync,
        bypass,
        control_video,
        ic_lora_strength,
        video_start_frame,
        **kwargs,
    ):
        if bypass:
            return (positive, negative, latent)

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        # ------------------------------------------------------------------
        # STEP 1: IC-LoRA VIDEO PROCESSING (runs before the sequencer)
        # ------------------------------------------------------------------
        if control_video is not None and ic_lora_strength > 0.0:
            # Resize video to match the latent at pixel level
            pixel_w = latent_width * scale_factors[2]
            pixel_h = latent_height * scale_factors[1]

            # Reference video is in [B, H, W, C]; convert to PyTorch [B, C, H, W]
            video_resized = control_video.permute(0, 3, 1, 2)
            video_resized = F.interpolate(video_resized, size=(pixel_h, pixel_w), mode="bilinear", align_corners=False)
            video_resized = video_resized.permute(0, 2, 3, 1)  # Back to standard Comfy format

            # Trim video frames to match the maximum latent length
            max_pixel_frames = latent_length * scale_factors[0]
            video_cut = video_resized[video_start_frame : video_start_frame + max_pixel_frames]

            # Encode video guidance through VAE
            encoded_video_img, encoded_video_latent = LTXVAddGuide.encode(
                vae, latent_width, latent_height, video_cut, scale_factors
            )

            # Apply video control to prompts
            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive,
                negative,
                0,  # Start from the first latent frame
                latent_image,
                noise_mask,
                encoded_video_latent,
                ic_lora_strength,
                scale_factors,
            )

        # ------------------------------------------------------------------
        # STEP 2: STANDARD KEYFRAME SEQUENCER (Multi-Image)
        # ------------------------------------------------------------------
        batch_size = int(multi_input.shape[0]) if multi_input is not None else 0
        effective_count = max(0, min(int(num_images), batch_size))

        for index in range(1, effective_count + 1):
            image = multi_input[index - 1:index]
            frame_index = None
            if insert_mode == "frames":
                raw_frame_index = kwargs.get(f"insert_frame_{index}")
                if raw_frame_index not in (None, ""):
                    frame_index = int(raw_frame_index)
            else:
                insert_seconds = kwargs.get(f"insert_second_{index}")
                if insert_seconds is not None:
                    frame_index = int(float(insert_seconds) * frame_rate)

            if frame_index is None:
                continue

            raw_strength = kwargs.get(f"strength_{index}", 1.0)
            strength = float(raw_strength)

            if strength <= 0.0:
                continue
            if strength > 1.0:
                strength = 1.0

            encoded_image, encoded_latent = LTXVAddGuide.encode(vae, latent_width, latent_height, image, scale_factors)
            conditioning_frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                positive, latent_length, len(encoded_image), frame_index, scale_factors
            )

            if latent_idx + encoded_latent.shape[2] > latent_length:
                # Skip if out of bounds instead of crashing the entire process
                continue

            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive,
                negative,
                conditioning_frame_idx,
                latent_image,
                noise_mask,
                encoded_latent,
                strength,
                scale_factors,
            )

        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


# Node mappings for ComfyUI registration
# All Saykor nodes use the "saykor_" prefix for node class IDs
NODE_CLASS_MAPPINGS = {
    "saykor_ltx_sequencer_with_ic_lora": SaykorLTXSequencerWithICLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "saykor_ltx_sequencer_with_ic_lora": "(Saykor) LTX Sequencer with IC-LoRA",
}
