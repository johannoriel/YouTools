import random
import numpy as np
from PIL import Image
from moviepy import *
import math

def replace_with_image(main_clip, start_sec, end_sec, image_path, target_size, animation_type="random"):
    """Replace a section of the video with an animated image.

    Args:
        main_clip: Main video clip
        start_sec: Start of the section to replace (seconds)
        end_sec: End of the section to replace (seconds)
        image_path: Path to the image to insert
        target_size: Target size (width, height) of the video
        animation_type: Type of animation ("zoom", "falling", "swinging", "horizontal_bounce", "random")
    """
    duration = end_sec - start_sec
    audio_clip = main_clip.subclipped(start_sec, end_sec).audio

    # Load the image
    img = Image.open(image_path)
    img_w, img_h = img.size
    target_w, target_h = target_size

    # Calculate the scaling factor to fit the image entirely within the target size
    scale_factor = min(target_w / img_w, target_h / img_h)
    base_w = int(img_w * scale_factor)
    base_h = int(img_h * scale_factor)

    # Position "normale" (centrée)
    normal_x = (target_w - base_w) // 2
    normal_y = (target_h - base_h) // 2

    # Select animation if random
    if animation_type == "random":
        animations = ["spinning_mirror"]
        animation_type = random.choice(animations)

    # Define the animation functions
    def zoom_animation(t, progress):
        """Zoom animation (original functionality)"""
        zoom_factor = 1.10
        current_zoom = 1 + (zoom_factor - 1) * progress

        # Random zoom center (avoid edges)
        zoom_x = random.uniform(0.2, 0.8)
        zoom_y = random.uniform(0.2, 0.8)

        new_w = int(base_w * current_zoom)
        new_h = int(base_h * current_zoom)

        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2

        zoomed_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        frame = Image.new("RGB", target_size, (0, 0, 0))
        frame.paste(zoomed_img, (paste_x, paste_y))
        return np.array(frame)

    def falling_animation(t, progress):
        """Image falls from mid-height with acceleration, bounces, then stabilizes"""
        # Start position (mid-height)
        start_y = - target_h   # Mi-hauteur approximative

        # Use same scale as zoom (fully visible, max size without overflow)
        scale_factor = min(target_w / img_w, target_h / img_h)
        final_w = int(img_w * scale_factor)
        final_h = int(img_h * scale_factor)
        paste_x = (target_w - final_w) // 2
        paste_y = (target_h - final_h) // 2  # Final position matches zoom start

        if progress < 0.6:
            # Falling phase (60% of time) with acceleration (quadratic easing)
            fall_progress = progress / 0.6
            # Tomber de start_y vers paste_y (inversion de la direction)
            y_pos = start_y - (start_y - paste_y) * (fall_progress ** 2)  # Chute accélérée
        elif progress < 0.8:
            # Bounce phase (20% of time)
            bounce_progress = (progress - 0.6) / 0.2
            overshoot = 100 * (1 - bounce_progress)
            y_pos = paste_y - overshoot * math.sin(bounce_progress * math.pi)
        else:
            # Stable phase (20% of time)
            y_pos = paste_y

        frame = Image.new("RGB", target_size, (0, 0, 0))
        frame.paste(img.resize((final_w, final_h), Image.Resampling.NEAREST), (paste_x, int(y_pos)))
        return np.array(frame)

    def swinging_animation(t, progress):
        """Image oscillates around its center at final position"""
        # Use same scale as zoom (fully visible, max size without overflow)
        scale_factor = min(target_w / img_w, target_h / img_h)
        final_w = int(img_w * scale_factor)
        final_h = int(img_h * scale_factor)
        center_x = target_w // 2
        center_y = target_h // 2  # Centre de l'écran

        if progress < 0.7:
            # Oscillation phase (70% of time)
            osc_progress = progress / 0.7
            # Oscillation décroissante autour de 0°
            angle = (math.pi / 4) * math.sin(osc_progress * math.pi * 3) * (1 - osc_progress)
        else:
            # Stable phase (30% of time)
            angle = 0

        # Rotate image around its center
        rotated_img = img.resize((final_w, final_h), Image.Resampling.NEAREST).rotate(
            math.degrees(angle), expand=True, resample=Image.Resampling.NEAREST
        )
        rot_w, rot_h = rotated_img.size

        # Position to keep the center fixed at (target_w/2, target_h/2)
        paste_x = center_x - rot_w // 2
        paste_y = center_y - rot_h // 2

        frame = Image.new("RGB", target_size, (0, 0, 0))
        frame.paste(rotated_img, (int(paste_x), int(paste_y)), rotated_img if rotated_img.mode == 'RGBA' else None)
        return np.array(frame)

    def horizontal_bounce_animation(t, progress):
        """Horizontal bounce with 1-2 bounces then stabilizes"""
        # Start position (left of screen)
        start_x = -base_w

        # Use same scale as zoom (fully visible, max size without overflow)
        scale_factor = min(target_w / img_w, target_h / img_h)
        final_w = int(img_w * scale_factor)
        final_h = int(img_h * scale_factor)
        end_x = (target_w - final_w) // 2
        paste_y = (target_h - final_h) // 2  # Final position matches zoom start

        if progress < 0.5:
            # Sliding phase (50% of time)
            x_pos = start_x + (end_x - start_x + 100) * (progress / 0.5)
        elif progress < 0.8:
            # Bounce phase (30% of time)
            bounce_progress = (progress - 0.5) / 0.3
            overshoot = 100 * math.sin(bounce_progress * math.pi * 2) * (1 - bounce_progress)
            x_pos = end_x - overshoot
        else:
            # Stable phase (20% of time)
            x_pos = end_x

        frame = Image.new("RGB", target_size, (0, 0, 0))
        frame.paste(img.resize((final_w, final_h), Image.Resampling.LANCZOS), (int(x_pos), paste_y))
        return np.array(frame)

    def spinning_mirror_animation(t, progress):
        """Image spins with horizontal mirror and flattening effect, 2 oscillations in 1s, then stabilizes"""
        # Use same scale as zoom (fully visible, max size without overflow)
        scale_factor = min(target_w / img_w, target_h / img_h)
        final_w = int(img_w * scale_factor)
        final_h = int(img_h * scale_factor)
        center_x = target_w // 2
        center_y = target_h // 2  # Centre final de l'écran

        # Durée totale de l'animation (en secondes)
        oscillation_duration = 1.0  # 1 seconde pour 2 oscillations
        oscillation_progress = min(t / oscillation_duration, 1.0)

        sine_value = math.cos(oscillation_progress * math.pi * 2)
        abs_sine = abs(sine_value)

        # Appliquer le miroir horizontal si sinus négatif
        base_img = img if sine_value >= 0 else img.transpose(Image.FLIP_LEFT_RIGHT)

        # Aplatissement horizontal
        new_w = int(final_w * (0.1 + 0.9 * abs_sine))  # Largeur varie entre 10% et 100%
        resized_img = base_img.resize((new_w, final_h), Image.Resampling.NEAREST)

        # Position pour centrer l'image
        paste_x = center_x - new_w // 2  # Ajuster pour le centre horizontal après resize
        paste_y = center_y - final_h // 2

        frame = Image.new("RGB", target_size, (0, 0, 0))
        frame.paste(resized_img, (int(paste_x), int(paste_y)), resized_img if resized_img.mode == 'RGBA' else None)
        return np.array(frame)

    # Main frame generator
    def make_frame(t):
        progress = min(t / duration, 1.0)

        if animation_type == "zoom":
            return zoom_animation(t, progress)
        elif animation_type == "falling":
            return falling_animation(t, progress)
        elif animation_type == "horizontal_bounce":
            return horizontal_bounce_animation(t, progress)
        elif animation_type == "swinging":
            return swinging_animation(t, progress)
        elif animation_type == "spinning_mirror":
            return spinning_mirror_animation(t, progress)
        else:
            return zoom_animation(t, progress)  # fallback

    # Create an animated clip
    animated_img_clip = VideoClip(make_frame, duration=duration)

    # Add audio if present
    if audio_clip:
        animated_img_clip = animated_img_clip.with_audio(audio_clip)

    # Concatenate clips
    return concatenate_videoclips([
        main_clip.subclipped(0, start_sec),
        animated_img_clip,
        main_clip.subclipped(end_sec)
    ])
