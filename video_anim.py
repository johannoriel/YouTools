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
        animations = ["zoom", "falling", "swinging", "horizontal_bounce"]
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
        """Image falls from above with bounce effect to normal position"""
        # Start position (above screen)
        start_y = -base_h
        end_y = normal_y

        # Bounce animation (easeOutBounce approximation)
        if progress < 0.7:
            # Falling phase (70% of time)
            y_pos = start_y + (end_y - start_y + 100) * (progress / 0.7)
        else:
            # Bounce phase (30% of time)
            bounce_progress = (progress - 0.7) / 0.3
            overshoot = 100 * (1 - bounce_progress)
            y_pos = end_y - overshoot * math.sin(bounce_progress * math.pi * 2)

        paste_x = normal_x
        paste_y = int(max(0, y_pos))

        frame = Image.new("RGB", target_size, (0, 0, 0))
        frame.paste(img.resize((base_w, base_h), Image.Resampling.LANCZOS), (paste_x, paste_y))
        return np.array(frame)

    def horizontal_bounce_animation(t, progress):
        """Horizontal bouncing animation"""
        # Start position (left of screen)
        start_x = -base_w
        end_x = normal_x

        if progress < 0.7:
            # Sliding phase (70% of time)
            x_pos = start_x + (end_x - start_x + 100) * (progress / 0.7)
        else:
            # Bounce phase (30% of time)
            bounce_progress = (progress - 0.7) / 0.3
            overshoot = 100 * (1 - bounce_progress)
            x_pos = end_x - overshoot * math.sin(bounce_progress * math.pi * 2)

        paste_x = int(max(0, x_pos))
        paste_y = normal_y

        frame = Image.new("RGB", target_size, (0, 0, 0))
        frame.paste(img.resize((base_w, base_h), Image.Resampling.LANCZOS), (paste_x, paste_y))
        return np.array(frame)

    def swinging_animation(t, progress):
        """Pendulum swing from top-left corner"""
        # Anchor point (top-left corner)
        anchor_x = 0
        anchor_y = 0

        # Pendulum length (distance to normal position)
        length_x = normal_x - anchor_x
        length_y = normal_y - anchor_y

        if progress < 0.4:
            # Initial drop (40% of time)
            swing_progress = progress / 0.4
            angle = -math.pi/2 * (1 - swing_progress)
        else:
            # Swinging phase (60% of time)
            swing_progress = (progress - 0.4) / 0.6
            # Damped oscillation
            angle = (math.pi/8) * math.exp(-swing_progress * 3) * math.sin(swing_progress * math.pi * 4)

        # Calculate position along arc
        x_pos = anchor_x + length_x * (1 - math.cos(angle))
        y_pos = anchor_y + length_y * math.sin(abs(angle))

        paste_x = int(max(0, x_pos))
        paste_y = int(max(0, y_pos))

        frame = Image.new("RGB", target_size, (0, 0, 0))
        frame.paste(img.resize((base_w, base_h), Image.Resampling.LANCZOS), (paste_x, paste_y))
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
