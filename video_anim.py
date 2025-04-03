import random
import numpy as np
from moviepy import *
import math

def replace_with_image(main_clip, start_sec, end_sec, image_path, target_size, animation_type="random", background="video"):
    """Replace a section of the video with an animated image using MoviePy effects.

    Args:
        main_clip: Main video clip
        start_sec: Start of the section to replace (seconds)
        end_sec: End of the section to replace (seconds)
        image_path: Path to the image to insert
        target_size: Target size (width, height) of the video
        animation_type: Type of animation ("zoom", "falling", "swinging", "horizontal_bounce", "spinning_mirror", "fade", "random")
        background: Background type ("video" for original video, "green" for green screen)
    """
    duration = end_sec - start_sec
    audio_clip = main_clip.subclipped(start_sec, end_sec).audio

    # Load and prepare the image as a clip
    img_clip = ImageClip(image_path)
    img_w, img_h = img_clip.w, img_clip.h
    target_w, target_h = target_size

    # Calculate scaling factor to fit image within target size
    scale_factor = min(target_w / img_w, target_h / img_h)
    base_w = int(img_w * scale_factor)
    base_h = int(img_h * scale_factor)

    # Base resize and position
    img_clip = img_clip.with_effects([vfx.Resize(width=base_w, height=base_h)])
    normal_x = (target_w - base_w) // 2
    normal_y = (target_h - base_h) // 2

    # Select animation if random
    if animation_type == "random":
        animations = ["zoom", "falling", "swinging", "horizontal_bounce", "spinning_mirror", "fade"]
        animation_type = random.choice(animations)

    # Define background
    if background == "video":
        background_clip = main_clip.subclipped(start_sec, end_sec)
    elif background == "green":
        background_clip = ColorClip(size=target_size, color=(0, 255, 0), duration=duration)
    else:
        background_clip = ColorClip(size=target_size, color=(0, 0, 0), duration=duration)

    # Animation functions using MoviePy effects and transform
    def apply_zoom(clip):
        zoom_x = random.uniform(0.2, 0.8) * base_w
        zoom_y = random.uniform(0.2, 0.8) * base_h
        def zoom_filter(get_frame, t):
            progress = min(t / duration, 1.0)
            zoom_factor = 1 + (1.10 - 1) * progress
            zoomed_clip = clip.with_effects([vfx.Resize(zoom_factor)])
            frame = zoomed_clip.get_frame(t)
            return frame
        return clip.with_position((normal_x, normal_y)).with_duration(duration).transform(zoom_filter)

    def apply_falling(clip):
        def falling_filter(t):
            progress = min(t / duration, 1.0)
            if progress < 0.6:
                fall_progress = progress / 0.6
                y_pos = -target_h + (target_h + normal_y) * (fall_progress ** 2)
            elif progress < 0.8:
                bounce_progress = (progress - 0.6) / 0.2
                overshoot = 100 * (1 - bounce_progress)
                y_pos = normal_y - overshoot * math.sin(bounce_progress * math.pi)
            else:
                y_pos = normal_y
            return (normal_x, y_pos)
        return clip.with_position(falling_filter).with_duration(duration)

    def apply_swinging(clip):
        def swinging_filter(t):
            progress = min(t / duration, 1.0)
            if progress < 0.7:
                osc_progress = progress / 0.7
                angle = math.degrees((math.pi / 4) * math.sin(osc_progress * math.pi * 3) * (1 - osc_progress))
            else:
                angle = 0
            return angle
        return clip.with_mask().with_duration(duration).rotated(swinging_filter, expand=True).with_position(('center', 'center'))

    def apply_horizontal_bounce(clip):
        def bounce_filter(t):
            progress = min(t / duration, 1.0)
            final_w = int(img_w * scale_factor)
            final_h = int(img_h * scale_factor)
            end_x = (target_w - final_w) // 2
            if progress < 0.5:
                x_pos = -base_w + (end_x + base_w + 100) * (progress / 0.5)
            elif progress < 0.8:
                bounce_progress = (progress - 0.5) / 0.3
                overshoot = 100 * math.sin(bounce_progress * math.pi * 2) * (1 - bounce_progress)
                x_pos = normal_x - overshoot
            else:
                x_pos = normal_x
            return (x_pos, normal_y)
        return clip.with_duration(duration).with_position(bounce_filter)

    def apply_spinning_mirror(clip):
        def spinning_filter(get_frame, t):
            progress = min(t / 1.0, 1.0)  # 1s oscillation
            sine_value = math.cos(progress * math.pi * 2)
            scale = 0.1 + 0.9 * abs(sine_value)
            effects = [vfx.Resize((int(base_w * scale), int(base_h)))]
            if sine_value < 0:
                effects.append(vfx.MirrorX())
            return clip.with_effects(effects).get_frame(t)
        return clip.with_duration(duration).transform(spinning_filter).with_position(('center', 'center'))

    def apply_fade(clip):
        return clip.with_position((normal_x, normal_y)).with_duration(duration).with_effects([vfx.FadeIn(0.9)])

    # Apply selected animation
    if animation_type == "zoom":
        animated_clip = apply_zoom(img_clip)
    elif animation_type == "falling":
        animated_clip = apply_falling(img_clip)
    elif animation_type == "swinging":
        animated_clip = apply_swinging(img_clip)
    elif animation_type == "horizontal_bounce":
        animated_clip = apply_horizontal_bounce(img_clip)
    elif animation_type == "spinning_mirror":
        animated_clip = apply_spinning_mirror(img_clip)
    elif animation_type == "fade":
        animated_clip = apply_fade(img_clip)
    else:
        print("Invalid animation type")
        animated_clip = apply_zoom(img_clip)  # Fallback

    # Composite with background
    final_clip = CompositeVideoClip([background_clip, animated_clip])

    # Add audio if present
    if audio_clip:
        final_clip = final_clip.with_audio(audio_clip)

    # Concatenate clips
    return concatenate_videoclips([
        main_clip.subclipped(0, start_sec),
        final_clip,
        main_clip.subclipped(end_sec)
    ])
