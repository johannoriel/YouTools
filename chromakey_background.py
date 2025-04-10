#!/usr/bin/env python3

# pip install opencv-python-headless moviepy scikit-learn
from sklearn.cluster import KMeans
import os
import sys
import time
from moviepy import VideoFileClip, AudioFileClip
import cv2
import numpy as np
import tempfile
import argparse
import logging

#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.CRITICAL + 1)
logger = logging.getLogger(__name__)

def progress_bar(i, total):
    # Calculer le pourcentage terminé
    progress = (i / total) * 100

    # Effacer la ligne précédente
    sys.stdout.write('\r')
    sys.stdout.flush()

    # Afficher la barre de progression
    sys.stdout.write("[%-50s] %.2f%%" % ('=' * int(progress / 2), progress))
    sys.stdout.flush()


def find_latest_video(directory, exclude='output.mp4'):
    files = os.listdir(directory)
    videos = [file for file in files if file.endswith(
        ('.mp4', '.mkv')) and file != exclude]
    videos.sort(key=lambda x: os.path.getmtime(
        os.path.join(directory, x)), reverse=True)
    if videos:
        return os.path.join(directory, videos[0])
    else:
        return None

# https://colorspire.com/rgb-color-wheel/


# Augmentation de la tolérance pour plus de flexibilité
def rgb_to_hsv_range(color_rgb, tolerance_hue=15, tolerance_sat=50, tolerance_val=50, exact_color=False):
    color_hsv = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    if exact_color:
        # Pas de tolérance si couleur exacte
        tolerance_hue = tolerance_sat = tolerance_val = 2  # Petite marge pour bruit minime
    lower = np.array([max(0, color_hsv[0] - tolerance_hue),
                      max(0, color_hsv[1] - tolerance_sat),
                      max(0, color_hsv[2] - tolerance_val)])
    upper = np.array([min(179, color_hsv[0] + tolerance_hue),
                      min(255, color_hsv[1] + tolerance_sat),
                      min(255, color_hsv[2] + tolerance_val)])
    return lower, upper

def suppress_color_spill(frame, mask, target_hue=60, hue_shift=50):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    spill_mask = cv2.inRange(hue, target_hue - 25, target_hue + 25)
    spill_mask = cv2.bitwise_and(spill_mask, cv2.bitwise_not(mask))
    hsv[:, :, 0] = np.where(spill_mask > 0, (hue + hue_shift) % 180, hue)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def chroma_key(foreground_path, background_path, output_path, color_to_replace=[0, 255, 0], exact_color=False):
    if not foreground_path:
        print("Aucune vidéo admissible trouvée.")
        return

    # Charger les clips
    background_clip = VideoFileClip(background_path)
    foreground_clip = VideoFileClip(foreground_path)
    audio = foreground_clip.audio

    # Synchroniser le framerate
    background_iterator = iter(background_clip.iter_frames(fps=foreground_clip.fps))

    # Configurer la sortie vidéo
    output_video = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), foreground_clip.fps, foreground_clip.size
    )

    # Définir la plage HSV
    if exact_color:
        # Utiliser la couleur exacte avec marge minimale
        lower_color, upper_color = rgb_to_hsv_range(color_to_replace, exact_color=True)
    else:
        # Mode robuste avec détection dynamique
        lower_color, upper_color = rgb_to_hsv_range(color_to_replace)

    i = 0
    m = int(foreground_clip.duration * foreground_clip.fps)
    for frame in foreground_clip.iter_frames():
        i += 1
        progress_bar(i, m)

        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        # Créer le masque
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Post-traitement du masque
        if exact_color:
            # Moins de post-traitement car couleur précise
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=1)
        else:
            # Post-traitement robuste pour zones incertaines
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=5)
            mask = cv2.erode(mask, kernel, iterations=2)
            kernel_right = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.uint8)
            mask = cv2.dilate(mask, kernel_right, iterations=3)

        mask_inv = cv2.bitwise_not(mask)

        # Récupérer et redimensionner le background
        try:
            background_frame = next(background_iterator)
        except StopIteration:
            background_iterator = iter(background_clip.iter_frames(fps=foreground_clip.fps))
            background_frame = next(background_iterator)
        background_bgr_resized = cv2.resize(
            background_frame, (bgr_frame.shape[1], bgr_frame.shape[0])
        )
        background_bgr_resized = cv2.cvtColor(background_bgr_resized, cv2.COLOR_RGB2BGR)

        # Supprimer les reflets verts
        frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        frame_rgb = suppress_color_spill(frame_rgb, mask)

        # Appliquer les masques
        foreground = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask_inv.astype(np.uint8))
        background = cv2.bitwise_and(background_bgr_resized, background_bgr_resized, mask=mask.astype(np.uint8))

        # Combiner
        combined = cv2.add(foreground, background)
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        output_video.write(combined_rgb)

    output_video.release()

    # Ajouter l’audio
    _, temp_output_path = tempfile.mkstemp(suffix='.mp4')
    final_clip_no_audio = VideoFileClip(output_path)
    final_clip = final_clip_no_audio.with_audio(audio)
    final_clip.write_videofile(temp_output_path, codec="libx264", audio_codec="aac")

    # Nettoyage
    final_clip_no_audio.close()
    final_clip.close()
    audio.close()
    os.replace(temp_output_path, output_path)


def sample_video_colors2(video_path, samples=10):
    """Échantillonne des frames de la vidéo et trouve la couleur dominante dans chaque frame."""
    cap = cv2.VideoCapture(video_path)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = frames_count // samples

    dominant_colors = []

    for i in range(0, frames_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            dominant_color = find_dominant_color(frame, k=1)
            dominant_colors.append(dominant_color)

    cap.release()
    return np.mean(dominant_colors, axis=0)


def sample_video_colors(video_path, target_color_rgb=[0, 255, 0], samples=10, threshold=30):
    """
    Échantillonne des frames de la vidéo et trouve la moyenne des pixels proches de la couleur cible en RGB.
    """
    cap = cv2.VideoCapture(video_path)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = frames_count // samples

    pixels_near_target = []

    for i in range(0, frames_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            filtered_pixels = filter_pixels_near_color(
                frame, target_color_rgb, threshold)
            pixels_near_target.extend(filtered_pixels)

    cap.release()

    if pixels_near_target:
        return np.mean(pixels_near_target, axis=0)
    else:
        return None


def filter_pixels_near_color(image_bgr, target_color_rgb, threshold=30):
    """
    Filtre les pixels d'une image pour ne garder que ceux qui sont proches d'une couleur cible en RGB.
    L'image d'entrée est en BGR (format OpenCV) et est convertie en RGB pour le traitement.
    """
    # Convertir l'image de BGR (OpenCV) en RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Calcul de la distance au carré pour rester cohérent avec le seuil au carré
    distance_squared = np.sum(
        (image_rgb - np.array(target_color_rgb))**2, axis=2)
    mask = distance_squared < (threshold**2)
    filtered_pixels = image_rgb.reshape(-1, 3)[mask.reshape(-1)]
    return filtered_pixels


def find_dominant_color(image_bgr, target_color_rgb=[0, 255, 0], threshold=30):
    """
    Trouve la couleur moyenne des pixels proches de la couleur cible en RGB dans l'image.
    L'image d'entrée est en BGR (format OpenCV) et est convertie en RGB pour le traitement.
    """
    filtered_pixels = filter_pixels_near_color(
        image_bgr, target_color_rgb, threshold)

    if filtered_pixels.shape[0] == 0:
        print("Aucun pixel ne correspond au critère. Augmentez le seuil.")
        return None

    return np.mean(filtered_pixels, axis=0)


def find_dominant_color_kmeans(image_bgr, k=1):
    """
    Trouve la couleur dominante dans l'image en utilisant KMeans.
    """
    # Convertir l'image en RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Redimensionner l'image pour accélérer le traitement
    pixels = image_rgb.reshape(-1, 3)

    # Appliquer KMeans pour trouver les couleurs dominantes
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Retourner la couleur dominante
    return kmeans.cluster_centers_[0]


def sample_video_colors_kmeans(video_path, samples=10):
    """Échantillonne des frames de la vidéo et trouve la couleur dominante dans chaque frame en utilisant KMeans."""
    cap = cv2.VideoCapture(video_path)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = frames_count // samples

    dominant_colors = []

    for i in range(0, frames_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            dominant_color = find_dominant_color_kmeans(frame, k=1)
            dominant_colors.append(dominant_color)

    cap.release()
    return np.mean(dominant_colors, axis=0)


def replace_background(video_file, background, result_file, target_color_rgb=[0, 255, 0], exact_color=False):
    print(f"Chromakey background replacement: {video_file}, {background}, {result_file}")
    chroma_key(video_file, background, result_file, color_to_replace=target_color_rgb, exact_color=exact_color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Appliquer un effet chroma key.')
    parser.add_argument('foreground_video', type=str, help='Chemin vers la vidéo de premier plan.')
    parser.add_argument('background_video', type=str, help='Chemin vers la vidéo de fond.')
    parser.add_argument('--exact_color', action='store_true', help='Utiliser une couleur exacte sans marge large.')
    args = parser.parse_args()

    result_file = 'video YT.mp4'
    if os.path.exists(result_file):
        os.remove(result_file)
        print(f"Le fichier '{result_file}' a été supprimé.")

    replace_background(args.foreground_video, args.background_video, result_file, exact_color=args.exact_color)

if __name__ == "__main__":
    main()
