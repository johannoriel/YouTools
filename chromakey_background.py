#!/usr/bin/env python3

#pip install opencv-python-headless moviepy scikit-learn
import os, sys, time
from moviepy.editor import VideoFileClip, AudioFileClip
import cv2
import numpy as np
import tempfile
import argparse

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
    videos = [file for file in files if file.endswith(('.mp4', '.mkv')) and file != exclude]
    videos.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    if videos:
        return os.path.join(directory, videos[0])
    else:
        return None    
 
# https://colorspire.com/rgb-color-wheel/
def rgb_to_hsv_range(color_rgb, tolerance=10):  # Augmentation de la tolérance pour plus de flexibilité
    color_hsv = cv2.cvtColor(np.uint8([[color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    print(f"Dominant color HSV : {color_hsv}")

    # Ajustement des plages de saturation et de valeur basé sur la couleur détectée
    lower_saturation = max(0, color_hsv[1] - 40)  # 40 de tolérance pour la saturation
    upper_saturation = min(255, color_hsv[1] + 40)
    lower_value = max(0, color_hsv[2] - 40)  # 40 de tolérance pour la valeur
    upper_value = min(255, color_hsv[2] + 40)

    # Calcul des plages de teinte, saturation et valeur
    lower = np.array([max(0, color_hsv[0] - tolerance), lower_saturation, lower_value])
    upper = np.array([min(179, color_hsv[0] + tolerance), upper_saturation, upper_value])
    return lower, upper    

def chroma_key(foreground_path, background_path, output_path, color_to_replace=[0, 255, 0]):
    
    if not foreground_path:
        print("Aucune vidéo admissible trouvée.")
        return
    
    lower_color, upper_color = rgb_to_hsv_range(color_to_replace)
    print(f"Color range HSV : {lower_color}, {upper_color}")
        
    # Chargement des clips vidéo
    background_clip = VideoFileClip(background_path)
    foreground_clip = VideoFileClip(foreground_path)
    
    # Sauvegarde de la piste audio du clip de premier plan
    audio = foreground_clip.audio
    
    # Préparation de l'itérateur pour la vidéo de fond mise en boucle
    background_iterator = iter(background_clip.iter_frames(fps=foreground_clip.fps))
    
    # Configuration du codec et création de la vidéo de sortie
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), foreground_clip.fps, foreground_clip.size)
    i=0
    m= int(foreground_clip.duration * foreground_clip.fps)
    for frame in foreground_clip.iter_frames():
        i+=1
        progress_bar(i,m)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask_inv = cv2.bitwise_not(mask)
        
        try:
            background_frame = next(background_iterator)
        except StopIteration:
            background_iterator = iter(background_clip.iter_frames())
            background_frame = next(background_iterator)
        
        background_bgr_resized = cv2.resize(background_frame, (bgr_frame.shape[1], bgr_frame.shape[0]))
        
        foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)
        background = cv2.bitwise_and(background_bgr_resized, background_bgr_resized, mask=mask)
        
        combined = cv2.add(foreground, background)
        # debug
        #combined = cv2.bitwise_or(foreground, background)
        #combined = foreground
        #combined = background
        #combined = mask
        
        # Convertir en RGB pour la cohérence avec MoviePy
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        
        output_video.write(combined_rgb)
    output_video.release()
    
    # Utilisez un fichier temporaire pour la sortie finale
    _, temp_output_path = tempfile.mkstemp(suffix='.mp4')
    
    # Création d'un clip vidéo sans audio à partir du fichier de sortie
    final_clip_no_audio = VideoFileClip(output_path)
    
    # Ajout de la piste audio au clip vidéo
    final_clip = final_clip_no_audio.set_audio(audio)
    
    # Écriture du clip final avec l'audio sur le disque, utilisant le fichier temporaire
    final_clip.write_videofile(temp_output_path, codec="libx264", audio_codec="aac")
    
    # Nettoyage: Fermez tous les clips pour libérer leurs ressources
    final_clip_no_audio.close()
    final_clip.close()
    audio.close()
    
    # Remplacez la vidéo originale sans audio par la nouvelle vidéo avec audio
    os.replace(temp_output_path, output_path)


import cv2
import numpy as np
from sklearn.cluster import KMeans

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
    
def sample_video_colors(video_path, samples=10):
    cap = cv2.VideoCapture(video_path)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = frames_count // samples
    
    dominant_colors = []
    
    for i in range(0, frames_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            dominant_color = find_dominant_color(frame, k=1)
            if dominant_color is not None:  # Assurez-vous que la couleur dominante n'est pas None
                dominant_colors.append(dominant_color)
    
    cap.release()

    # Gérer le cas où dominant_colors est vide
    if dominant_colors:
        return np.mean(dominant_colors, axis=0)
    else:
        return None  # ou retourner une valeur par défaut
    
def filter_pixels_near_color(image_bgr, target_color_rgb, threshold=30):
    """
    Filtre les pixels d'une image pour ne garder que ceux qui sont proches d'une couleur cible en RGB.
    L'image d'entrée est en BGR (format OpenCV) et est convertie en RGB pour le traitement.
    """
    # Convertir l'image de BGR (OpenCV) en RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Calcul de la distance au carré pour rester cohérent avec le seuil au carré
    distance_squared = np.sum((image_rgb - np.array(target_color_rgb))**2, axis=2)
    mask = distance_squared < (threshold**2)
    filtered_pixels = image_rgb.reshape(-1, 3)[mask.reshape(-1)]
    return filtered_pixels

def find_dominant_color(image_bgr, target_color_rgb=[0, 255, 0], k=1, threshold=100):
    """
    Trouve la couleur dominante (en RGB) dans l'image en utilisant le clustering K-means, en se concentrant sur les pixels proches d'une couleur cible en RGB.
    L'image d'entrée est en BGR (format OpenCV) et est convertie en RGB pour le traitement.
    """
    filtered_pixels = filter_pixels_near_color(image_bgr, target_color_rgb, threshold)
    
    # S'assurer que filtered_pixels est au moins un ensemble de pixels avant de continuer
    if filtered_pixels.shape[0] == 0:
        print("Aucun pixel ne correspond au critère. Augmentez le seuil.")
        return None

    filtered_pixels = np.float32(filtered_pixels)

    # S'assurer que k est valide
    k = min(k, filtered_pixels.shape[0])
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, centers = cv2.kmeans(filtered_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Trouver la couleur dominante
    unique, counts = np.unique(labels, return_counts=True)
    dominant_rgb = centers[unique[np.argmax(counts)]]
    return dominant_rgb

def replace_background(video_file, background, result_file):
    print(f"Chromakey backgroudn replacement : {video_file}, {background}, {result_file}")
    dominant_color = sample_video_colors(video_file, samples=10)
    #dominant_color = [20, 255, 78]
    #dominant_color = [0, 255, 0]
    dominant_color = [0, 128, 0]
    print(f"Couleur dominante trouvée : {dominant_color}")
    chroma_key(video_file, background, result_file, dominant_color)

"""
parser = argparse.ArgumentParser(description='Appliquer un effet chroma key sur une vidéo en utilisant une vidéo de fond spécifiée.')
parser.add_argument('foreground_video', type=str, help='Le chemin vers la vidéo de premier plan à traiter.')
parser.add_argument('background_video', type=str, help='Le chemin vers la vidéo de fond à utiliser.')
args = parser.parse_args()

# Utilisation
result_file = 'video YT.mp4'

# Vérifier si le fichier existe dans le répertoire courant
if os.path.exists(result_file):
    os.remove(result_file)  # Supprimer le fichier
    print(f"Le fichier '{result_file}' a été supprimé.")

video_path = find_latest_video('.')
directory = '.'
foreground_path = find_latest_video(directory)
print(foreground_path)

#args.foreground_video, args.background_video
replace_background(foreground_path, 'background.mp4', result_file)
"""
