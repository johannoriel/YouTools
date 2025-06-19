import os
import pandas as pd
import whisper
from PIL import Image
import io
import cv2
import base64
import ffmpeg
import streamlit as st
from moviepy import *
from pydub import AudioSegment
import random
import numpy as np
import spacy
from pyannote.audio import Pipeline
import yt_dlp
import getpass
import tempfile
import subprocess


def convert_time_to_seconds(time_str):
    # Convert a time string "mm:ss" to seconds
    if isinstance(time_str, str):
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds
    elif isinstance(time_str, int):
        return time_str
    else:
        return 0


def extract_video_section(input_path, output_dir, start_time=None, end_time=None, video_length=None):
    # If no start_time and end_time are provided, return the full video path
    if start_time is None and end_time is None:
        return input_path

    # Convert time from "mm:ss" to seconds
    start_sec = convert_time_to_seconds(start_time) if start_time else 0
    end_sec = convert_time_to_seconds(end_time) if end_time else video_length

    # Extract the video section using moviepy
    new_file_name = os.path.join(
        output_dir, "section_" + os.path.basename(input_path))
    print(f"Extracting section {start_sec} -> {end_sec}...")
    ffmpeg_extract_subclip(input_path, start_sec,
                           end_sec, targetname=new_file_name)
    print(f"Section extracted {new_file_name}")

    # Delete the full downloaded video to save space
    # os.remove(input_path) #debug only

    return new_file_name


def download_video_dlp(url, output_dir, start_time=None, end_time=None):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, 'downloaded_video.%(ext)s')
    }
    # print(ydl_opts)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    new_file_name = extract_video_section(
        filename, output_dir, start_time, end_time)
    return new_file_name, info


def download_audio_with_auth(url, output_dir, cookie_file):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'cookiefile': cookie_file,
        'extract_audio': True,
        'audio_format': 'mp3',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def image_to_base64(image_path):
    """Convertit une image en URL de données Base64."""
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
            mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        print(f"Erreur lors de l'encodage de {image_path}: {e}")
        return None


def scan_videos(directory, extensions, recursive=True):
    def process_video_file(file_path, root_dir):
        """Sous-fonction pour traiter un fichier vidéo individuel"""
        vtt_path = os.path.splitext(file_path)[0] + ".vtt"
        has_subtitles = os.path.exists(vtt_path)
        relative_dir = os.path.relpath(os.path.dirname(file_path), root_dir)

        cap = cv2.VideoCapture(file_path)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / \
            cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
        cap.release()

        return {
            "Video": os.path.basename(file_path),
            "Directory": relative_dir if relative_dir != "." else "",
            "Duration": f"{int(duration // 3600):02d}:{int((duration % 3600) // 60):02d}:{int(duration % 60):02d}",
            "Has Subtitles": has_subtitles,
            "Full Path": file_path
        }

    video_data = []

    if recursive:
        # Mode récursif
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    video_path = os.path.join(root, file)
                    video_data.append(
                        process_video_file(video_path, directory))
    else:
        # Mode non-récursif
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in extensions):
                video_path = os.path.join(directory, file)
                if os.path.isfile(video_path):  # Vérification supplémentaire
                    video_data.append(
                        process_video_file(video_path, directory))

    return pd.DataFrame(video_data)


def load_subtitles_and_chapters(vtt_path):
    subtitles = []
    chapters = []
    if not os.path.exists(vtt_path):
        return pd.DataFrame(columns=["Start", "End", "Text", "Thumbnail"]), pd.DataFrame(columns=["Start", "End", "Title", "Duration"])

    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        chapter_section = False
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "CHAPTERS":
                chapter_section = True
                i += 1
                continue

            # Gestion des sous-titres ou chapitres
            if "-->" in line:
                start, end = line.split(" --> ")
                i += 1  # Passer à la ligne suivante (début du texte/titre)

                # Lire toutes les lignes de texte jusqu'à une ligne vide ou un nouveau timecode
                text_lines = []
                while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                    text_lines.append(lines[i].strip())
                    i += 1

                # Joindre les lignes avec des retours à la ligne
                text = "\n".join(text_lines)

                if not chapter_section:
                    # C'est un sous-titre
                    subtitles.append(
                        {"Start": start, "End": end, "Text": text, "Thumbnail": None})
                else:
                    # C'est un chapitre
                    start_sec = sum(float(x) * 60 ** j for j, x in enumerate(
                        reversed(start.split(":")[:-1]))) + float(start.split(":")[-1])
                    end_sec = sum(float(x) * 60 ** j for j, x in enumerate(
                        reversed(end.split(":")[:-1]))) + float(end.split(":")[-1])
                    duration = end_sec - start_sec
                    chapters.append({
                        "Start": start,
                        "End": end,
                        "Title": text,  # Texte multiligne
                        "Duration": f"{int(duration // 3600):02d}:{int((duration % 3600) // 60):02d}:{int(duration % 60):02d}"
                    })
            else:
                i += 1  # Passer les lignes inutiles (WEBVTT, numéros, etc.)

    return pd.DataFrame(subtitles), pd.DataFrame(chapters)


def save_vtt(vtt_path, subtitles_df, chapters_df):
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, row in subtitles_df.iterrows():
            f.write(f"{i + 1}\n")
            f.write(f"{row['Start']} --> {row['End']}\n")
            f.write(f"{row['Text']}\n\n")
        if not chapters_df.empty:
            f.write("CHAPTERS\n\n")
            for _, row in chapters_df.iterrows():
                f.write(f"{row['Start']} --> {row['End']}\n")
                f.write(f"{row['Title']}\n\n")


def generate_subtitles(video_path, model_name):
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path)
    vtt_path = os.path.splitext(video_path)[0] + ".vtt"
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, segment in enumerate(result["segments"]):
            start = format_time(segment["start"])
            end = format_time(segment["end"])
            text = segment["text"]
            f.write(f"{i + 1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    return vtt_path


def convert_to_mp4(video_path):
    output_path = os.path.splitext(video_path)[0] + ".mp4"
    if os.path.exists(output_path):
        os.remove(output_path)  # Écraser le fichier existant
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream, output_path, vcodec="h264", acodec="aac", strict="experimental")
        ffmpeg.run(stream)
        st.success(f"Converted {os.path.basename(video_path)} to MP4!")
    except Exception as e:
        st.error(f"Conversion failed: {str(e)}")
    return output_path


def rename_video(old_path, new_name):
    """
    Renomme un fichier vidéo et tous les fichiers associés (même nom de base, différentes extensions)

    Args:
        old_path (str): Chemin complet du fichier original
        new_name (str): Nouveau nom de fichier (sans extension)
    """
    import os
    import glob

    # Récupérer le répertoire et l'ancien nom de base
    directory = os.path.dirname(old_path)
    old_base = os.path.splitext(os.path.basename(old_path))[0]

    # Trouver tous les fichiers avec le même nom de base
    pattern = os.path.join(directory, f"{old_base}.*")
    matching_files = glob.glob(pattern)

    # Renommer chaque fichier trouvé
    for file_path in matching_files:
        # Garder la même extension
        ext = os.path.splitext(file_path)[1]
        new_path = os.path.join(directory, f"{new_name}{ext}")

        # Renommer le fichier
        try:
            os.rename(file_path, new_path)
        except Exception as e:
            raise Exception(
                f"Failed to rename {file_path} to {new_path}: {str(e)}")


def merge_videos(video_paths, output_dir):
    """
    Fusionne plusieurs vidéos en une seule avec un pré-encodage individuel pour garantir la compatibilité.

    Args:
        video_paths (list): Liste des chemins des fichiers vidéo à fusionner.
        output_dir (str): Répertoire de sortie pour la vidéo fusionnée.

    Returns:
        str: Chemin de la vidéo fusionnée.
    """
    import os
    import ffmpeg
    import streamlit as st
    from tempfile import NamedTemporaryFile

    output_path = os.path.join(output_dir, "merge.mp4")
    if os.path.exists(output_path):
        os.remove(output_path)  # Écraser le fichier existant

    try:
        # Étape 1 : Pré-encoder chaque vidéo dans un format standardisé
        temp_files = []
        for video_path in video_paths:
            # Créer un fichier temporaire pour la vidéo pré-encodée
            with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_path = temp_file.name
                temp_files.append(temp_path)

                # Ré-encodage avec paramètres stricts
                stream = ffmpeg.input(video_path)
                stream = ffmpeg.output(
                    stream,
                    temp_path,
                    vcodec="libx264",      # Codec vidéo H.264 avec la bibliothèque libx264
                    preset="medium",       # Preset équilibré pour qualité/vitesse
                    # Qualité raisonnable (0-51, plus bas = meilleure qualité)
                    crf=23,
                    acodec="aac",          # Codec audio AAC
                    ar=44100,              # Fréquence d'échantillonnage audio standardisée
                    ac=2,                  # 2 canaux audio (stéréo)
                    # Framerate fixé à 30 fps (ajustable si besoin)
                    r=30,
                    strict="experimental",
                    map_metadata="-1",     # Supprimer les métadonnées héritées
                    movflags="faststart"   # Optimisation pour le streaming
                )
                ffmpeg.run(stream, overwrite_output=True)
                st.info(
                    f"Pré-encodage terminé pour {os.path.basename(video_path)}")

        # Étape 2 : Créer un fichier de concaténation
        concat_file_path = os.path.join(output_dir, "concat_list.txt")
        with open(concat_file_path, "w") as f:
            for temp_path in temp_files:
                f.write(f"file '{os.path.abspath(temp_path)}'\n")

        # Étape 3 : Fusionner les vidéos pré-encodées
        stream = ffmpeg.input(concat_file_path, format='concat', safe=0)
        stream = ffmpeg.output(
            stream,
            output_path,
            vcodec="libx264",      # Ré-encodage final en H.264
            acodec="aac",
            ar=44100,              # Assurer une fréquence audio cohérente
            ac=2,                  # Stéréo
            r=30,                  # Framerate cohérent
            preset="medium",
            crf=23,
            strict="experimental",
            map_metadata="-1",
            movflags="faststart"
        )
        ffmpeg.run(stream, overwrite_output=True)

        # Nettoyage des fichiers temporaires
        os.remove(concat_file_path)
        for temp_path in temp_files:
            os.remove(temp_path)

        st.success(f"Videos merged into {output_path}!")
        return output_path

    except Exception as e:
        st.error(f"Merge failed: {str(e)}")
        print(f"Error details: {str(e)}")  # Pour debug
        # Nettoyer les fichiers temporaires en cas d'échec
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return None


def split_video_fast(video_path, split_time_ms, output_dir):
    """Découpe une vidéo en deux parties à un point donné en millisecondes."""
    split1_path = os.path.join(output_dir, "split1.mp4")
    split2_path = os.path.join(output_dir, "split2.mp4")
    try:
        # Convertir millisecondes en secondes avec précision pour FFmpeg
        split_time = split_time_ms / 1000.0
        stream1 = ffmpeg.input(video_path).output(
            split1_path, t=split_time, vcodec="copy", acodec="copy")
        ffmpeg.run(stream1)
        stream2 = ffmpeg.input(video_path, ss=split_time).output(
            split2_path, vcodec="copy", acodec="copy")
        ffmpeg.run(stream2)
        st.success(f"Video split into {split1_path} and {split2_path}!")
    except Exception as e:
        st.error(f"Split failed: {str(e)}")
    return split1_path, split2_path


def split_video(video_path, output_dir, split_time_ms):
    """Découpe une vidéo en deux parties à un point donné en millisecondes avec ré-encodage."""
    split1_path = os.path.join(output_dir, "split1.mp4")
    split2_path = os.path.join(output_dir, "split2.mp4")
    try:
        # Convertir millisecondes en secondes avec précision
        split_time = split_time_ms / 1000.0

        # Première partie : de 0 à split_time
        stream1 = ffmpeg.input(video_path, ss=0).output(
            split1_path,
            t=split_time,
            vcodec="h264",  # Ré-encoder en H.264
            acodec="aac",   # Ré-encoder en AAC
            strict="experimental",
            map_metadata="-1",  # Supprimer les métadonnées héritées
            reset_timestamps=1  # Réinitialiser les timestamps
        )
        ffmpeg.run(stream1)

        # Deuxième partie : de split_time à la fin
        stream2 = ffmpeg.input(video_path, ss=split_time).output(
            split2_path,
            vcodec="h264",
            acodec="aac",
            strict="experimental",
            map_metadata="-1",
            reset_timestamps=1
        )
        ffmpeg.run(stream2)

        st.success(f"Video split into {split1_path} and {split2_path}!")
    except Exception as e:
        st.error(f"Split failed: {str(e)}")
    return split1_path, split2_path


def split_by_chapters(video_path, output_dir, chapters_df=None):
    """Découpe une vidéo en segments basés sur les chapitres ou un seul point de découpe.

    Args:
        video_path (str): Chemin de la vidéo source.
        output_dir (str): Répertoire de sortie pour les fichiers découpés.
        chapters_df (pd.DataFrame, optional): DataFrame contenant Start, End, et Title des chapitres.

    Returns:
        list: Liste des chemins des fichiers générés.
    """
    output_files = []
    try:
        for i, row in chapters_df.iterrows():
            start_time = parse_timecode_to_ms(
                row["Start"]) / 1000.0  # Convertir en secondes
            end_time = parse_timecode_to_ms(row["End"]) / 1000.0
            duration = end_time - start_time
            # Nom du fichier : "n - titre.mp4" (première ligne du titre seulement)
            title = row["Title"].split("\n")[0].replace(
                ":", "-").replace("/", "-")  # Remplacer caractères interdits
            output_path = os.path.join(output_dir, f"{i + 1} - {title}.mp4")

            stream = ffmpeg.input(video_path, ss=start_time).output(
                output_path, t=duration, vcodec="h264", acodec="aac", strict="experimental",
                map_metadata="-1", reset_timestamps=1
            )
            ffmpeg.run(stream)
            output_files.append(output_path)
        st.success(f"Video split into {len(output_files)} chapters!")
    except Exception as e:
        st.error(f"Split failed: {str(e)}")
        return []

    return output_files


def delete_videos(video_paths):
    for path in video_paths:
        vtt_path = os.path.splitext(path)[0] + ".vtt"
        try:
            print(f"Deleting {path}...")
            os.remove(path)
            if os.path.exists(vtt_path):
                os.remove(vtt_path)
            st.success(f"Deleted {os.path.basename(path)}!")
        except Exception as e:
            st.error(f"Deletion failed for {os.path.basename(path)}: {str(e)}")


def generate_thumbnail(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    # timestamp en millisecondes maintenant
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((100, 56), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        thumbnail_bytes = buf.getvalue()
        cap.release()
        return f"data:image/png;base64,{base64.b64encode(thumbnail_bytes).decode('utf-8')}"
    cap.release()
    return None


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_timecode_to_ms(timecode):
    """Convertit un timecode HH:MM:SS.mmm en millisecondes."""
    try:
        parts = timecode.split(":")
        if len(parts) != 3:
            raise ValueError(f"Timecode invalide : {timecode}")

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_part = parts[2]  # "SS.mmm"

        # Séparer secondes et millisecondes
        if "." in seconds_part:
            seconds, millis = seconds_part.split(".")
            seconds = int(seconds)
            # Remplir avec des zéros si millisecondes < 3 chiffres
            millis = int(millis.ljust(3, "0")[:3])
        else:
            seconds = int(seconds_part)
            millis = 0

        total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + millis
        return total_ms
    except Exception as e:
        st.error(
            f"Erreur lors de la conversion du timecode {timecode} : {str(e)}")
        return 0


def insert_video(main_clip, start_sec, video_path_insert, target_size):
    """Insère une vidéo à une position donnée."""
    insert_clip = VideoFileClip(video_path_insert).resized(target_size)
    duration_change = insert_clip.duration
    new_clip = concatenate_videoclips([
        main_clip.subclipped(0, start_sec),
        insert_clip,
        main_clip.subclipped(start_sec)
    ])
    return new_clip, duration_change


def calculate_target_size(source_size, target_size):
    """
    Calcule la taille cible en conservant l'aspect ratio de la source
    pour qu'elle s'adapte au maximum dans la taille cible.

    Args:
        source_size: Tuple (width, height) de la vidéo source
        target_size: Tuple (width, height) de la zone cible

    Returns:
        Tuple (new_width, new_height) de la taille calculée
    """
    source_width, source_height = source_size  # Déjà un tuple, pas besoin de .size
    target_width, target_height = target_size

    source_ratio = source_width / source_height
    target_ratio = target_width / target_height

    if source_ratio > target_ratio:
        # La source est plus large que la cible, on ajuste selon la largeur
        new_width = target_width
        new_height = int(target_width / source_ratio)
    else:
        # La source est plus haute que la cible, on ajuste selon la hauteur
        new_height = target_height
        new_width = int(target_height * source_ratio)

    return (new_width, new_height)


def replace_with_video(main_clip, start_sec, end_sec, video_path_replace, target_size, background="video"):
    """Remplace une section par une autre vidéo en conservant l'aspect ratio.

    Args:
        main_clip: Clip vidéo principal
        start_sec: Début de la section à remplacer (secondes)
        end_sec: Fin de la section à remplacer (secondes)
        video_path_replace: Chemin de la vidéo de remplacement
        target_size: Taille cible (width, height)
        background: Type de fond ("video" pour la vidéo originale, "green" pour fond vert)

    Returns:
        Tuple: (nouveau clip, changement de durée)
    """
    original_duration = end_sec - start_sec
    replace_clip = VideoFileClip(video_path_replace)
    duration_change = replace_clip.duration - original_duration

    # Calcul de la nouvelle taille en conservant l'aspect ratio
    new_size = calculate_target_size(replace_clip.size, target_size)

    # Préparation du fond
    if background == "video":
        background_clip = main_clip.subclipped(start_sec, end_sec)
    elif background == "green":
        background_clip = ColorClip(size=target_size, color=(
            0, 255, 0), duration=min(replace_clip.duration, original_duration))
    else:  # par défaut fond noir
        background_clip = ColorClip(size=target_size, color=(
            0, 0, 0), duration=min(replace_clip.duration, original_duration))

    # Redimensionnement et positionnement avec API MoviePy 2.0
    resized_clip = replace_clip.with_effects(
        [vfx.Resize(width=new_size[0], height=new_size[1])])
    x_center = (target_size[0] - new_size[0]) // 2
    y_center = (target_size[1] - new_size[1]) // 2

    # Composition de la section remplacée
    if replace_clip.duration > original_duration:
        replace_clip = CompositeVideoClip([
            background_clip,
            resized_clip.subclipped(0, original_duration).with_position(
                (x_center, y_center))
        ])
    else:
        replace_clip = CompositeVideoClip([
            background_clip,
            resized_clip.with_position((x_center, y_center))
        ])

    # Construction du clip final
    clips = [main_clip.subclipped(0, start_sec), replace_clip]

    # Ajouter le reste de la vidéo originale si la vidéo de remplacement est plus courte
    if replace_clip.duration < original_duration:
        remaining_duration = original_duration - replace_clip.duration
        filler_clip = main_clip.subclipped(
            end_sec - remaining_duration, end_sec)
        clips.append(filler_clip)

    clips.append(main_clip.subclipped(end_sec))

    new_clip = concatenate_videoclips(clips)
    return new_clip, duration_change


def replace_video_keep_audio(main_clip, start_sec, end_sec, video_path_replace, target_size, background="video"):
    """Remplace une section par une vidéo en conservant l'audio original, avec option de fond.

    Args:
        main_clip: Clip vidéo principal
        start_sec: Début de la section à remplacer (secondes)
        end_sec: Fin de la section à remplacer (secondes)
        video_path_replace: Chemin de la vidéo de remplacement
        target_size: Taille cible (width, height)
        background: Type de fond ("video" pour la vidéo originale, "green" pour fond vert)
    """
    duration = end_sec - start_sec
    replace_clip = VideoFileClip(video_path_replace)
    original_audio = main_clip.subclipped(start_sec, end_sec).audio

    if replace_clip.duration > duration:
        replace_clip = replace_clip.subclipped(0, duration)

    # Calcul de la nouvelle taille en conservant l'aspect ratio
    new_size = calculate_target_size(replace_clip.size, target_size)

    # Préparation du fond
    if background == "video":
        background_clip = main_clip.subclipped(start_sec, end_sec)
    elif background == "green":
        background_clip = ColorClip(size=target_size, color=(
            0, 255, 0), duration=replace_clip.duration)
    else:  # par défaut fond noir (comportement original)
        background_clip = ColorClip(size=target_size, color=(
            0, 0, 0), duration=replace_clip.duration)

    # Redimensionnement et positionnement avec API MoviePy 2.0
    resized_clip = replace_clip.with_effects(
        [vfx.Resize(width=new_size[0], height=new_size[1])])
    x_center = (target_size[0] - new_size[0]) // 2
    y_center = (target_size[1] - new_size[1]) // 2

    # Composition finale
    replace_clip = CompositeVideoClip([
        background_clip,
        resized_clip.with_position((x_center, y_center))
    ]).with_audio(original_audio)

    # Construction du clip final
    clips = [main_clip.subclipped(0, start_sec), replace_clip]
    if replace_clip.duration < duration:
        filler_start = start_sec + replace_clip.duration
        filler_clip = main_clip.subclipped(filler_start, end_sec)
        clips.append(filler_clip)

    clips.append(main_clip.subclipped(end_sec))
    return concatenate_videoclips(clips)


# Dans video_utils.py

def add_animated_text(main_clip, start_sec, end_sec, text, animation_type, anim_duration_sec, target_size, font, font_size, use_green_background=True, position="center", text_style="outline"):
    """Ajoute du texte animé sur une section de la vidéo avec style ajustable."""
    duration = end_sec - start_sec
    audio_clip = main_clip.subclipped(start_sec, end_sec).audio

    # Création du texte principal (blanc avec contour si text_style="outline")
    text_content = text.replace("\\", "\n")
    txt_clip = TextClip(
        text=text_content,  # Explicitement passer le texte
        font=f"{font}",
        font_size=font_size,
        color="white",
        method="caption",
        # Limite la largeur à 80% de la vidéo
        size=(int(target_size[0] * 0.8), None),
        # Contour noir pour "outline"
        stroke_color="#008CCF" if text_style == "outline" else None,
        stroke_width=4 if text_style == "outline" else 0,  # Épaisseur du contour
    ).with_duration(duration)

    # Boîte noire (uniquement pour text_style="box")
    text_padding = 10
    if text_style == "box":
        text_box = ColorClip(
            size=(txt_clip.w + 2 * text_padding,
                  txt_clip.h + 2 * text_padding),
            color=(0, 0, 0),
            duration=duration
        )
    else:
        text_box = None

    # Animation et positionnement
    if animation_type == "fromLeft":
        def position_function(t):
            if t < anim_duration_sec:
                x = -txt_clip.w + \
                    (target_size[0] / 2 + txt_clip.w / 2) * \
                    (t / anim_duration_sec)
            else:
                x = (target_size[0] - txt_clip.w) / 2  # Centré horizontalement

            # Ajuster la position verticale selon le paramètre 'position'
            if position == "center":
                y = (target_size[1] - txt_clip.h) / 2  # Milieu de l'écran
            elif position == "bottom":
                # Position en bas, avec une marge de 20 pixels
                y = target_size[1] - txt_clip.h - 20
            return (x, y)

        txt_clip = txt_clip.with_position(position_function)
        if text_box:
            text_box = text_box.with_position(
                lambda t: (position_function(
                    t)[0] - text_padding, position_function(t)[1] - text_padding)
            )
    else:
        # Sans animation, position fixe
        x = (target_size[0] - txt_clip.w) / 2  # Centré horizontalement
        if position == "center":
            y = (target_size[1] - txt_clip.h) / 2  # Milieu de l'écran
        elif position == "bottom":
            # Position en bas, avec une marge de 20 pixels
            y = target_size[1] - txt_clip.h - 20

        txt_clip = txt_clip.with_position((x, y))
        if text_box:
            text_box = text_box.with_position(
                (x - text_padding, y - text_padding))

    # Composition finale
    if use_green_background:
        background = ColorClip(size=target_size, color=(
            0, 255, 0), duration=duration)
        if text_style == "box":
            final_clip = CompositeVideoClip(
                [background, text_box, txt_clip], size=target_size)
        else:
            final_clip = CompositeVideoClip(
                [background, txt_clip], size=target_size)
    else:
        original_segment = main_clip.subclipped(start_sec, end_sec)
        if text_style == "box":
            final_clip = CompositeVideoClip(
                [original_segment, text_box, txt_clip], size=target_size)
        else:
            final_clip = CompositeVideoClip(
                [original_segment, txt_clip], size=target_size)

    if audio_clip:
        final_clip = final_clip.with_audio(audio_clip)

    return concatenate_videoclips([
        main_clip.subclipped(0, start_sec),
        final_clip,
        main_clip.subclipped(end_sec)
    ])

# Dans video_utils.py

# Dans video_utils.py


def insert_video_with_text(main_clip, start_sec, video_path_insert, text, target_size, font, font_size, use_green_background=True, text_style="outline"):
    """Insère une vidéo avec du texte statique en bas, la durée du texte correspondant à celle de la vidéo insérée."""
    insert_clip = VideoFileClip(video_path_insert).resized(target_size)
    duration = insert_clip.duration
    end_sec = start_sec + duration

    # Création du texte principal (blanc avec contour si text_style="outline")
    text_content = text.replace("\\", "\n")
    txt_clip = TextClip(
        text=text_content,
        font=f"{font}",
        font_size=font_size,
        color="white",
        method="caption",
        # Limite la largeur à 80% de la vidéo
        size=(int(target_size[0] * 0.8), None),
        # Contour noir pour "outline"
        stroke_color="#008CCF" if text_style == "outline" else None,
        stroke_width=4 if text_style == "outline" else 0,  # Épaisseur du contour
    ).with_duration(duration)

    # Boîte noire (uniquement pour text_style="box")
    text_padding = 10
    if text_style == "box":
        text_box = ColorClip(
            size=(txt_clip.w + 2 * text_padding,
                  txt_clip.h + 2 * text_padding),
            color=(0, 0, 0),
            duration=duration
        )
    else:
        text_box = None

    # Positionnement statique en bas
    x = (target_size[0] - txt_clip.w) / 2  # Centré horizontalement
    # Position en bas, avec une marge de 20 pixels
    y = target_size[1] - txt_clip.h - 20

    # Appliquer la position au clip texte
    txt_clip = txt_clip.with_position((x, y))
    if text_box:
        text_box = text_box.with_position((x - text_padding, y - text_padding))

    # Composition de la vidéo insérée avec texte
    if text_style == "box":
        final_insert_clip = CompositeVideoClip(
            [insert_clip, text_box, txt_clip], size=target_size)
    else:
        final_insert_clip = CompositeVideoClip(
            [insert_clip, txt_clip], size=target_size)

    # Concaténation avec la vidéo principale
    new_clip = concatenate_videoclips([
        main_clip.subclipped(0, start_sec),
        final_insert_clip,
        main_clip.subclipped(start_sec)
    ])
    return new_clip, duration


def remove_section(main_clip, start_sec, end_sec):
    """Supprime une section de la vidéo entre start_sec et end_sec."""
    duration_change = end_sec - start_sec
    new_clip = concatenate_videoclips([
        main_clip.subclipped(0, start_sec),
        main_clip.subclipped(end_sec)
    ])
    # Retourne la nouvelle vidéo et la différence de durée (négative car suppression)
    return new_clip, -duration_change


def normalize_audio(video_path, reference_audio_path, make_backup=True):
    """Normalise le son d'une vidéo en utilisant un fichier audio de référence."""
    import os
    import streamlit as st

    # Configurer le chemin vers ffmpeg (ajustez selon votre système)
    # Vérifiez ce chemin sur votre système
    AudioSegment.converter = "/usr/bin/ffmpeg"

    try:
        # Charger le fichier de référence
        reference_audio = AudioSegment.from_file(reference_audio_path)
        target_dBFS = reference_audio.dBFS  # Niveau sonore cible

        # Charger l'audio de la vidéo
        audio = AudioSegment.from_file(video_path)

        # Calculer la différence de volume
        difference = target_dBFS - audio.dBFS

        # Appliquer le gain pour normaliser
        normalized_audio = audio + difference

        # Exporter l'audio normalisé temporairement
        temp_audio_path = os.path.splitext(video_path)[0] + "_temp_audio.mp3"
        normalized_audio.export(temp_audio_path, format="mp3")

        # Renommer l'ancienne vidéo en backup
        backup_path = os.path.splitext(video_path)[0] + "_backup.mp4"
        if os.path.exists(backup_path):
            os.remove(backup_path)  # Supprimer un ancien backup s'il existe
        os.rename(video_path, backup_path)

        # Recomposer la vidéo avec l'audio normalisé
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{backup_path}" -i "{temp_audio_path}" '
            f'-c:v copy -map 0:v:0 -map 1:a:0 "{video_path}" -y'
        )
        os.system(ffmpeg_cmd)

        # Supprimer le fichier temporaire
        os.remove(temp_audio_path)
        if not make_backup:
            os.remove(backup_path)

        st.success(f"Audio normalized for {os.path.basename(video_path)}!")
    except Exception as e:
        st.error(
            f"Audio normalization failed for {os.path.basename(video_path)}: {str(e)}")

def normalize_full_audio(video_path, reference_audio_path, make_backup=True):
    """Normalise dynamiquement le son d'une vidéo avec un compresseur en utilisant un fichier audio de référence."""
    import os
    import streamlit as st
    from pydub import AudioSegment
    import subprocess

    # Configurer le chemin vers ffmpeg
    AudioSegment.converter = "/usr/bin/ffmpeg"

    # Initialiser les variables pour le nettoyage
    temp_audio_path = None
    compressed_audio_path = None
    backup_path = None

    try:
        # Vérifier si ffmpeg est accessible
        if not os.path.exists(AudioSegment.converter):
            raise FileNotFoundError(f"FFmpeg not found at {AudioSegment.converter}")

        # Vérifier si les fichiers d'entrée existent
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path}")

        # Charger le fichier de référence pour obtenir le niveau cible
        reference_audio = AudioSegment.from_file(reference_audio_path)
        target_dBFS = reference_audio.dBFS  # Niveau sonore cible

        # Extraire l'audio de la vidéo
        temp_audio_path = os.path.splitext(video_path)[0] + "_temp_audio.mp3"
        ffmpeg_cmd_extract = [
            AudioSegment.converter, '-y',
            '-i', video_path,
            '-vn', '-acodec', 'mp3',
            temp_audio_path
        ]
        print(f"Executing FFmpeg extract command: {' '.join(ffmpeg_cmd_extract)}")
        result = subprocess.run(ffmpeg_cmd_extract, capture_output=True, text=True, check=True)
        print(f"FFmpeg extract stdout: {result.stdout}")
        print(f"FFmpeg extract stderr: {result.stderr}")

        # Vérifier si le fichier temporaire a été créé
        if not os.path.exists(temp_audio_path):
            raise FileNotFoundError(f"Temporary audio file not created: {temp_audio_path}")

        # Charger l'audio extrait pour analyser son niveau
        audio = AudioSegment.from_file(temp_audio_path)
        input_dBFS = audio.dBFS
        print(f"Target dBFS: {target_dBFS}, Input dBFS: {input_dBFS}")

        # Calculer le makeup gain pour aligner sur le niveau cible
        makeup_gain = target_dBFS - input_dBFS
        print(f"Calculated makeup gain (before clamping): {makeup_gain}")

        # Clamper makeup_gain dans la plage valide [1, 64] pour acompressor
        makeup_gain = max(1, min(64, makeup_gain))
        print(f"Clamped makeup gain: {makeup_gain}")

        # Appliquer le compresseur avec ffmpeg
        compressed_audio_path = os.path.splitext(video_path)[0] + "_compressed_audio.mp3"
        ffmpeg_cmd_compress = [
            AudioSegment.converter, '-y',
            '-i', temp_audio_path,
            '-filter:a', f'acompressor=threshold=-30dB:ratio=4:attack=20:release=200:makeup={makeup_gain}',
            '-c:a', 'mp3',
            compressed_audio_path
        ]
        print(f"Executing FFmpeg compress command: {' '.join(ffmpeg_cmd_compress)}")
        result = subprocess.run(ffmpeg_cmd_compress, capture_output=True, text=True, check=True)
        print(f"FFmpeg compress stdout: {result.stdout}")
        print(f"FFmpeg compress stderr: {result.stderr}")

        # Vérifier si le fichier compressé a été créé
        if not os.path.exists(compressed_audio_path):
            raise FileNotFoundError(f"Compressed audio file not created: {compressed_audio_path}")

        # Renommer l'ancienne vidéo en backup
        backup_path = os.path.splitext(video_path)[0] + "_backup.mp4"
        if os.path.exists(backup_path):
            os.remove(backup_path)  # Supprimer un ancien backup s'il existe
        os.rename(video_path, backup_path)

        # Recomposer la vidéo avec l'audio compressé
        ffmpeg_cmd_recompose = [
            AudioSegment.converter, '-y',
            '-i', backup_path,
            '-i', compressed_audio_path,
            '-c:v', 'copy',
            '-map', '0:v:0', '-map', '1:a:0',
            video_path
        ]
        print(f"Executing FFmpeg recompose command: {' '.join(ffmpeg_cmd_recompose)}")
        result = subprocess.run(ffmpeg_cmd_recompose, capture_output=True, text=True, check=True)
        print(f"FFmpeg recompose stdout: {result.stdout}")
        print(f"FFmpeg recompose stderr: {result.stderr}")

        # Supprimer les fichiers temporaires (sauf backup_path si make_backup=True)
        for temp_file in [temp_audio_path, compressed_audio_path]:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
        if not make_backup and backup_path and os.path.exists(backup_path):
            os.remove(backup_path)

        st.success(f"Audio dynamically normalized for {os.path.basename(video_path)}!")
    except Exception as e:
        st.error(f"Dynamic audio normalization failed for {os.path.basename(video_path)}: {str(e)}")
        raise
    finally:
        # Nettoyage des fichiers temporaires
        for temp_file in [temp_audio_path, compressed_audio_path]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Failed to remove {temp_file}: {str(e)}")
        # Supprimer backup_path uniquement si make_backup=False
        if not make_backup and backup_path and os.path.exists(backup_path):
            try:
                os.remove(backup_path)
            except Exception as e:
                print(f"Failed to remove {backup_path}: {str(e)}")

def replace_audio(main_clip, start_sec, end_sec, audio_path, target_size):
    """Remplace l'audio d'une section par un nouvel audio, ajustant la vitesse de la vidéo si nécessaire.

    Args:
        main_clip: Clip vidéo principal
        start_sec: Début de la section (secondes)
        end_sec: Fin de la section (secondes)
        audio_path: Chemin du fichier audio (.mp3, .ogg, .wav)
        target_size: Taille cible (width, height)

    Returns:
        Tuple: (nouveau clip, changement de durée)
    """
    original_duration = end_sec - start_sec
    audio_clip = AudioFileClip(audio_path)
    new_audio_duration = audio_clip.duration

    # Calculer le facteur de vitesse pour ajuster la vidéo à la durée de l'audio
    speed_factor = original_duration / \
        new_audio_duration if new_audio_duration != 0 else 1.0

    # Extraire la section à modifier
    section_clip = main_clip.subclipped(start_sec, end_sec)

    # Ajuster la vitesse de la vidéo
    if speed_factor != 1.0:
        section_clip = section_clip.fx(vfx.speedx, speed_factor)

    # Appliquer le nouvel audio
    section_clip = section_clip.with_audio(audio_clip)

    # Construire le clip final
    clips = [
        main_clip.subclipped(0, start_sec),
        section_clip,
        main_clip.subclipped(end_sec)
    ]
    new_clip = concatenate_videoclips(clips)

    # Calculer le changement de durée
    duration_change = new_audio_duration - original_duration

    return new_clip, duration_change


def insert_audio(main_clip, start_sec, audio_path, target_size):
    """Insère un audio en créant une vidéo statique à partir de l'image au timecode de départ.

    Args:
        main_clip: Clip vidéo principal (VideoFileClip)
        start_sec: Point d'insertion (secondes)
        audio_path: Chemin du fichier audio (.mp3, .ogg, .wav)
        target_size: Taille cible (width, height)

    Returns:
        Tuple: (nouveau clip, durée de l'audio inséré)

    Raises:
        ValueError: Si main_clip est None, start_sec invalide, ou audio_path incorrect.
    """

    # Vérifier main_clip
    if main_clip is None or not hasattr(main_clip, 'get_frame'):
        raise ValueError("main_clip is None or invalid")

    # Vérifier start_sec
    if start_sec < 0 or start_sec > main_clip.duration:
        raise ValueError(
            f"start_sec ({start_sec}) is out of bounds for clip duration ({main_clip.duration})")

    audio_clip = None
    static_clip = None
    # Charger l'audio
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration

    sample = audio_clip.get_frame(0)  # Essayer de lire le premier frame audio

    # Capturer l'image au timecode de départ
    frame = main_clip.get_frame(start_sec)
    if frame is None:
        raise ValueError(f"Failed to get frame at {start_sec} seconds")

    # Créer un clip statique avec l'image
    static_clip = ImageClip(frame, duration=audio_duration)
    static_clip = static_clip.resized(target_size)
    static_clip = static_clip.with_audio(audio_clip)

    # Vérifier que static_clip a un audio valide
    if static_clip.audio is None:
        raise ValueError("Failed to attach audio to static_clip")

    # Construire le clip final
    new_clip = concatenate_videoclips([
        main_clip.subclipped(0, start_sec),
        static_clip,
        main_clip.subclipped(start_sec)
    ])

    # Vérifier la validité de new_clip
    if new_clip.audio is None:
        raise ValueError("new_clip has no audio after concatenation")

    return new_clip, audio_duration


def transcribe_video_whisper_cli(video_path, output_format, whisper_path, whisper_model, ffmpeg_path, lang):
    """Transcrit une vidéo en utilisant whisper.cpp en ligne de commande.

    Args:
        video_path (str): Chemin vers le fichier vidéo à transcrire
        output_format (str): Format de sortie ('txt' ou 'srt')
        whisper_path (str): Chemin vers l'exécutable whisper.cpp
        whisper_model (str): Modèle whisper à utiliser (tiny, base, small, medium, large)
        ffmpeg_path (str): Chemin vers l'exécutable ffmpeg
        lang (str): Langue de la vidéo (code à 2 lettres)

    Returns:
        str: Le contenu de la transcription ou None en cas d'erreur
    """
    print("Executed by user :", getpass.getuser())

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name

    try:
        # Conversion de la vidéo en audio WAV 16kHz
        print(f"Conversion to {temp_audio_path} 16bits...")
        ffmpeg_command = [
            ffmpeg_path, '-y',
            '-i', video_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            temp_audio_path
        ]
        print("Commande ffmpeg:", " ".join(ffmpeg_command))
        try:
            result = subprocess.run(
                ffmpeg_command, check=True, capture_output=True, text=True)
            print("Output STDOUT:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error while executing ffmpeg:")
            print(e.stderr)

        # Transcription avec whisper.cpp
        print(f"Transcription with whisper {whisper_model}...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as temp_output:
            output_file = temp_output.name
        file_without_extension, file_extension = os.path.splitext(output_file)
        expanded_whisper_path = os.path.expanduser(whisper_path)
        whisper_command = [
            expanded_whisper_path,
            "-m", f"{os.path.dirname(expanded_whisper_path)}/models/ggml-{whisper_model}.bin",
            "-f", temp_audio_path,
            "-l", lang,
            "-of", file_without_extension,
            "-otxt" if output_format == "txt" else "-osrt"
        ]
        print("Command whisper:", " ".join(whisper_command))
        try:
            result = subprocess.run(
                whisper_command, check=True, capture_output=True, text=True)
            print("Sortie STDOUT:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error while executing whisper:")
            print(e.stderr)
        print('Transcription done')

        with open(output_file, 'r') as f:
            transcript = f.read()

        os.remove(output_file)
        return transcript

    except subprocess.CalledProcessError as e:
        st.error(f"{t('transcript_error_transcribing')}{e.stderr}")
        return None

    finally:
        # Nettoyage des fichiers temporaires
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(f"transcript.{output_format}"):
            os.remove(f"transcript.{output_format}")
