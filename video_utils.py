import os
import pandas as pd
import whisper
from PIL import Image
import io
import cv2
import base64
import ffmpeg
import streamlit as st

def scan_videos(directory, extensions):
    video_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_path = os.path.join(root, file)
                vtt_path = os.path.splitext(video_path)[0] + ".vtt"
                has_subtitles = os.path.exists(vtt_path)
                relative_dir = os.path.relpath(root, directory)
                cap = cv2.VideoCapture(video_path)
                duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
                cap.release()
                video_data.append({
                    "Video": file,
                    "Directory": relative_dir if relative_dir != "." else "",
                    "Duration": f"{int(duration // 3600):02d}:{int((duration % 3600) // 60):02d}:{int(duration % 60):02d}",
                    "Has Subtitles": has_subtitles,
                    "Full Path": video_path
                })
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

                text = "\n".join(text_lines)  # Joindre les lignes avec des retours à la ligne

                if not chapter_section:
                    # C'est un sous-titre
                    subtitles.append({"Start": start, "End": end, "Text": text, "Thumbnail": None})
                else:
                    # C'est un chapitre
                    start_sec = sum(float(x) * 60 ** j for j, x in enumerate(reversed(start.split(":")[:-1]))) + float(start.split(":")[-1])
                    end_sec = sum(float(x) * 60 ** j for j, x in enumerate(reversed(end.split(":")[:-1]))) + float(end.split(":")[-1])
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
        stream = ffmpeg.output(stream, output_path, vcodec="h264", acodec="aac", strict="experimental")
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
            raise Exception(f"Failed to rename {file_path} to {new_path}: {str(e)}")

def merge_videos(video_paths, output_dir):
    output_path = os.path.join(output_dir, "merge.mp4")
    inputs = [ffmpeg.input(path) for path in video_paths]
    try:
        stream = ffmpeg.concat(*inputs, v=1, a=1).output(output_path)
        ffmpeg.run(stream)
        st.success("Videos merged into merge.mp4!")
    except Exception as e:
        st.error(f"Merge failed: {str(e)}")
    return output_path

def split_video_fast(video_path, split_time_ms, output_dir):
    """Découpe une vidéo en deux parties à un point donné en millisecondes."""
    split1_path = os.path.join(output_dir, "split1.mp4")
    split2_path = os.path.join(output_dir, "split2.mp4")
    try:
        # Convertir millisecondes en secondes avec précision pour FFmpeg
        split_time = split_time_ms / 1000.0
        stream1 = ffmpeg.input(video_path).output(split1_path, t=split_time, vcodec="copy", acodec="copy")
        ffmpeg.run(stream1)
        stream2 = ffmpeg.input(video_path, ss=split_time).output(split2_path, vcodec="copy", acodec="copy")
        ffmpeg.run(stream2)
        st.success(f"Video split into {split1_path} and {split2_path}!")
    except Exception as e:
        st.error(f"Split failed: {str(e)}")
    return split1_path, split2_path

def split_video(video_path, split_time_ms, output_dir):
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

def split_by_chapters(video_path, output_dir, chapters_df=None, split_time_ms=None):
    """Découpe une vidéo en segments basés sur les chapitres ou un seul point de découpe.

    Args:
        video_path (str): Chemin de la vidéo source.
        output_dir (str): Répertoire de sortie pour les fichiers découpés.
        chapters_df (pd.DataFrame, optional): DataFrame contenant Start, End, et Title des chapitres.
        split_time_ms (int, optional): Point de découpe unique en millisecondes (si aucun chapitre n’est fourni).

    Returns:
        list: Liste des chemins des fichiers générés.
    """
    output_files = []
    try:
        # Cas 1 : Découpage par un seul point (ancien comportement de split_video)
        if split_time_ms is not None and chapters_df is None:
            split_time = split_time_ms / 1000.0  # Convertir en secondes
            split1_path = os.path.join(output_dir, "split1.mp4")
            split2_path = os.path.join(output_dir, "split2.mp4")

            stream1 = ffmpeg.input(video_path, ss=0).output(
                split1_path, t=split_time, vcodec="h264", acodec="aac", strict="experimental",
                map_metadata="-1", reset_timestamps=1
            )
            ffmpeg.run(stream1)
            stream2 = ffmpeg.input(video_path, ss=split_time).output(
                split2_path, vcodec="h264", acodec="aac", strict="experimental",
                map_metadata="-1", reset_timestamps=1
            )
            ffmpeg.run(stream2)
            output_files = [split1_path, split2_path]
            st.success(f"Video split into {split1_path} and {split2_path}!")

        # Cas 2 : Découpage par chapitres
        elif chapters_df is not None and not chapters_df.empty:
            for i, row in chapters_df.iterrows():
                start_time = parse_timecode_to_ms(row["Start"]) / 1000.0  # Convertir en secondes
                end_time = parse_timecode_to_ms(row["End"]) / 1000.0
                duration = end_time - start_time
                # Nom du fichier : "n - titre.mp4" (première ligne du titre seulement)
                title = row["Title"].split("\n")[0].replace(":", "-").replace("/", "-")  # Remplacer caractères interdits
                output_path = os.path.join(output_dir, f"{i + 1} - {title}.mp4")

                stream = ffmpeg.input(video_path, ss=start_time).output(
                    output_path, t=duration, vcodec="h264", acodec="aac", strict="experimental",
                    map_metadata="-1", reset_timestamps=1
                )
                ffmpeg.run(stream)
                output_files.append(output_path)
            st.success(f"Video split into {len(output_files)} chapters!")

        else:
            st.error("No chapters or split time provided for splitting.")
            return []

    except Exception as e:
        st.error(f"Split failed: {str(e)}")
        return []

    return output_files

def delete_videos(video_paths):
    for path in video_paths:
        vtt_path = os.path.splitext(path)[0] + ".vtt"
        try:
            os.remove(path)
            if os.path.exists(vtt_path):
                os.remove(vtt_path)
            st.success(f"Deleted {os.path.basename(path)}!")
        except Exception as e:
            st.error(f"Deletion failed for {os.path.basename(path)}: {str(e)}")

def generate_thumbnail(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp)  # timestamp en millisecondes maintenant
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
        st.error(f"Erreur lors de la conversion du timecode {timecode} : {str(e)}")
        return 0
