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
        return pd.DataFrame(columns=["Start", "End", "Text", "Thumbnail"]), []
    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        chapter_section = False
        for i in range(len(lines)):
            line = lines[i].strip()
            if line == "CHAPTERS":
                chapter_section = True
                continue
            if not chapter_section and "-->" in line and i + 1 < len(lines) and not lines[i].isdigit():
                start, end = line.split(" --> ")
                text = lines[i + 1].strip()
                subtitles.append({"Start": start, "End": end, "Text": text, "Thumbnail": None})
            elif chapter_section and "-->" in line and i + 1 < len(lines):
                start, end = line.split(" --> ")
                title = lines[i + 1].strip()
                start_sec = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start.split(":")[:-1]))) + float(start.split(":")[-1])
                end_sec = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end.split(":")[:-1]))) + float(end.split(":")[-1])
                duration = end_sec - start_sec
                chapters.append({
                    "Start": start,
                    "End": end,
                    "Title": title,
                    "Duration": f"{int(duration // 3600):02d}:{int((duration % 3600) // 60):02d}:{int(duration % 60):02d}"
                })
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

def rename_video(video_path, new_name):
    directory, old_name = os.path.split(video_path)
    extension = os.path.splitext(old_name)[1]
    new_path = os.path.join(directory, new_name + extension)
    os.rename(video_path, new_path)
    old_vtt = os.path.splitext(video_path)[0] + ".vtt"
    if os.path.exists(old_vtt):
        new_vtt = os.path.splitext(new_path)[0] + ".vtt"
        os.rename(old_vtt, new_vtt)
    return new_path

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
