from global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
import os
import pandas as pd
import whisper
from PIL import Image
import io
import cv2
import base64
import ffmpeg

# Traductions
translations["en"].update({
    "video_tab": "Video Editor",
    "video_header": "Video Editor Based on Subtitles",
    "video_config_workdir": "Working Directory",
    "video_config_workdir_default": "/path/to/videos",
    "video_list_label": "Available Videos",
    "video_generate_subtitles": "Generate Subtitles",
    "video_processing": "Processing subtitles generation...",
    "video_success": "Subtitles generated successfully for {video}!",
    "video_error": "An error occurred: {error}",
    "video_subtitles_label": "Subtitles for {video}",
    "video_model_label": "Transcription Model",
    "video_chapter_title": "Chapter Title",
    "video_add_chapter": "Add Chapter",
    "video_delete_chapter": "Delete Chapter",
    "video_edit_chapter": "Edit Chapter",
    "video_chapters_label": "Chapters",
    "video_generate_thumbnails": "Generate Thumbnails",
    "video_refresh_thumbnails": "Refresh Thumbnails",
    "video_convert_to_mp4": "Convert to MP4",
    "video_convert_to_mp4": "Convert to MP4",
    "video_rename": "Rename",
    "video_merge": "Merge Videos",
    "video_split": "Split Video",
    "video_extensions_label": "Filter by Extensions",
})

translations["fr"].update({
    "video_tab": "Éditeur Vidéo",
    "video_header": "Éditeur Vidéo Basé sur les Sous-titres",
    "video_config_workdir": "Répertoire de travail",
    "video_config_workdir_default": "/chemin/vers/vidéos",
    "video_list_label": "Vidéos Disponibles",
    "video_generate_subtitles": "Générer les Sous-titres",
    "video_processing": "Génération des sous-titres en cours...",
    "video_success": "Sous-titres générés avec succès pour {video} !",
    "video_error": "Une erreur s'est produite : {error}",
    "video_subtitles_label": "Sous-titres pour {video}",
    "video_model_label": "Modèle de transcription",
    "video_chapter_title": "Titre du chapitre",
    "video_add_chapter": "Ajouter un chapitre",
    "video_delete_chapter": "Supprimer un chapitre",
    "video_edit_chapter": "Modifier un chapitre",
    "video_chapters_label": "Chapitres",
    "video_generate_thumbnails": "Générer les vignettes",
    "video_refresh_thumbnails": "Rafraîchir les vignettes",
    "video_convert_to_mp4": "Convertir en MP4",
    "video_convert_to_mp4": "Convertir en MP4",
    "video_rename": "Renommer",
    "video_merge": "Fusionner les vidéos",
    "video_split": "Découper la vidéo",
    "video_extensions_label": "Filtrer par extensions",
})

class VideoPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.working_dir = None

    def get_config_fields(self):
        return {
            "video_workdir": {
                "type": "text",
                "label": t("video_config_workdir"),
                "default": t("video_config_workdir_default")
            }
        }

    def get_tabs(self):
        return [{"name": t("video_tab"), "plugin": "videoplugin"}]

    def scan_videos(self, directory, extensions):
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

    def load_subtitles_and_chapters(self, vtt_path):
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

    def save_vtt(self, vtt_path, subtitles_df, chapters_df):
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

    def generate_subtitles(self, video_path, model_name):
        model = whisper.load_model(model_name)
        result = model.transcribe(video_path)
        vtt_path = os.path.splitext(video_path)[0] + ".vtt"
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, segment in enumerate(result["segments"]):
                start = self.format_time(segment["start"])
                end = self.format_time(segment["end"])
                text = segment["text"]
                f.write(f"{i + 1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        return vtt_path

    def convert_to_mp4(self, video_path):
        output_path = os.path.splitext(video_path)[0] + ".mp4"
        if not os.path.exists(output_path):
            try:
                stream = ffmpeg.input(video_path)
                stream = ffmpeg.output(stream, output_path, vcodec="copy", acodec="copy")
                ffmpeg.run(stream)
                st.success(f"Converted {os.path.basename(video_path)} to MP4!")
            except Exception as e:
                st.error(f"Conversion failed: {str(e)}")
        return output_path

    def rename_video(self, video_path, new_name):
        directory, old_name = os.path.split(video_path)
        extension = os.path.splitext(old_name)[1]
        new_path = os.path.join(directory, new_name + extension)
        os.rename(video_path, new_path)
        # Renommer le fichier .vtt associé s'il existe
        old_vtt = os.path.splitext(video_path)[0] + ".vtt"
        if os.path.exists(old_vtt):
            new_vtt = os.path.splitext(new_path)[0] + ".vtt"
            os.rename(old_vtt, new_vtt)
        return new_path

    def merge_videos(self, video_paths, output_dir):
        output_path = os.path.join(output_dir, "merge.mp4")
        inputs = [ffmpeg.input(path) for path in video_paths]
        try:
            stream = ffmpeg.concat(*inputs, v=1, a=1).output(output_path)
            ffmpeg.run(stream)
            st.success("Videos merged into merge.mp4!")
        except Exception as e:
            st.error(f"Merge failed: {str(e)}")
        return output_path

    def split_video(self, video_path, split_time, output_dir):
        split1_path = os.path.join(output_dir, "split1.mp4")
        split2_path = os.path.join(output_dir, "split2.mp4")
        try:
            # Première partie : du début jusqu'au point de coupe
            stream1 = ffmpeg.input(video_path).output(split1_path, t=split_time, vcodec="copy", acodec="copy")
            ffmpeg.run(stream1)
            # Deuxième partie : du point de coupe jusqu'à la fin
            stream2 = ffmpeg.input(video_path, ss=split_time).output(split2_path, vcodec="copy", acodec="copy")
            ffmpeg.run(stream2)
            st.success(f"Video split into {split1_path} and {split2_path}!")
        except Exception as e:
            st.error(f"Split failed: {str(e)}")
        return split1_path, split2_path

    def generate_thumbnail(self, video_path, timestamp):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
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

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def run(self, config):
        st.header(t("video_header"))
        self.working_dir = config.get(self.name, {}).get("video_workdir", t("video_config_workdir_default"))

        # 1. Première ligne avec sélecteur d'extensions
        col_model, col_thumb, col_refresh, col_ext = st.columns([2, 1, 1, 2])
        with col_model:
            model_options = ["base", "medium", "turbo", "large-v3", "large-v3-turbo"]
            selected_model = st.selectbox(t("video_model_label"), model_options, index=0)
        with col_thumb:
            generate_thumbnails = st.checkbox(t("video_generate_thumbnails"))
        with col_refresh:
            refresh_thumbnails = st.button(t("video_refresh_thumbnails"))
        with col_ext:
            extension_options = [".mp4", ".mkv", ".ogg"]
            selected_extensions = st.multiselect(t("video_extensions_label"), extension_options, default=extension_options)

        video_df = self.scan_videos(self.working_dir, selected_extensions)
        if video_df.empty:
            st.write("No videos found with the selected extensions.")
            return

        col1, col2 = st.columns([1, 3])

        with col1:
            st.write(t("video_list_label"))
            selected_videos = st.dataframe(
                video_df[["Video", "Directory", "Duration", "Has Subtitles"]],
                selection_mode="multi-row",
                on_select="rerun",
                key="video_selector",
                hide_index=True
            )

            # 2. Boutons Generate Subtitles et Convert to MP4 sur la même ligne
            col_gen, col_conv = st.columns(2)
            with col_gen:
                if st.button(t("video_generate_subtitles")) and selected_videos["selection"]["rows"]:
                    with st.spinner(t("video_processing")):
                        try:
                            for idx in selected_videos["selection"]["rows"]:
                                video_path = video_df.iloc[idx]["Full Path"]
                                if not video_df.iloc[idx]["Has Subtitles"]:
                                    self.generate_subtitles(video_path, selected_model)
                                    st.success(t("video_success").format(video=os.path.basename(video_path)))
                            st.rerun()
                        except Exception as e:
                            st.error(t("video_error").format(error=str(e)))
            with col_conv:
                if st.button(t("video_convert_to_mp4")) and selected_videos["selection"]["rows"]:
                    for idx in selected_videos["selection"]["rows"]:
                        video_path = video_df.iloc[idx]["Full Path"]
                        if not video_path.endswith(".mp4"):
                            self.convert_to_mp4(video_path)
                    st.rerun()

            # 3. Renommer la vidéo
            if selected_videos["selection"]["rows"]:
                selected_idx = selected_videos["selection"]["rows"][0]
                selected_video = video_df.iloc[selected_idx]
                new_name = st.text_input("New Video Name", os.path.splitext(selected_video["Video"])[0])
                if st.button(t("video_rename")) and new_name:
                    self.rename_video(selected_video["Full Path"], new_name)
                    st.rerun()

            # 4. Fusionner les vidéos
            if st.button(t("video_merge")) and selected_videos["selection"]["rows"]:
                video_paths = [video_df.iloc[idx]["Full Path"] for idx in selected_videos["selection"]["rows"]]
                self.merge_videos(video_paths, self.working_dir)
                st.rerun()

            if selected_videos["selection"]["rows"]:
                selected_idx = selected_videos["selection"]["rows"][0]
                selected_video = video_df.iloc[selected_idx]
                vtt_path = os.path.splitext(selected_video["Full Path"])[0] + ".vtt"
                if selected_video["Has Subtitles"]:
                    subtitles_df, chapters_df = self.load_subtitles_and_chapters(vtt_path)
                    if not chapters_df.empty:
                        st.write(t("video_chapters_label"))
                        selected_chapters = st.dataframe(
                            chapters_df,
                            selection_mode="multi-row",
                            on_select="rerun",
                            key="chapter_selector",
                            hide_index=True
                        )
                        if selected_chapters["selection"]["rows"]:
                            chapter_idx = selected_chapters["selection"]["rows"][0]
                            new_title = st.text_input(t("video_chapter_title"), chapters_df.iloc[chapter_idx]["Title"])
                            if st.button(t("video_edit_chapter")) and new_title:
                                chapters_df.at[chapter_idx, "Title"] = new_title
                                self.save_vtt(vtt_path, subtitles_df, chapters_df)
                                st.rerun()
                        if st.button(t("video_delete_chapter")) and selected_chapters["selection"]["rows"]:
                            chapter_idx = selected_chapters["selection"]["rows"][0]
                            chapters_df = chapters_df.drop(chapter_idx).reset_index(drop=True)
                            self.save_vtt(vtt_path, subtitles_df, chapters_df)
                            st.rerun()

        with col2:
            if selected_videos["selection"]["rows"]:
                selected_idx = selected_videos["selection"]["rows"][0]
                selected_video = video_df.iloc[selected_idx]
                vtt_path = os.path.splitext(selected_video["Full Path"])[0] + ".vtt"
                if selected_video["Has Subtitles"]:
                    subtitles_df, chapters_df = self.load_subtitles_and_chapters(vtt_path)

                    if "thumbnails" not in st.session_state:
                        st.session_state["thumbnails"] = {}
                    if refresh_thumbnails or (generate_thumbnails and not st.session_state["thumbnails"].get(vtt_path)):
                        st.session_state["thumbnails"][vtt_path] = {}
                        for i, row in subtitles_df.iterrows():
                            seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(row["Start"].split(":")[:-1]))) + float(row["Start"].split(":")[-1])
                            thumbnail = self.generate_thumbnail(selected_video["Full Path"], seconds)
                            st.session_state["thumbnails"][vtt_path][i] = thumbnail
                    if generate_thumbnails:
                        for i, thumbnail in st.session_state["thumbnails"].get(vtt_path, {}).items():
                            subtitles_df.at[i, "Thumbnail"] = thumbnail

                    if "chapter_selector" in st.session_state and st.session_state["chapter_selector"]["selection"]["rows"]:
                        selected_chapter_indices = st.session_state["chapter_selector"]["selection"]["rows"]
                        filtered_subtitles_df = pd.concat([
                            subtitles_df[(subtitles_df["Start"] >= chapters_df.iloc[idx]["Start"]) & (subtitles_df["End"] <= chapters_df.iloc[idx]["End"])]
                            for idx in selected_chapter_indices
                        ]).drop_duplicates()
                    else:
                        filtered_subtitles_df = subtitles_df

                    st.write(t("video_subtitles_label").format(video=selected_video["Video"]))
                    selected_subtitles = st.dataframe(
                        filtered_subtitles_df,
                        selection_mode="multi-row",
                        on_select="rerun",
                        key="subtitle_selector",
                        column_config={"Thumbnail": st.column_config.ImageColumn("Thumbnail", width="small") if generate_thumbnails else None},
                        hide_index=True
                    )

                    if selected_subtitles["selection"]["rows"]:
                        selected_subtitle_idx = selected_subtitles["selection"]["rows"][0]
                        selected_subtitle = filtered_subtitles_df.iloc[selected_subtitle_idx]
                        col_edit_start, col_edit_end, col_edit_text, col_edit_save = st.columns([1, 1, 3, 1])
                        with col_edit_start:
                            edited_start = st.text_input("Start", selected_subtitle["Start"], key="edit_start")
                        with col_edit_end:
                            edited_end = st.text_input("End", selected_subtitle["End"], key="edit_end")
                        with col_edit_text:
                            edited_text = st.text_input("Text", selected_subtitle["Text"], key="edit_text")
                        with col_edit_save:
                            if st.button("Save Subtitle"):
                                subtitles_df.at[selected_subtitle_idx, "Start"] = edited_start
                                subtitles_df.at[selected_subtitle_idx, "End"] = edited_end
                                subtitles_df.at[selected_subtitle_idx, "Text"] = edited_text
                                self.save_vtt(vtt_path, subtitles_df, chapters_df)
                                st.rerun()

                        # 5. Découper la vidéo
                        if st.button(t("video_split")):
                            split_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(selected_subtitle["Start"].split(":")[:-1]))) + float(selected_subtitle["Start"].split(":")[-1])
                            self.split_video(selected_video["Full Path"], split_time, self.working_dir)
                            st.rerun()

                        if selected_video["Full Path"].endswith(".mp4"):
                            start_sec = sum(float(x) * 60 ** i for i, x in enumerate(reversed(selected_subtitle["Start"].split(":")[:-1]))) + float(selected_subtitle["Start"].split(":")[-1])
                            end_sec = sum(float(x) * 60 ** i for i, x in enumerate(reversed(selected_subtitle["End"].split(":")[:-1]))) + float(selected_subtitle["End"].split(":")[-1])
                            st.video(
                                selected_video["Full Path"],
                                start_time=start_sec,
                                end_time=end_sec,
                                autoplay=True
                            )
                        else:
                            st.write("Video preview only available for .mp4 files.")

                    if selected_subtitles["selection"]["rows"] and len(selected_subtitles["selection"]["rows"]) >= 2:
                        start_idx = selected_subtitles["selection"]["rows"][0]
                        end_idx = selected_subtitles["selection"]["rows"][-1]
                        start_time = filtered_subtitles_df.iloc[start_idx]["Start"]
                        end_time = filtered_subtitles_df.iloc[end_idx]["End"]
                        chapter_title = st.text_input(t("video_chapter_title"), "")
                        if st.button(t("video_add_chapter")) and chapter_title:
                            new_chapter = pd.DataFrame([{"Start": start_time, "End": end_time, "Title": chapter_title}])
                            chapters_df = pd.concat([chapters_df, new_chapter], ignore_index=True)
                            self.save_vtt(vtt_path, subtitles_df, chapters_df)
                            st.rerun()
                else:
                    st.write("No subtitles available for this video yet.")
            else:
                st.write("Select a video to view subtitles.")

if __name__ == "__main__":
    st.write("Video Plugin standalone test")
