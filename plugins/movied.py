from tkinter.constants import VERTICAL
from tarfile import version
from enum import verify
import base64
from global_vars import translations, t
from app import Plugin
import streamlit as st
import pandas as pd
import os
from video_utils import *
import json
from moviepy import VideoFileClip

# Translations
translations["en"].update({
    "movied_tab": "Automated Video Editor",
    "movied_header": "Automated Video Editor",
    "movied_workdir": "Movie Editor Working Directory",
    "movied_workdir_default": "/path/to/videos",
    "movied_media_dirs": "Media Directories (one per line)",
    "movied_media_dirs_default": "/path/to/images\n/path/to/videos",
    "movied_video_list": "Available Videos",
    "movied_generate_transcript": "Generate Transcript",
    "movied_processing": "Generating transcript...",
    "movied_success": "Transcript generated for {video}!",
    "movied_error": "Error: {error}",
    "movied_subtitles": "Subtitles for {video}",
    "movied_start_time": "Start Time",
    "movied_end_time": "End Time",
    "movied_operations": "Operations Queue",
    "movied_replace_image": "Replace with Image",
    "movied_insert_video": "Insert Video at Start",
    "movied_replace_video": "Replace with Video",
    "movied_generate": "Generate Video",
    "movied_model_label": "Transcription Model",
    "movied_replace_video_keep_audio": "Replace Video (Keep Original Audio)",
    "movied_filter_media_dir": "Filter by Media Directory",
    "movied_all_directories": "All Directories",
    "movied_video_thumbnail": "Thumbnail",
    "movied_font_label": "Font",
    "movied_font_size_label": "Font Size",
    "movied_animate_text": "Animate Text",
    "movied_text_input": "Enter text (use \\ for line breaks)",
    "movied_text_operations": "Text Operations",
    "movied_text_background": "Text Background",
    "movied_green_background": "Green Background",
    "movied_original_video": "Original Video",
    "movied_remove_section": "Remove Section",
})

translations["fr"].update({
    "movied_tab": "Éditeur Vidéo Automatisé",
    "movied_header": "Éditeur Vidéo Automatisé",
    "movied_workdir": "Répertoire de travail Editeur de Vidéo Automatisé",
    "movied_workdir_default": "/chemin/vers/vidéos",
    "movied_media_dirs": "Répertoires de médias (un par ligne)",
    "movied_media_dirs_default": "/chemin/vers/images\n/chemin/vers/vidéos",
    "movied_video_list": "Vidéos disponibles",
    "movied_generate_transcript": "Générer le transcript",
    "movied_processing": "Génération du transcript en cours...",
    "movied_success": "Transcript généré pour {video} !",
    "movied_error": "Erreur : {error}",
    "movied_subtitles": "Sous-titres pour {video}",
    "movied_start_time": "Heure de début",
    "movied_end_time": "Heure de fin",
    "movied_operations": "File d'opérations",
    "movied_replace_image": "Remplacer par une image",
    "movied_insert_video": "Insérer une vidéo au début",
    "movied_replace_video": "Remplacer par une vidéo",
    "movied_generate": "Générer la vidéo",
    "movied_model_label": "Modèle de transcription",
    "movied_replace_video_keep_audio": "Remplacer la vidéo (Garder l'audio original)",
    "movied_filter_media_dir": "Filtrer par répertoire de médias",
    "movied_all_directories": "Tous les répertoires",
    "movied_video_thumbnail": "Vignette",
    "movied_font_label": "Police",
    "movied_font_size_label": "Taille de la police",
    "movied_animate_text": "Animer le texte",
    "movied_text_input": "Entrez le texte (utilisez \\ pour les sauts de ligne)",
    "movied_text_operations": "Opérations de texte",
    "movied_text_background": "Fond du texte",
    "movied_green_background": "Fond vert",
    "movied_original_video": "Vidéo originale",
    "movied_remove_section": "Supprimer la section",
})


class MoviedPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.working_dir = None
        self.media_dirs = []

    def get_config_fields(self):
        return {
            "movied_workdir": {
                "type": "text",
                "label": t("movied_workdir"),
                "default": t("movied_workdir_default")
            },
            "movied_media_dirs": {
                "type": "textarea",
                "label": t("movied_media_dirs"),
                "default": t("movied_media_dirs_default")
            }
        }

    def get_tabs(self):
        return [{"name": t("movied_tab"), "plugin": "movied"}]

    def setup_header(self):
        st.header(t("movied_header"))

    def setup_controls(self):
        with st.expander("Options"):
            selected_model = st.selectbox(t("movied_model_label"), [
                                          "base", "medium", "turbo", "large-v3", "large-v3-turbo"], index=0)
            thumbnail_size = st.selectbox(
                "Thumbnail Size",
                ["small", "medium", "large"],
                index=1,
                key="thumbnail_size"
            )
            font = st.selectbox(
                t("movied_font_label"),
                ["Arial", "Times New Roman", "Courier New", "Verdana"],
                index=0,
                key="font_select"
            )
            font_size = st.slider(
                t("movied_font_size_label"),
                50, 200, 100, step=5,
                key="font_size_slider"
            )
            text_background = st.selectbox(
                t("movied_text_background"),
                [t("movied_green_background"), t("movied_original_video")],
                index=0,
                key="text_background_select"
            )
            return selected_model, thumbnail_size, font, font_size, text_background

    def list_videos(self):
        video_extensions = [".mp4", ".mkv", ".avi"]
        videos = []
        for file in os.listdir(self.working_dir):
            if os.path.splitext(file)[1].lower() in video_extensions:
                full_path = os.path.join(self.working_dir, file)
                vtt_path = os.path.splitext(full_path)[0] + ".vtt"
                videos.append({
                    "Video": file,
                    "Full Path": full_path,
                    "Has Transcript": os.path.exists(vtt_path)
                })
        return pd.DataFrame(videos)

    def display_videos(self, video_df):
        st.write(t("movied_video_list"))
        selected_video = st.dataframe(
            video_df[["Video", "Has Transcript"]],
            selection_mode="single-row",
            on_select="rerun",
            key="video_selector",
            hide_index=True
        )
        return selected_video

    def handle_transcript(self, selected_video, video_df, selected_model):
        if selected_video["selection"]["rows"]:
            idx = selected_video["selection"]["rows"][0]
            video_info = video_df.iloc[idx]
            vtt_path = os.path.splitext(video_info["Full Path"])[0] + ".vtt"

            # Always display the "Generate Transcript" button
            if st.button(t("movied_generate_transcript")):
                with st.spinner(t("movied_processing")):
                    try:
                        generate_subtitles(
                            video_info["Full Path"], selected_model)
                        subtitles_df, _ = load_subtitles_and_chapters(vtt_path)
                        st.session_state["subtitles_df"] = subtitles_df
                        st.session_state["current_vtt_path"] = vtt_path
                        st.success(t("movied_success").format(
                            video=os.path.basename(video_info["Full Path"])))
                        st.rerun()
                    except Exception as e:
                        st.error(t("movied_error").format(error=str(e)))
                        return None, None, None

            # Load existing subtitles if available
            if os.path.exists(vtt_path):
                if "subtitles_df" not in st.session_state or st.session_state.get("current_vtt_path") != vtt_path:
                    subtitles_df, _ = load_subtitles_and_chapters(vtt_path)
                    st.session_state["subtitles_df"] = subtitles_df
                    st.session_state["current_vtt_path"] = vtt_path
                else:
                    subtitles_df = st.session_state["subtitles_df"]

                st.write(t("movied_subtitles").format(
                    video=video_info["Video"]))
                selected_subtitles = st.dataframe(
                    subtitles_df[["Start", "End", "Text"]],
                    selection_mode="multi-row",
                    on_select="rerun",
                    key="subtitle_selector",
                    hide_index=True
                )
                return selected_subtitles, subtitles_df, vtt_path
            return None, None, None
        return None, None, None

    def handle_section(self, selected_subtitles, subtitles_df):
        if selected_subtitles and selected_subtitles["selection"]["rows"]:
            selected_indices = selected_subtitles["selection"]["rows"]
            if not subtitles_df.empty and selected_indices[0] < len(subtitles_df):
                start_time = subtitles_df.iloc[selected_indices[0]]["Start"]
                end_time = subtitles_df.iloc[selected_indices[-1]]["End"]

                col1, col2 = st.columns(2)
                with col1:
                    edited_start = st.text_input(
                        t("movied_start_time"), start_time, key="start_time")
                with col2:
                    edited_end = st.text_input(
                        t("movied_end_time"), end_time, key="end_time")
                return edited_start, edited_end
            else:
                st.error(
                    "Selected subtitle index out of bounds or subtitles DataFrame is empty.")
                return None, None
        return None, None

    def list_media_files(self):
        # Clé pour stocker les vignettes dans session_state
        if "thumbnail_size" not in st.session_state:
            st.session_state["thumbnail_size"] = "medium"  # Valeur par défaut
        current_size = st.session_state["thumbnail_size"]

        # Vérifier si les vignettes doivent être régénérées
        regenerate = ("media_thumbnails" not in st.session_state or
                      st.session_state.get("last_thumbnail_size") != current_size)

        if regenerate:
            media_files = {"images": [], "videos": []}
            for dir_path in self.media_dirs:
                if not os.path.exists(dir_path):
                    continue
                files = [f for f in os.listdir(
                    dir_path) if os.path.isfile(os.path.join(dir_path, f))]
                for file in files:
                    full_path = os.path.join(dir_path, file)
                    if file.lower().endswith((".jpg", ".png")):
                        base64_url = image_to_base64(full_path)
                        if base64_url:
                            media_files["images"].append({
                                "File": file,
                                "Path": full_path,
                                "Preview": base64_url
                            })
                    elif file.lower().endswith((".mp4", ".mkv", ".avi")):
                        thumbnail = generate_thumbnail(full_path, 0)
                        if thumbnail:
                            media_files["videos"].append({
                                "File": file,
                                "Path": full_path,
                                "Preview": thumbnail
                            })
            st.session_state["media_thumbnails"] = {
                "images": pd.DataFrame(media_files["images"]),
                "videos": pd.DataFrame(media_files["videos"])
            }
            st.session_state["last_thumbnail_size"] = current_size

        # Récupérer depuis session_state
        image_df = st.session_state["media_thumbnails"]["images"]
        video_df = st.session_state["media_thumbnails"]["videos"]
        return image_df, video_df

    def handle_operations(self, start_time, end_time, video_path, vtt_path, thumbnail_size, font, font_size):
        if start_time and end_time:
            try:
                image_df, video_df = self.list_media_files()
            except ValueError as e:
                st.error(f"Error unpacking media files: {e}")
                return

            if image_df.empty and video_df.empty:
                st.warning(
                    "No media files found in the configured directories.")
                return

            st.subheader("Media Selection")
            col1, col2 = st.columns(2)

            media_dir_options = [t("movied_all_directories")] + self.media_dirs
            default_dir_index = 0

            with col1:
                st.write("Images for Replacement")
                image_filter_dir = st.selectbox(
                    t("movied_filter_media_dir"),
                    media_dir_options,
                    index=default_dir_index,
                    key="image_filter_selectbox"
                )
                filtered_image_df = image_df if image_filter_dir == t("movied_all_directories") else image_df[
                    image_df["Path"].str.startswith(image_filter_dir)
                ]
                selected_image = st.dataframe(
                    filtered_image_df[["File", "Preview"]],
                    column_config={
                        "File": st.column_config.TextColumn("Image Name"),
                        "Preview": st.column_config.ImageColumn(
                            "Preview",
                            help="Preview of the image",
                            width=thumbnail_size  # "small", "medium", ou "large"
                        )
                    },
                    height=200,
                    hide_index=True,
                    selection_mode="single-row",
                    on_select="rerun",
                    key="image_media_selector",
                    # row_height=75, #https://github.com/streamlit/streamlit/issues/7266#event-16543333224
                )
                selected_image_path = (filtered_image_df.iloc[selected_image["selection"]["rows"][0]]["Path"]
                                       if selected_image["selection"]["rows"] else None)
                if st.button(t("movied_replace_image"), key="replace_image_btn"):
                    if selected_image_path:
                        operation = f"replace_image {start_time} {end_time} {selected_image_path}"
                        self.add_to_operations(operation)
                    else:
                        st.warning("Please select an image first.")

            with col2:
                st.write("Video Operations")
                video_filter_dir = st.selectbox(
                    t("movied_filter_media_dir"),
                    media_dir_options,
                    index=default_dir_index,
                    key="video_filter_selectbox"
                )
                if video_filter_dir == t("movied_all_directories"):
                    filtered_video_df = video_df
                else:
                    expected_paths = video_df["File"].apply(
                        lambda f: os.path.join(video_filter_dir, f))
                    filtered_video_df = video_df[
                        video_df["Path"].isin(expected_paths)
                    ]
                selected_video = st.dataframe(
                    filtered_video_df[["File", "Preview"]],
                    column_config={
                        "File": st.column_config.TextColumn("Video Name"),
                        "Preview": st.column_config.ImageColumn(
                            t("movied_video_thumbnail"),
                            help="Thumbnail of the video",
                            width=thumbnail_size  # "small", "medium", ou "large"
                        )
                    },
                    height=200,
                    hide_index=True,
                    selection_mode="single-row",
                    on_select="rerun",
                    key="video_media_selector"
                )
                selected_video_path = (filtered_video_df.iloc[selected_video["selection"]["rows"][0]]["Path"]
                                       if selected_video["selection"]["rows"] else None)

                col_video1, col_video2, col_video3 = st.columns(3)
                with col_video1:
                    if st.button(t("movied_insert_video"), key="insert_video_btn"):
                        if selected_video_path:
                            operation = f"insert_video {start_time} {selected_video_path}"
                            self.add_to_operations(operation)
                        else:
                            st.warning("Please select a video first.")

                with col_video2:
                    if st.button(t("movied_replace_video"), key="replace_video_btn"):
                        if selected_video_path:
                            operation = f"replace_video {start_time} {end_time} {selected_video_path}"
                            self.add_to_operations(operation)
                        else:
                            st.warning("Please select a video first.")

                with col_video3:
                    if st.button(t("movied_replace_video_keep_audio"), key="replace_video_keep_audio_btn"):
                        if selected_video_path:
                            operation = f"replace_video_keep_audio {start_time} {end_time} {selected_video_path}"
                            self.add_to_operations(operation)
                        else:
                            st.warning("Please select a video first.")

            if st.button(t("movied_remove_section"), key="remove_section_btn"):
                operation = f"remove_section {start_time} {end_time}"
                self.add_to_operations(operation)
            st.write(t("movied_text_operations"))
            text_input = st.text_area(
                t("movied_text_input"), height=100, key="text_input")
            if st.button(t("movied_animate_text"), key="animate_text_btn"):
                if text_input:
                    # Convertir les sauts de ligne en \\
                    text_command = text_input.replace("\n", "\\")
                    operation = f"addtext {start_time} {end_time} fromLeft 1s {text_command}"
                    self.add_to_operations(operation)
                else:
                    st.warning("Please enter text first.")

            operations = st.text_area(t("movied_operations"), value=st.session_state.get(
                "operations", ""), key="operations_area")
            st.session_state["operations"] = operations

            if st.button(t("movied_generate"), key="generate_btn") and operations:
                self.execute_operations(
                    video_path, vtt_path, operations, font, font_size)

    def add_to_operations(self, operation):
        current_ops = st.session_state.get("operations", "")
        st.session_state["operations"] = f"{current_ops}\n{operation}".strip()

    def parse_timecode(self, timecode):
        h, m, s = map(float, timecode.replace(",", ".").split(":"))
        return h * 3600 + m * 60 + s

    def adjust_subtitles(self, subtitles_df, start_time, duration_change):
        start_seconds = self.parse_timecode(start_time)
        for i, row in subtitles_df.iterrows():
            row_start = self.parse_timecode(row["Start"])
            row_end = self.parse_timecode(row["End"])
            if row_start >= start_seconds:
                # Adjust start time
                new_start = row_start + duration_change
                hours = int(new_start // 3600)
                minutes = int((new_start % 3600) // 60)
                seconds = new_start % 60
                subtitles_df.at[i,
                                "Start"] = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

                # Adjust end time
                new_end = row_end + duration_change
                hours = int(new_end // 3600)
                minutes = int((new_end % 3600) // 60)
                seconds = new_end % 60
                subtitles_df.at[i,
                                "End"] = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        return subtitles_df

    def execute_operations(self, video_path, vtt_path, operations, font, font_size):
        with st.spinner("Processing video operations..."):
            try:
                from video_utils import (replace_with_image, insert_video,
                                         replace_with_video, replace_video_keep_audio,
                                         add_animated_text, remove_section)

                # Initialisation
                main_clip = VideoFileClip(video_path)
                target_size = (main_clip.w, main_clip.h)
                subtitles_df, _ = load_subtitles_and_chapters(vtt_path)
                duration_offset = 0

                # Préparation du fond pour le texte
                text_background = st.session_state.get(
                    "text_background_select", t("movied_green_background"))
                use_green_background = text_background == t(
                    "movied_green_background")

                with st.expander("Debug Information"):
                    # Parcourir chaque opération
                    for op in operations.split("\n"):
                        if not op.strip():
                            continue
                        parts = op.split(maxsplit=5)
                        cmd = parts[0]
                        st.write(f"Processing: {op}")

                        # Extraire et parser les timecodes (commun à toutes les commandes sauf insert_video qui n'a qu'un start_time)
                        if cmd == "insert_video":
                            start_time = parts[1]
                            start_sec = self.parse_timecode(
                                start_time) + duration_offset
                            end_sec = None
                            remaining_args = " ".join(parts[2:])
                        else:
                            start_time, end_time = parts[1], parts[2]
                            start_sec = self.parse_timecode(
                                start_time) + duration_offset
                            end_sec = self.parse_timecode(
                                end_time) + duration_offset
                            remaining_args = " ".join(
                                parts[3:]) if len(parts) > 3 else ""

                        # Exécuter la commande correspondante
                        if cmd == "replace_image":
                            image_path = remaining_args
                            main_clip = replace_with_image(
                                main_clip, start_sec, end_sec, image_path, target_size)
                            col1, _ = st.columns([1, 3])
                            with col1:
                                st.image(
                                    image_path, caption=f"Using image: {image_path}", width=100)

                        elif cmd == "insert_video":
                            video_path_insert = remaining_args
                            main_clip, duration_change = insert_video(
                                main_clip, start_sec, video_path_insert, target_size)
                            subtitles_df = self.adjust_subtitles(
                                subtitles_df, start_time, duration_change)
                            duration_offset += duration_change
                            col1, _ = st.columns([1, 3])
                            with col1:
                                st.video(video_path_insert)

                        elif cmd == "replace_video":
                            video_path_replace = remaining_args
                            main_clip, duration_change = replace_with_video(
                                main_clip, start_sec, end_sec, video_path_replace, target_size)
                            subtitles_df = self.adjust_subtitles(
                                subtitles_df, end_time, duration_change)
                            duration_offset += duration_change
                            col1, _ = st.columns([1, 3])
                            with col1:
                                st.video(
                                    video_path_replace, caption=f"Using video: {video_path_replace}", width=100)

                        elif cmd == "replace_video_keep_audio":
                            video_path_replace = remaining_args
                            main_clip = replace_video_keep_audio(
                                main_clip, start_sec, end_sec, video_path_replace, target_size)
                            col1, _ = st.columns([1, 3])
                            with col1:
                                st.video(video_path_replace)

                        elif cmd == "addtext":
                            animation_type, anim_duration, text = parts[3], parts[4], parts[5]
                            anim_duration_sec = float(anim_duration[:-1])
                            main_clip = add_animated_text(
                                main_clip, start_sec, end_sec, text, animation_type,
                                anim_duration_sec, target_size, font, font_size,
                                use_green_background=use_green_background
                            )

                        elif cmd == "remove_section":
                            main_clip, duration_change = remove_section(
                                main_clip, start_sec, end_sec)
                            subtitles_df = self.adjust_subtitles(
                                subtitles_df, start_time, duration_change)
                            duration_offset += duration_change
                            st.write(
                                f"Section removed from {start_time} to {end_time}")

                    # Sauvegarde de la vidéo éditée et des sous-titres ajustés
                    output_path = os.path.splitext(
                        video_path)[0] + "_edited.mp4"
                    main_clip.write_videofile(
                        output_path, codec="libx264", audio_codec="aac")
                    save_vtt(vtt_path, subtitles_df, pd.DataFrame())
                    st.success(
                        f"Video generated successfully at {output_path}")
                    st.rerun()
            except Exception as e:
                st.error(t("movied_error").format(error=str(e)))
                raise e
            finally:
                main_clip.close()

    def run(self, config):
        self.working_dir = config.get(self.name, {}).get(
            "movied_workdir", t("movied_workdir_default"))
        self.media_dirs = config.get(self.name, {}).get(
            "movied_media_dirs", t("movied_media_dirs_default")).split("\n")

        self.setup_header()
        selected_model, thumbnail_size, font, font_size, text_background = self.setup_controls()

        video_df = self.list_videos()
        if video_df.empty:
            st.write("No videos found in the working directory.")
            return

        selected_video = self.display_videos(video_df)
        selected_subtitles, subtitles_df, vtt_path = self.handle_transcript(
            selected_video, video_df, selected_model)
        start_time, end_time = self.handle_section(
            selected_subtitles, subtitles_df)

        if selected_video["selection"]["rows"]:
            video_path = video_df.iloc[selected_video["selection"]
                                       ["rows"][0]]["Full Path"]
            self.handle_operations(
                start_time, end_time, video_path, vtt_path, thumbnail_size, font, font_size)


if __name__ == "__main__":
    st.write("Movied Plugin standalone test")
