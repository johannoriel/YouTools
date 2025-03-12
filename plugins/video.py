from global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
from video_utils import (
    scan_videos, load_subtitles_and_chapters, save_vtt, generate_subtitles,
    convert_to_mp4, rename_video, merge_videos, split_video, delete_videos,
    generate_thumbnail, format_time
)
import pandas as pd
import os

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
    "video_convert_to_mp4": "Convert to MP4",
    "video_rename": "Rename",
    "video_merge": "Merge Videos",
    "video_split": "Split Video",
    "video_extensions_label": "Filter by Extensions",
    "video_delete": "Delete Videos",
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
    "video_convert_to_mp4": "Convertir en MP4",
    "video_rename": "Renommer",
    "video_merge": "Fusionner les vidéos",
    "video_split": "Découper la vidéo",
    "video_extensions_label": "Filtrer par extensions",
    "video_delete": "Supprimer les vidéos",
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

    def setup_header(self):
        st.header(t("video_header"))

    def setup_controls(self):
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
        return selected_model, generate_thumbnails, refresh_thumbnails, selected_extensions

    def display_videos(self, video_df):
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
        return col1, col2, selected_videos

    def handle_video_actions(self, col1, selected_videos, video_df, selected_model):
        with col1:
            col_gen, col_conv = st.columns(2)
            with col_gen:
                if st.button(t("video_generate_subtitles")) and selected_videos["selection"]["rows"]:
                    with st.spinner(t("video_processing")):
                        try:
                            for idx in selected_videos["selection"]["rows"]:
                                video_path = video_df.iloc[idx]["Full Path"]
                                generate_subtitles(video_path, selected_model)
                                st.success(t("video_success").format(video=os.path.basename(video_path)))
                            st.rerun()
                        except Exception as e:
                            st.error(t("video_error").format(error=str(e)))
            with col_conv:
                if st.button(t("video_convert_to_mp4")) and selected_videos["selection"]["rows"]:
                    for idx in selected_videos["selection"]["rows"]:
                        video_path = video_df.iloc[idx]["Full Path"]
                        if not video_path.endswith(".mp4"):
                            convert_to_mp4(video_path)
                    st.rerun()

            if selected_videos["selection"]["rows"]:
                selected_idx = selected_videos["selection"]["rows"][0]
                selected_video = video_df.iloc[selected_idx]
                new_name = st.text_input("New Video Name", os.path.splitext(selected_video["Video"])[0])
                if st.button(t("video_rename")) and new_name:
                    rename_video(selected_video["Full Path"], new_name)
                    st.rerun()

            if st.button(t("video_merge")) and selected_videos["selection"]["rows"]:
                video_paths = [video_df.iloc[idx]["Full Path"] for idx in selected_videos["selection"]["rows"]]
                merge_videos(video_paths, self.working_dir)
                st.rerun()

            if st.button(t("video_delete")) and selected_videos["selection"]["rows"]:
                video_paths = [video_df.iloc[idx]["Full Path"] for idx in selected_videos["selection"]["rows"]]
                delete_videos(video_paths)
                st.rerun()

    def handle_chapters(self, col1, selected_videos, video_df):
        with col1:
            if selected_videos["selection"]["rows"]:
                selected_idx = selected_videos["selection"]["rows"][0]
                selected_video = video_df.iloc[selected_idx]
                vtt_path = os.path.splitext(selected_video["Full Path"])[0] + ".vtt"
                if selected_video["Has Subtitles"]:
                    subtitles_df, chapters_df = load_subtitles_and_chapters(vtt_path)
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
                                save_vtt(vtt_path, subtitles_df, chapters_df)
                                st.rerun()
                        if st.button(t("video_delete_chapter")) and selected_chapters["selection"]["rows"]:
                            chapter_idx = selected_chapters["selection"]["rows"][0]
                            chapters_df = chapters_df.drop(chapter_idx).reset_index(drop=True)
                            save_vtt(vtt_path, subtitles_df, chapters_df)
                            st.rerun()

    def handle_subtitles(self, col2, selected_videos, video_df, generate_thumbnails, refresh_thumbnails):
        with col2:
            if selected_videos["selection"]["rows"]:
                selected_idx = selected_videos["selection"]["rows"][0]
                selected_video = video_df.iloc[selected_idx]
                vtt_path = os.path.splitext(selected_video["Full Path"])[0] + ".vtt"

                if selected_video["Full Path"].endswith(".mp4"):
                    st.video(selected_video["Full Path"], autoplay=True)

                if selected_video["Has Subtitles"]:
                    subtitles_df, chapters_df = load_subtitles_and_chapters(vtt_path)

                    if "thumbnails" not in st.session_state:
                        st.session_state["thumbnails"] = {}
                    if refresh_thumbnails or (generate_thumbnails and not st.session_state["thumbnails"].get(vtt_path)):
                        st.session_state["thumbnails"][vtt_path] = {}
                        for i, row in subtitles_df.iterrows():
                            seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(row["Start"].split(":")[:-1]))) + float(row["Start"].split(":")[-1])
                            thumbnail = generate_thumbnail(selected_video["Full Path"], seconds)
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
                                if edited_start != selected_subtitle["Start"] and selected_subtitle_idx > 0:
                                    subtitles_df.at[selected_subtitle_idx - 1, "End"] = edited_start
                                if edited_end != selected_subtitle["End"] and selected_subtitle_idx < len(subtitles_df) - 1:
                                    subtitles_df.at[selected_subtitle_idx + 1, "Start"] = edited_end
                                subtitles_df.at[selected_subtitle_idx, "Start"] = edited_start
                                subtitles_df.at[selected_subtitle_idx, "End"] = edited_end
                                subtitles_df.at[selected_subtitle_idx, "Text"] = edited_text
                                save_vtt(vtt_path, subtitles_df, chapters_df)
                                st.rerun()

                        if st.button(t("video_split")):
                            split_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(selected_subtitle["Start"].split(":")[:-1]))) + float(selected_subtitle["Start"].split(":")[-1])
                            split_video(selected_video["Full Path"], split_time, self.working_dir)
                            st.rerun()

                        if selected_video["Full Path"].endswith(".mp4"):
                            st.video(
                                selected_video["Full Path"],
                                start_time=selected_subtitle["Start"],
                                end_time=selected_subtitle["End"],
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
                            save_vtt(vtt_path, subtitles_df, chapters_df)
                            st.rerun()
                else:
                    st.write("No subtitles available for this video yet.")
            else:
                st.write("Select a video to view subtitles.")

    def run(self, config):
        self.working_dir = config.get(self.name, {}).get("video_workdir", t("video_config_workdir_default"))

        self.setup_header()
        selected_model, generate_thumbnails, refresh_thumbnails, selected_extensions = self.setup_controls()

        video_df = scan_videos(self.working_dir, selected_extensions)
        if video_df.empty:
            st.write("No videos found with the selected extensions.")
            return

        col1, col2, selected_videos = self.display_videos(video_df)
        self.handle_video_actions(col1, selected_videos, video_df, selected_model)
        self.handle_chapters(col1, selected_videos, video_df)
        self.handle_subtitles(col2, selected_videos, video_df, generate_thumbnails, refresh_thumbnails)

if __name__ == "__main__":
    st.write("Video Plugin standalone test")
