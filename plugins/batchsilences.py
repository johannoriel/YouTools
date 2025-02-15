import os
import streamlit as st
import pandas as pd
from app import Plugin
from global_vars import translations, t
from plugins.common import list_all_video_files
from moviepy.editor import VideoFileClip, concatenate_videoclips
from typing import List

# Add translations
translations["en"].update({
    "batchsilences_tab": "Batch Silence Removal",
    "batchsilences_header": "Remove Silences from Multiple Videos",
    "batchsilences_select_videos": "Select videos to process",
    "batchsilences_process_button": "Remove Silences from Selected Videos",
    "batchsilences_merge_button": "Merge Selected Videos",
    "batchsilences_process_and_merge_button": "Remove Silences and Merge Selected Videos",
    "batchsilences_processing": "Processing videos...",
    "batchsilences_success": "Successfully processed {count} videos",
    "batchsilences_no_videos": "No videos selected for processing",
    "batchsilences_merge_success": "Successfully merged {count} videos",
    "batchsilences_merge_error": "Error merging videos: {error}",
    "batchsilences_reorder_videos": "Reorder Videos",
    "batchsilences_reorder_instructions": "Drag and drop to reorder the videos for merging."
})

translations["fr"].update({
    "batchsilences_tab": "Suppression de Silences en Batch",
    "batchsilences_header": "Supprimer les Silences de Plusieurs Vidéos",
    "batchsilences_select_videos": "Sélectionner les vidéos à traiter",
    "batchsilences_process_button": "Supprimer les Silences des Vidéos Sélectionnées",
    "batchsilences_merge_button": "Fusionner les Vidéos Sélectionnées",
    "batchsilences_process_and_merge_button": "Supprimer les Silences et Fusionner les Vidéos Sélectionnées",
    "batchsilences_processing": "Traitement des vidéos en cours...",
    "batchsilences_success": "{count} vidéos traitées avec succès",
    "batchsilences_no_videos": "Aucune vidéo sélectionnée pour le traitement",
    "batchsilences_merge_success": "{count} vidéos fusionnées avec succès",
    "batchsilences_merge_error": "Erreur lors de la fusion des vidéos : {error}",
    "batchsilences_reorder_videos": "Réorganiser les Vidéos",
    "batchsilences_reorder_instructions": "Glissez-déposez pour réorganiser l'ordre des vidéos avant la fusion."
})


class BatchsilencesPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.trimsilences_plugin = self.plugin_manager.get_plugin(
            'trimsilences')

    def get_config_fields(self):
        return {}  # No specific configuration needed

    def get_tabs(self):
        return [{"name": t("batchsilences_tab"), "plugin": "batchsilences"}]

    def merge_videos(self, video_paths: List[str], output_path: str) -> str:
        """
        Fusionne plusieurs vidéos en une seule.

        Args:
            video_paths: Liste des chemins des vidéos à fusionner
            output_path: Chemin de sortie pour la vidéo fusionnée

        Returns:
            Chemin de la vidéo fusionnée ou message d'erreur
        """
        try:
            clips = [VideoFileClip(path) for path in video_paths]
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path, codec="libx264")
            return output_path
        except Exception as e:
            return t("batchsilences_merge_error").format(error=str(e))

    def run(self, config):
        st.header(t("batchsilences_header"))

        # Get video files
        work_directory = config['common']['work_directory']
        video_files = list_all_video_files(work_directory)

        if not video_files:
            st.warning(t("transcript_no_videos"))
            return

        # Allow multiple video selection
        selected_videos = st.multiselect(
            t("batchsilences_select_videos"),
            options=[v[0] for v in video_files]
        )

        # Reorder videos using a data editor
        if selected_videos:
            st.subheader(t("batchsilences_reorder_videos"))
            st.markdown(t("batchsilences_reorder_instructions"))

            # Create a DataFrame for reordering
            video_df = pd.DataFrame({
                "Video Name": selected_videos,
                "Order": range(1, len(selected_videos) + 1)
            })

            # Allow reordering
            edited_df = st.data_editor(
                video_df,
                key="video_order_editor",
                num_rows="dynamic"
            )

            # Sort videos based on the new order
            edited_df = edited_df.sort_values(by="Order")
            ordered_video_names = edited_df["Video Name"].tolist()

            # Get full paths of the ordered videos
            ordered_video_paths = [
                next(v[1] for v in video_files if v[0] == name) for name in ordered_video_names]
        else:
            ordered_video_paths = []

        # Process button
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(t("batchsilences_process_button")):
                if not selected_videos:
                    st.warning(t("batchsilences_no_videos"))
                    return

                with st.spinner(t("batchsilences_processing")):
                    processed_count = 0
                    for video_name in selected_videos:
                        # Find the full path of the selected video
                        video_path = next(
                            v[1] for v in video_files if v[0] == video_name)

                        # Remove silence
                        result = self.trimsilences_plugin.remove_silence(
                            video_path,
                            config['trimsilences']['silence_threshold'],
                            config['trimsilences']['silence_duration'],
                            work_directory
                        )

                        # Check if processing was successful
                        if isinstance(result, str) and not (result.startswith("Erreur") or result.startswith("Une erreur")):
                            processed_count += 1
                            st.text(f"Processed: {video_name}")

                    st.success(t("batchsilences_success").format(
                        count=processed_count))

        with col2:
            if st.button(t("batchsilences_process_and_merge_button")):
                if not selected_videos:
                    st.warning(t("batchsilences_no_videos"))
                    return

                with st.spinner(t("batchsilences_processing")):
                    processed_count = 0
                    processed_videos = []
                    for video_name in selected_videos:
                        # Find the full path of the selected video
                        video_path = next(
                            v[1] for v in video_files if v[0] == video_name)

                        # Remove silence
                        result = self.trimsilences_plugin.remove_silence(
                            video_path,
                            config['trimsilences']['silence_threshold'],
                            config['trimsilences']['silence_duration'],
                            work_directory
                        )

                        # Check if processing was successful
                        if isinstance(result, str) and not (result.startswith("Erreur") or result.startswith("Une erreur")):
                            processed_count += 1
                            processed_videos.append(result)
                            st.text(f"Processed: {video_name}")

                    if processed_videos:
                        # Merge processed videos in the specified order
                        output_path = os.path.join(
                            work_directory, "merged_video.mp4")
                        merge_result = self.merge_videos(
                            processed_videos, output_path)
                        if not merge_result.startswith(t("batchsilences_merge_error")):
                            st.success(t("batchsilences_merge_success").format(
                                count=processed_count))
                            st.success(
                                f"Merged video saved at: {merge_result}")
                        else:
                            st.error(merge_result)
                    else:
                        st.warning("No videos were processed successfully.")

        with col3:
            if st.button(t("batchsilences_merge_button")):
                if not selected_videos:
                    st.warning(t("batchsilences_no_videos"))
                    return

                with st.spinner(t("batchsilences_processing")):
                    # Use the ordered video paths for merging
                    output_path = os.path.join(
                        work_directory, "merged_video.mp4")
                    merge_result = self.merge_videos(
                        ordered_video_paths, output_path)
                    if not merge_result.startswith(t("batchsilences_merge_error")):
                        st.success(t("batchsilences_merge_success").format(
                            count=len(selected_videos)))
                        st.success(f"Merged video saved at: {merge_result}")
                    else:
                        st.error(merge_result)
