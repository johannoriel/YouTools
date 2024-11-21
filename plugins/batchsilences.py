import os
import streamlit as st
from app import Plugin
from global_vars import translations, t
from plugins.common import list_all_video_files

# Add translations
translations["en"].update({
    "batchsilences_tab": "Batch Silence Removal",
    "batchsilences_header": "Remove Silences from Multiple Videos",
    "batchsilences_select_videos": "Select videos to process",
    "batchsilences_process_button": "Remove Silences from Selected Videos",
    "batchsilences_processing": "Processing videos...",
    "batchsilences_success": "Successfully processed {count} videos",
    "batchsilences_no_videos": "No videos selected for processing"
})

translations["fr"].update({
    "batchsilences_tab": "Suppression de Silences en Batch",
    "batchsilences_header": "Supprimer les Silences de Plusieurs Vidéos",
    "batchsilences_select_videos": "Sélectionner les vidéos à traiter",
    "batchsilences_process_button": "Supprimer les Silences des Vidéos Sélectionnées",
    "batchsilences_processing": "Traitement des vidéos en cours...",
    "batchsilences_success": "{count} vidéos traitées avec succès",
    "batchsilences_no_videos": "Aucune vidéo sélectionnée pour le traitement"
})

class BatchsilencesPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.trimsilences_plugin = self.plugin_manager.get_plugin('trimsilences')

    def get_config_fields(self):
        return {}  # No specific configuration needed

    def get_tabs(self):
        return [{"name": t("batchsilences_tab"), "plugin": "batchsilences"}]

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

        # Process button
        if st.button(t("batchsilences_process_button")):
            if not selected_videos:
                st.warning(t("batchsilences_no_videos"))
                return

            with st.spinner(t("batchsilences_processing")):
                processed_count = 0
                for video_name in selected_videos:
                    # Find the full path of the selected video
                    video_path = next(v[1] for v in video_files if v[0] == video_name)

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

                st.success(t("batchsilences_success").format(count=processed_count))
