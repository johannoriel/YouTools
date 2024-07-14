from global_vars import translations, t
from app import Plugin
from plugins.common import list_video_files
import streamlit as st
import os
from chromakey_background import replace_background

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "chromakey_title": "Background Replacement (Chromakey)",
    "chromakey_background_dir_label": "Background video directory",
    "chromakey_select_video_label": "Select a video to process",
    "chromakey_select_background_label": "Select a background video",
    "chromakey_apply_button": "Apply Chromakey",
    "chromakey_processing_spinner": "Processing...",
    "chromakey_success_message": "Processing completed. Output file: ",
    "chromakey_error_message": "An error occurred during processing: ",
    "chromakey_warning_message": "Please select a video and a background."
})
translations["fr"].update({
    "chromakey_title": "Remplacement du fond (Chromakey)",
    "chromakey_background_dir_label": "Répertoire des vidéos de fond",
    "chromakey_select_video_label": "Sélectionner une vidéo à traiter",
    "chromakey_select_background_label": "Sélectionner une vidéo de fond",
    "chromakey_apply_button": "Appliquer le Chromakey",
    "chromakey_processing_spinner": "Traitement en cours...",
    "chromakey_success_message": "Traitement terminé. Fichier de sortie : ",
    "chromakey_error_message": "Une erreur s'est produite lors du traitement : ",
    "chromakey_warning_message": "Veuillez sélectionner une vidéo et un fond."
})

class ChromakeyPlugin(Plugin):
    def get_config_fields(self):
        return {
            "background_directory": {
                "type": "text",
                "label": t("chromakey_background_dir_label"),
                "default": "/home/joriel/Vidéos/Background"
            }
        }

    def get_config_ui(self, config):
        updated_config = {}
        updated_config["background_directory"] = st.text_input(
            t("chromakey_background_dir_label"),
            value=config.get("background_directory", "/home/joriel/Vidéos/Backgrounds")
        )
        return updated_config

    def get_tabs(self):
        return [{"name": "Chromakey", "plugin": "chromakey"}]

    def run(self, config):
        st.header(t("chromakey_title"))

        work_directory = config['common']['work_directory']
        background_directory = config['chromakey']['background_directory']

        original_files, trimed_files, _ = list_video_files(work_directory)
        video_files = original_files + trimed_files
        background_files = [f for f in os.listdir(background_directory) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        selected_video = st.selectbox(t("chromakey_select_video_label"), [file for file, _, _ in video_files])
        selected_background = st.selectbox(t("chromakey_select_background_label"), background_files)

        if st.button(t("chromakey_apply_button")):
            if selected_video and selected_background:
                video_path = os.path.join(work_directory, selected_video)
                background_path = os.path.join(background_directory, selected_background)
                result_filename = f"chroma_{selected_video.replace('outfile_', '')}"
                result_path = os.path.join(work_directory, result_filename)

                with st.spinner(t("chromakey_processing_spinner")):
                    try:
                        replace_background(video_path, background_path, result_path)
                        st.success(f"{t('chromakey_success_message')}{result_filename}")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"{t('chromakey_error_message')}{str(e)}")
            else:
                st.warning(t("chromakey_warning_message"))
