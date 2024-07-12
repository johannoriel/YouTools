from app import Plugin, list_video_files
import streamlit as st
import os
from chromakey_background import replace_background

class ChromakeyPlugin(Plugin):
    def get_config_fields(self):
        return {
            "background_directory": {
                "type": "text",
                "label": "Répertoire des vidéos de fond",
                "default": "/home/joriel/Vidéos/Backgrounds"
            }
        }

    def get_config_ui(self, config):
        updated_config = {}
        updated_config['separator_chromakey'] = st.header('Chroma Key')
        updated_config["background_directory"] = st.text_input(
            "Répertoire des vidéos de fond",
            value=config.get("background_directory", "/home/joriel/Vidéos/Backgrounds")
        )
        return updated_config

    def get_tabs(self):
        return [{"name": "Chromakey", "plugin": "chromakey"}]

    def run(self, config):
        st.header("Remplacement du fond (Chromakey)")
        
        work_directory = config['common']['work_directory']
        background_directory = config['chromakey']['background_directory']
        
        original_files, trimed_files, _ = list_video_files(work_directory)
        video_files = original_files + trimed_files
        background_files = [f for f in os.listdir(background_directory) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        selected_video = st.selectbox("Sélectionner une vidéo à traiter", [file for file, _, _ in video_files])
        selected_background = st.selectbox("Sélectionner une vidéo de fond", background_files)
        
        if st.button("Appliquer le Chromakey"):
            if selected_video and selected_background:
                video_path = os.path.join(work_directory, selected_video)
                background_path = os.path.join(background_directory, selected_background)
                result_filename = f"chroma_{selected_video.replace('outfile_', '')}"
                result_path = os.path.join(work_directory, result_filename)
                
                with st.spinner("Traitement en cours..."):
                    try:
                        replace_background(video_path, background_path, result_path)
                        st.success(f"Traitement terminé. Fichier de sortie : {result_filename}")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Une erreur s'est produite lors du traitement : {str(e)}")
            else:
                st.warning("Veuillez sélectionner une vidéo et un fond.")
