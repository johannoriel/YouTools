from lib.global_vars import translations, t
from app import Plugin
from plugins.common import list_video_files
import streamlit as st
import os
from lib.chromakey_background import replace_background
import cv2

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
    "chromakey_warning_message": "Please select a video and a background.",
    "default_background_label": "Default Background",
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
    "chromakey_warning_message": "Veuillez sélectionner une vidéo et un fond.",
    "default_background_label": "Fond par défaut",
})


class ChromakeyPlugin(Plugin):
    def get_config_fields(self):
        return {
            "background_directory": {
                "type": "text",
                "label": t("chromakey_background_dir_label"),
                "default": "/home/joriel/Vidéos/Background"
            },
            "default_target_color": {  # Nouveau champ pour la couleur par défaut
                "type": "text",
                "label": "Couleur cible par défaut (format hexadécimal)",
                "default": "#00FF00"  # Vert par défaut
            },
            "default_background": {  # Nouveau champ pour le fond par défaut
                "type": "select",
                "label": t("default_background_label"),
                "default": ""  # Vide par défaut, sera rempli dynamiquement
            }
        }

    def get_config_ui(self, config):
        updated_config = {}
        updated_config["background_directory"] = st.text_input(
            t("chromakey_background_dir_label"),
            value=config.get("background_directory",
                             "/home/joriel/Vidéos/Backgrounds")
        )
        updated_config["default_target_color"] = st.text_input(
            "Couleur cible par défaut (format hexadécimal)",
            value=config.get("default_target_color", "#00FF00")
        )

        # Récupérer la liste des fichiers vidéo dans background_directory
        background_directory = updated_config["background_directory"]
        background_files = []
        if os.path.exists(background_directory):
            background_files = [f for f in os.listdir(background_directory)
                               if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            background_files.insert(0, "")  # Ajouter une option vide

        # Sélecteur pour le fond par défaut
        updated_config["default_background"] = st.selectbox(
            "Fond par défaut",
            options=background_files,
            index=background_files.index(config.get("default_background", ""))
            if config.get("default_background", "") in background_files else 0
        )

        return updated_config

    def get_tabs(self):
        return [{"name": "Chromakey", "plugin": "chromakey"}]

    def run(self, config):
        st.header(t("chromakey_title"))

        work_directory = config['common']['work_directory']
        background_directory = config['chromakey']['background_directory']
        default_background = config['chromakey'].get("default_background", "")  # Récupérer le fond par défaut

        original_files, trimed_files, _, _ = list_video_files(work_directory)
        video_files = original_files + trimed_files
        background_files = [f for f in os.listdir(
            background_directory) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        selected_background = st.selectbox(
            t("chromakey_select_background_label"),
            background_files,
            index=background_files.index(default_background) if default_background in background_files else 0
        )

        selected_video = st.selectbox(t("chromakey_select_video_label"), [
                                      file for file, _, _ in video_files])

        # Extraire la première image de la vidéo sélectionnée pour la prévisualisation
        video_path = os.path.join(work_directory, selected_video)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            # Convertir l'image de BGR (OpenCV) à RGB pour l'affichage dans Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Première image de la vidéo",
                     use_container_width=True)
        cap.release()

        # Sélecteur de couleur avec la valeur par défaut de la configuration
        default_target_color = config['chromakey']["default_target_color"]
        target_color_rgb = st.color_picker(
            "Choisissez la couleur du fond à remplacer",
            default_target_color  # Utiliser la couleur par défaut de la configuration
        )
        st.write(
            f"Valeur hexadécimale de la couleur sélectionnée : `{target_color_rgb}`")

        # Convertir la couleur hexadécimale en RGB
        target_color_rgb = [int(target_color_rgb.lstrip('#')[
                                i:i+2], 16) for i in (0, 2, 4)]
        exact = st.checkbox("Exact chromakey", value=True)

        if st.button(t("chromakey_apply_button")):
            if selected_video and selected_background:
                background_path = os.path.join(
                    background_directory, selected_background)
                result_filename = f"chroma_{selected_video.replace('outfile_', '')}"
                result_path = os.path.join(work_directory, result_filename)

                with st.spinner(t("chromakey_processing_spinner")):
                    try:
                        # Passer la couleur cible à la fonction replace_background
                        replace_background(video_path, background_path, result_path, target_color_rgb, exact_color=exact)
                        st.success(
                            f"{t('chromakey_success_message')}{result_filename}")
                        st.video(result_path, autoplay=True, muted=True)
                    except Exception as e:
                        st.error(f"{t('chromakey_error_message')}{str(e)}")
            else:
                st.warning(t("chromakey_warning_message"))
