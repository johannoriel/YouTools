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
from media_selector import media_selector

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
    "movied_reference_audio": "Reference Audio File",
    "movied_normalize_audio": "Normalize Audio",
    "movied_normalizing": "Normalizing audio...",
    "movied_add_bottom_text": "Add Bottom Text",
    "movied_insert_video_with_text": "Insert Video with Text",
    "movied_text_style_label": "Text Style",
    "movied_text_style_outline": "Outline",
    "movied_text_style_box": "Box",
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
    "movied_reference_audio": "Fichier Audio de Référence",
    "movied_normalize_audio": "Normaliser le Son",
    "movied_normalizing": "Normalisation du son en cours...",
    "movied_add_bottom_text": "Ajouter du Texte en Bas",
    "movied_insert_video_with_text": "Insérer une Vidéo avec Texte",
    "movied_text_style_label": "Style de Texte",
    "movied_text_style_outline": "Contour",
    "movied_text_style_box": "Boîte",
})


class MoviedPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.working_dir = None
        self.media_dirs = []
        self.reference_audio_path = None

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
            },
            "movied_reference_audio": {  # Nouveau champ
                "type": "text",
                "label": t("movied_reference_audio"),
                "default": "/path/to/sample.mp3"
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
            try:
                from matplotlib import font_manager
                from fontTools.ttLib import TTFont
                import os

                # Obtenir la liste des fichiers .ttf
                font_files = font_manager.findSystemFonts(
                    fontpaths=None, fontext='ttf')

                # Dictionnaire pour associer les noms de polices aux chemins
                font_dict = {}
                for font_path in font_files:
                    try:
                        # Charger le fichier .ttf avec fontTools
                        font = TTFont(font_path)
                        # Extraire le nom de la police (nameID 4 correspond au nom complet, souvent "Arial Bold")
                        font_name = None
                        for record in font['name'].names:
                            if record.nameID == 4:  # nameID 4 = nom complet de la police
                                # Gérer les encodages potentiellement problématiques
                                try:
                                    font_name = record.string.decode('utf-8')
                                except UnicodeDecodeError:
                                    font_name = record.string.decode(
                                        'latin-1', errors='ignore')
                                break
                        if font_name:
                            font_dict[font_name] = font_path
                        else:
                            # Si le nom n'est pas trouvé, utiliser le nom du fichier comme secours
                            font_name = os.path.basename(font_path)
                            font_dict[font_name] = font_path
                    except Exception as e:
                        print(
                            f"Impossible de lire la police {font_path} : {str(e)}")
                        continue

                # Liste des noms de polices pour l'affichage dans la selectbox
                font_names = sorted(font_dict.keys())
                # Liste des chemins correspondants (sera utilisée comme valeur réelle)
                font_paths = [font_dict[name] for name in font_names]

                if not font_paths:
                    raise ValueError("Aucune police trouvée.")
            except Exception as e:
                st.warning(
                    f"Impossible de lister les polices : {str(e)}. Utilisation d'une liste par défaut.")
                font_names = ["Arial", "Arial Bold", "Times New Roman",
                              "Times New Roman Bold", "Courier New", "Verdana"]
                # Dans le cas par défaut, on suppose que les noms fonctionnent directement
                font_paths = font_names

            # Trouver l'index de "Arial Bold" pour le sélectionner par défaut
            default_index = 0
            for i, name in enumerate(font_names):
                if "Arial Bold" in name:
                    default_index = i
                    break

            # Sélectionner la police (afficher le nom, mais retourner le chemin)
            font_path = st.selectbox(
                t("movied_font_label"),
                options=font_paths,  # Les valeurs sont les chemins
                format_func=lambda x: font_names[font_paths.index(
                    x)] if x in font_paths else x,  # Afficher les noms lisibles
                index=default_index,  # Sélectionner "Arial Bold" par défaut
                key="font_select"
            )
            font = font_path  # Le chemin est directement utilisé
            font_size = st.slider(
                t("movied_font_size_label"),
                50, 200, 100, step=5,
                key="font_size_slider"
            )
            text_background = st.selectbox(
                t("movied_text_background"),
                [t("movied_green_background"), t("movied_original_video")],
                index=1,
                key="text_background_select"
            )
            text_style = st.selectbox(
                t("movied_text_style_label"),
                [t("movied_text_style_outline"), t("movied_text_style_box")],
                index=0,  # "Outline" par défaut
                key="text_style_select"
            )
            return selected_model, thumbnail_size, font, font_size, text_background, text_style

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
            key="movied_selector",
            hide_index=True
        )
        return selected_video

    def handle_transcript(self, selected_video, video_df, selected_model):
        if selected_video["selection"]["rows"]:
            idx = selected_video["selection"]["rows"][0]
            video_info = video_df.iloc[idx]
            vtt_path = os.path.splitext(video_info["Full Path"])[0] + ".vtt"

            # Créer deux colonnes pour les boutons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("movied_generate_transcript")):
                    with st.spinner(t("movied_processing")):
                        try:
                            generate_subtitles(
                                video_info["Full Path"], selected_model)
                            subtitles_df, _ = load_subtitles_and_chapters(
                                vtt_path)
                            st.session_state["subtitles_df"] = subtitles_df
                            st.session_state["current_vtt_path"] = vtt_path
                            st.success(t("movied_success").format(
                                video=os.path.basename(video_info["Full Path"])))
                            st.rerun()
                        except Exception as e:
                            st.error(t("movied_error").format(error=str(e)))
                            return None, None, None

            with col2:
                if st.button(t("movied_normalize_audio")):
                    with st.spinner(t("movied_normalizing")):
                        try:
                            normalize_audio(
                                video_info["Full Path"], self.reference_audio_path)
                            st.rerun()  # Relancer pour refléter les changements
                        except Exception as e:
                            st.error(t("movied_error").format(error=str(e)))

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

    def show_media_preview(self, media_path):
        """Affiche une prévisualisation du média dans une colonne centrale (1/3 de la largeur)."""
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])  # 3 colonnes égales
        with col2:  # Colonne centrale pour la prévisualisation
            st.subheader("Preview")
            if media_path.lower().endswith(('.jpg', '.png')):
                st.image(media_path, use_container_width=True)  # Ajuste à la largeur de la colonne
            elif media_path.lower().endswith(('.mp4', '.mkv', '.avi')):
                st.video(media_path, format="video/mp4", autoplay=True)

    def handle_operations(self, start_time, end_time, video_path, vtt_path, thumbnail_size, font, font_size):
        if not video_path:
            st.warning("Please select a video to edit first.")
            return

        st.subheader("Media Selection and Operations")

        col1, col2 = st.columns(2)

        # Directory selection
        media_dir_options = [t("movied_all_directories")] + self.media_dirs
        selected_dirs = col1.multiselect(
            t("movied_filter_media_dir"),
            options=media_dir_options,
            default=[t("movied_all_directories")],
            key="media_dir_select"
        )

        # Determine which directories to pass to media_selector
        if t("movied_all_directories") in selected_dirs:
            dirs_to_scan = self.media_dirs
        else:
            dirs_to_scan = [d for d in selected_dirs if d != t("movied_all_directories")]

        if not dirs_to_scan:
            st.warning("Please select at least one directory.")
            return

        # Extension selection
        all_extensions = [".jpg", ".png", ".mp4", ".mkv", ".avi"]
        selected_extensions = col2.multiselect(
            "Filter by File Extensions",
            options=all_extensions,
            default=[".mp4", ".png", ".jpg"],
            key="extension_select"
        )

        if not selected_extensions:
            st.warning("Please select at least one file extension.")
            return

        # Single media selector for images and videos
        selected_media = media_selector(
            media_dirs=dirs_to_scan,
            extensions=selected_extensions,
            suffix="movied"
        )

        # Prévisualisation si un média est sélectionné
        if selected_media:
            self.show_media_preview(selected_media)

        # Determine media type
        is_image = selected_media and any(selected_media.lower().endswith(ext) for ext in [".jpg", ".png"])
        is_video = selected_media and any(selected_media.lower().endswith(ext) for ext in [".mp4", ".mkv", ".avi"])
        has_media = bool(selected_media)

        # Text input for operations that need it
        st.write(t("movied_text_operations"))
        text_input = st.text_input(
            t("movied_text_input"),
            key="text_input"
        )

        # All operations in a single row below text input
        st.write("Available Operations:")
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

        with col1:
            if st.button(t("movied_replace_image"), key="replace_image_btn", disabled=not is_image):
                if has_media:
                    operation = f"replace_image {start_time} {end_time} {selected_media}"
                    self.add_to_operations(operation)
                else:
                    st.warning("Please select an image.")

        with col2:
            if st.button(t("movied_insert_video"), key="insert_video_btn", disabled=not is_video):
                if has_media:
                    operation = f"insert_video {start_time} {selected_media}"
                    self.add_to_operations(operation)
                else:
                    st.warning("Please select a video.")

        with col3:
            if st.button(t("movied_replace_video"), key="replace_video_btn", disabled=not is_video):
                if has_media:
                    operation = f"replace_video {start_time} {end_time} {selected_media}"
                    self.add_to_operations(operation)
                else:
                    st.warning("Please select a video.")

        with col4:
            if st.button(t("movied_replace_video_keep_audio"), key="replace_video_keep_audio_btn", disabled=not is_video):
                if has_media:
                    operation = f"replace_video_keep_audio {start_time} {end_time} {selected_media}"
                    self.add_to_operations(operation)
                else:
                    st.warning("Please select a video.")

        with col5:
            if st.button(t("movied_animate_text"), key="animate_text_btn", disabled=not text_input):
                if text_input:
                    text_command = text_input.replace("\n", "\\")
                    operation = f"addtext {start_time} {end_time} fromLeft 1s {text_command}"
                    self.add_to_operations(operation)
                else:
                    st.warning("Please enter text first.")

        with col6:
            if st.button(t("movied_add_bottom_text"), key="add_bottom_text_btn", disabled=not text_input):
                if text_input:
                    text_command = text_input.replace("\n", "\\")
                    operation = f"addBottomText {start_time} {end_time} fromLeft 1s {text_command}"
                    self.add_to_operations(operation)
                else:
                    st.warning("Please enter text first.")

        with col7:
            if st.button(t("movied_insert_video_with_text"), key="insert_video_with_text_btn", disabled=not (is_video and text_input)):
                if has_media and text_input:
                    text_command = text_input.replace("\n", "\\")
                    operation = f"insertVideoWithText {start_time} {selected_media} | {text_command}"
                    self.add_to_operations(operation)
                else:
                    st.warning("Please select a video and enter text first.")

        with col8:
            if st.button(t("movied_remove_section"), key="remove_section_btn", disabled=not (start_time and end_time)):
                operation = f"remove_section {start_time} {end_time}"
                self.add_to_operations(operation)

        # Operations queue
        operations = st.text_area(t("movied_operations"), value=st.session_state.get("operations", ""), key="operations_area")
        st.session_state["operations"] = operations

        # Generate button
        if st.button(t("movied_generate"), key="generate_btn", type="primary") and operations:
            self.execute_operations(video_path, vtt_path, operations, font, font_size)

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

    def alert(self):
        notification_js = """
        <script>
        function sendBrowserNotification() {
            // Demande la permission si nécessaire
            if ("Notification" in window) {
                Notification.requestPermission().then(function (permission) {
                    if (permission === "granted") {
                        new Notification("Alerte Streamlit", {
                            body: "Génération terminée",
                            icon: "https://streamlit.io/favicon.ico"
                        });
                    }
                });
            } else {
                alert("Votre navigateur ne supporte pas les notifications desktop.");
            }
        }
        sendBrowserNotification();
        </script>
        """
        st.components.v1.html(notification_js)

    def execute_operations(self, video_path, vtt_path, operations, font, font_size):
        with st.spinner("Processing video operations..."):
            try:
                from video_utils import (replace_with_image, insert_video,
                                         replace_with_video, replace_video_keep_audio,
                                         add_animated_text, remove_section)

                main_clip = VideoFileClip(video_path)
                target_size = (main_clip.w, main_clip.h)
                subtitles_df, _ = load_subtitles_and_chapters(vtt_path)
                duration_offset = 0
                operation_log = []  # Liste pour stocker les opérations avec nature séparée

                text_background = st.session_state.get(
                    "text_background_select", t("movied_green_background"))
                use_green_background = text_background == t(
                    "movied_green_background")
                text_style = st.session_state.get(
                    "text_style_select", t("movied_text_style_outline"))
                text_style = "outline" if text_style == t(
                    "movied_text_style_outline") else "box"

                with st.expander("Debug Information"):
                    for op in operations.split("\n"):
                        if not op.strip():
                            continue
                        op_cleaned = op.split("//")[0].strip()
                        if not op_cleaned:
                            continue
                        parts = op_cleaned.split(maxsplit=5)
                        cmd = parts[0]
                        st.write(f"Processing: {op_cleaned}")

                        if cmd == "insert_video" or cmd == "insertVideoWithText":
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

                        real_start = self.format_timecode(start_sec)
                        real_end = self.format_timecode(
                            end_sec) if end_sec else None

                        if cmd == "replace_image":
                            image_path = remaining_args
                            main_clip = replace_with_image(
                                main_clip, start_sec, end_sec, image_path, target_size)
                            operation_log.append(
                                {"Nature": "replace_image", "Details": image_path, "Start": real_start, "End": real_end})

                        elif cmd == "insert_video":
                            video_path_insert = remaining_args
                            main_clip, duration_change = insert_video(
                                main_clip, start_sec, video_path_insert, target_size)
                            subtitles_df = self.adjust_subtitles(
                                subtitles_df, start_time, duration_change)
                            duration_offset += duration_change
                            operation_log.append(
                                {"Nature": "insert_video", "Details": video_path_insert, "Start": real_start, "End": None})

                        elif cmd == "insertVideoWithText":
                            # Séparer le timecode, puis utiliser | pour distinguer chemin et texte
                            start_time = parts[1]
                            rest = " ".join(parts[2:])
                            try:
                                video_path_insert, text = rest.split(
                                    " | ", 1)  # Séparer sur " | "
                            except ValueError:
                                raise ValueError(
                                    f"Invalid format for insertVideoWithText: {op_cleaned}. Use 'timecode path | text'")
                            start_sec = self.parse_timecode(
                                start_time) + duration_offset
                            real_start = self.format_timecode(start_sec)
                            main_clip, duration_change = insert_video_with_text(
                                main_clip, start_sec, video_path_insert, text, target_size, font, font_size, use_green_background, text_style)
                            subtitles_df = self.adjust_subtitles(
                                subtitles_df, start_time, duration_change)
                            duration_offset += duration_change
                            operation_log.append(
                                {"Nature": "insertVideoWithText", "Details": f"{video_path_insert} | {text}", "Start": real_start, "End": None})

                        elif cmd == "replace_video":
                            video_path_replace = remaining_args
                            main_clip, duration_change = replace_with_video(
                                main_clip, start_sec, end_sec, video_path_replace, target_size)
                            subtitles_df = self.adjust_subtitles(
                                subtitles_df, end_time, duration_change)
                            duration_offset += duration_change
                            operation_log.append(
                                {"Nature": "replace_video", "Details": video_path_replace, "Start": real_start, "End": real_end})

                        elif cmd == "replace_video_keep_audio":
                            video_path_replace = remaining_args
                            main_clip = replace_video_keep_audio(
                                main_clip, start_sec, end_sec, video_path_replace, target_size)
                            operation_log.append(
                                {"Nature": "replace_video_keep_audio", "Details": video_path_replace, "Start": real_start, "End": real_end})

                        elif cmd == "addtext":
                            animation_type, anim_duration, text = parts[3], parts[4], parts[5]
                            anim_duration_sec = float(anim_duration[:-1])
                            main_clip = add_animated_text(main_clip, start_sec, end_sec, text, animation_type,
                                                          anim_duration_sec, target_size, font, font_size,
                                                          use_green_background=use_green_background, position="center", text_style=text_style)
                            operation_log.append(
                                {"Nature": "addtext", "Details": f"{animation_type} {anim_duration} {text}", "Start": real_start, "End": real_end})

                        elif cmd == "addBottomText":
                            animation_type, anim_duration, text = parts[3], parts[4], parts[5]
                            anim_duration_sec = float(anim_duration[:-1])
                            main_clip = add_animated_text(main_clip, start_sec, end_sec, text, animation_type,
                                                          anim_duration_sec, target_size, font, font_size,
                                                          use_green_background=use_green_background, position="bottom", text_style=text_style)
                            operation_log.append(
                                {"Nature": "addBottomText", "Details": f"{animation_type} {anim_duration} {text}", "Start": real_start, "End": real_end})

                        elif cmd == "remove_section":
                            main_clip, duration_change = remove_section(
                                main_clip, start_sec, end_sec)
                            subtitles_df = self.adjust_subtitles(
                                subtitles_df, start_time, duration_change)
                            duration_offset += duration_change
                            operation_log.append(
                                {"Nature": "remove_section", "Details": "", "Start": real_start, "End": real_end})

                        else:
                            raise ValueError(f"Unknown command: {cmd}")

                output_path = os.path.splitext(video_path)[0] + "_edited.mp4"
                main_clip.write_videofile(
                    output_path, codec="libx264", audio_codec="aac")
                save_vtt(vtt_path, subtitles_df, pd.DataFrame())
                st.success(f"Video generated successfully at {output_path}")

                # Créer un DataFrame avec les opérations et timecodes réels
                operations_df = pd.DataFrame(operation_log, columns=[
                                             "Nature", "Details", "Start", "End"])
                st.session_state["operations_log"] = operations_df
                st.session_state["generated_video_path"] = output_path

                st.rerun()
            except Exception as e:
                st.error(t("movied_error").format(error=str(e)))
                raise e
            finally:
                main_clip.close()
                self.alert()

    def format_timecode(self, seconds):
        """Formate les secondes en timecode HH:MM:SS.mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def run(self, config):
        self.working_dir = config.get(self.name, {}).get(
            "movied_workdir", t("movied_workdir_default"))
        self.media_dirs = config.get(self.name, {}).get(
            "movied_media_dirs", t("movied_media_dirs_default")).split("\n")
        self.reference_audio_path = config.get(self.name, {}).get(
            "movied_reference_audio", "/path/to/sample.mp3")

        self.setup_header()
        selected_model, thumbnail_size, font, font_size, text_background, text_style = self.setup_controls()

        video_df = self.list_videos()
        if video_df.empty:
            st.write("No videos found in the working directory.")
            return

        selected_video = self.display_videos(video_df)
        selected_subtitles, subtitles_df, vtt_path = self.handle_transcript(
            selected_video, video_df, selected_model)
        start_time, end_time = self.handle_section(
            selected_subtitles, subtitles_df)

        # Toujours afficher les opérations, même sans sous-titres sélectionnés
        if selected_video["selection"]["rows"]:
            video_path = video_df.iloc[selected_video["selection"]
                                       ["rows"][0]]["Full Path"]
            self.handle_operations(
                start_time, end_time, video_path, vtt_path, thumbnail_size, font, font_size)
        else:
            self.handle_operations(
                None, None, None, None, thumbnail_size, font, font_size)

        # Afficher les résultats stockés dans la session après un rerun
        if "operations_log" in st.session_state:
            st.write("Operations with Real Timecodes:")
            # Activer la sélection d'une ligne dans le DataFrame
            selected_operation = st.dataframe(
                st.session_state["operations_log"],
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="operations_log_selector"
            )

            # Si une ligne est sélectionnée, récupérer le timecode de début
            start_time_seconds = None
            if selected_operation["selection"]["rows"]:
                selected_row = selected_operation["selection"]["rows"][0]
                selected_timecode = st.session_state["operations_log"].iloc[selected_row]["Start"]
                # Convertir le timecode (HH:MM:SS.mmm) en secondes
                h, m, s = map(float, selected_timecode.replace(
                    ",", ".").split(":"))
                start_time_seconds = h * 3600 + m * 60 + s

        if "generated_video_path" in st.session_state:
            st.write("Generated Video:")
            _, col, _ = st.columns(3)
            # Passer start_time à st.video si une ligne est sélectionnée
            col.video(
                st.session_state["generated_video_path"],
                start_time=start_time_seconds if start_time_seconds is not None else 0,
                autoplay=True,
            )


if __name__ == "__main__":
    st.write("Movied Plugin standalone test")
