from tkinter.constants import VERTICAL
from tarfile import version
from enum import verify
import base64
from lib.global_vars import translations, t, alert
from app import Plugin
import streamlit as st
import pandas as pd
import os
from lib.video_utils import load_subtitles_and_chapters
import json
from moviepy import VideoFileClip, concatenate_videoclips
from widgets.media_selector import media_selector, remote_media_selector, ALL_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS
from datetime import datetime
import glob
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
from lib.movied_commands import CommandOrchestrator

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
    "movied_insert_video": "Insert Video before",
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
    "movied_add_to_interest": "Add",
    "movied_remove_from_interest": "Remove",
    "movied_merge_subtitles": "Merge",
    "movied_multiple_selection": "Multiple Selection",
    "movied_category": "Category",
    "movied_complement": "Complement",
    "movied_merge_error_not_continuous": "Cannot merge: Selected subtitles are not continuous.",
    "movied_force_add": "Forced Add",
    "movied_edit_subtitles": "Edit Subtitles",
    "movied_final_selection": "Final Selection",
    "movied_refresh": "Refresh",
    "movied_sort_subtitles": "Sort",
    "movied_export": "Export",
    "movied_import": "Import",
    "movied_verify": "Verification",
    "movied_import_last": "Import Last",
    "movied_overlap_warning": "Warning: Operations overlap between {start1} - {end1} and {start2} - {end2}",
    "movied_all_ok": "All operations are OK",
    "movied_filter_videos": "Video Filters",
    "movied_extensions": "Extensions",
    "movied_filters": "Filters",
    "movied_mp4": "MP4",
    "movied_mkv": "MKV",
    "movied_exclude_edited": "Exclude _edited",
    "movied_exclude_chroma": "Exclude chroma_*",
    "movied_only_chroma": "Only chroma_*",
    "movied_apply_filters": "Apply Filters",
    "movied_format_column": "Format",
    "movied_transcript_column": "Transcript",
    "movied_suggestions": "Suggestions",
    "movied_block_add": "Block Add",
    "movied_replace_audio": "Replace Audio",
    "movied_insert_audio": "Insert Audio",
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
    "movied_insert_video": "Insérer une vidéo avant",
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
    "movied_add_to_interest": "Ajouter",
    "movied_remove_from_interest": "Supprimer",
    "movied_merge_subtitles": "Fusionner",
    "movied_multiple_selection": "Sélection Multiple",
    "movied_category": "Catégorie",
    "movied_complement": "Complément",
    "movied_merge_error_not_continuous": "Impossible de fusionner : Les sous-titres sélectionnés ne sont pas continus.",
    "movied_force_add": "Ajout Forcé",
    "movied_edit_subtitles": "Éditer les Sous-titres",
    "movied_final_selection": "Sélection Finale",
    "movied_refresh": "Rafraîchir",
    "movied_sort_subtitles": "Trier",
    "movied_export": "Exporter",
    "movied_import": "Importer",
    "movied_verify": "Vérification",
    "movied_import_last": "Importer le Dernier",
    "movied_overlap_warning": "Attention : Les opérations se chevauchent entre {start1} - {end1} et {start2} - {end2}",
    "movied_all_ok": "Tout est OK",
    "movied_filter_videos": "Filtres vidéos",
    "movied_extensions": "Extensions",
    "movied_filters": "Filtres",
    "movied_mp4": "MP4",
    "movied_mkv": "MKV",
    "movied_exclude_edited": "Exclure _edited",
    "movied_exclude_chroma": "Exclure chroma_*",
    "movied_only_chroma": "Uniquement chroma_*",
    "movied_apply_filters": "Appliquer les filtres",
    "movied_format_column": "Format",
    "movied_transcript_column": "Transcription",
    "movied_suggestions": "Suggestions",
    "movied_block_add": "Ajout bloc",
    "movied_replace_audio": "Remplacer l'Audio",
    "movied_insert_audio": "Insérer un Audio",
})


def time_to_seconds(time_str):
    h, m, s = map(float, time_str.replace(",", ".").split(":"))
    return h * 3600 + m * 60 + s


def time_to_milliseconds(time_str):
    h, m, s = map(float, time_str.replace(",", ".").split(":"))
    return int((h * 3600 + m * 60 + s) * 1000)


class MoviedPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.working_dir = plugin_manager.config.get(self.name, {}).get(
            "movied_workdir", t("movied_workdir_default"))
        self.media_dirs = []
        self.reference_audio_path = None
        self.orchestrator = CommandOrchestrator()

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
            "movied_reference_audio": {
                "type": "text",
                "label": t("movied_reference_audio"),
                "default": "/path/to/sample.mp3"
            },
            "illustration_prompt": {
                "type": "textarea",
                "label": "Prompt pour suggestions d'illustrations",
                "default": """
                    Analyse la transcription suivante d'une vidéo et identifie les sections qui bénéficieraient d'illustrations visuelles.
                    Pour chaque section, fournis :
                    1. Le timecode de début (format : HH:MM:SS.sss)
                    2. Un seul mot décrivant le thème de l'illustration

                    Retourne ta réponse dans ce format exact, une suggestion par ligne :
                    [ILLUSTRATION] HH:MM:SS.sss thème

                    Voici la transcription :
                    {transcript}
                """
            },
            "meme_prompt": {
                "type": "textarea",
                "label": "Prompt pour suggestions de mèmes",
                "default": """
                    Analyse la transcription suivante d'une vidéo et identifie les sections avec des émotions fortes adaptées à des mèmes.
                    Pour chaque section, fournis :
                    1. Le timecode de début (format : HH:MM:SS.sss)
                    2. Un seul mot décrivant l'émotion principale

                    Retourne ta réponse dans ce format exact, une suggestion par ligne :
                    [MEME] HH:MM:SS.sss émotion

                    Voici la transcription :
                    {transcript}
                """
            },
            "edit_suggestion_prompt": {
                "type": "textarea",
                "label": "Prompt pour suggestions d'édition (étape 3)",
                "default": """
                    Analyse la ligne suivante d'une transcription vidéo et propose :
                    - Si aucune catégorie n'est fournie ("{category}" est vide) : une catégorie ('illustration', 'meme', ou 'texte') et des compléments (mots séparés par des virgules)
                    - Si une catégorie est fournie ("{category}") : des compléments (mots séparés par des virgules) adaptés à la catégorie

                    Entrée : {start} - {end} - "{text}" - Catégorie actuelle : "{category}" - Compléments actuels : "{complement}"

                    Retourne ta réponse dans ce format exact :
                    - Avec catégorie vide : [SUGGESTION] catégorie complément1, complément2, ...
                    - Avec catégorie remplie : [SUGGESTION] complément1, complément2, ...

                    Ne réponds qu'une seule ligne par suggestion.
                """
            },
            "preview_buffer_seconds": {
                "type": "number",
                "label": "Preview Buffer (seconds)",
                "default": 3
            }
        }

    def get_tabs(self):
        return [{"name": t("movied_tab"), "plugin": "movied"}]

    def setup_header(self):
        st.header(t("movied_header"))

    def refresh_grid_key(self):
        operations_hash = hash(st.session_state.get(
            "operations", "")) if "operations" in st.session_state else 0
        grid_key = f"subtitles_grid_{operations_hash}"
        st.session_state.grid_key = grid_key

    def setup_controls(self):
        with st.sidebar.expander("Options"):
            selected_model = st.selectbox(t("movied_model_label"), [
                                          "base", "medium", "turbo", "large-v3", "large-v3-turbo"], index=4)
            thumbnail_size = st.selectbox(
                "Thumbnail Size",
                ["small", "medium", "large"],
                index=1,
                key="thumbnail_size"
            )
            preview_buffer = st.number_input(
                "Preview Buffer (seconds)",
                min_value=0.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                key="preview_buffer"
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
                        # print(f"Impossible de lire la police {font_path} : {str(e)}")
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
            return selected_model, thumbnail_size, font, font_size, text_background, text_style, preview_buffer

    def setup_grid_options(self, df):
        """Configure les options de la grille AgGrid"""
        grid_options = {
            "defaultColDef": {
                "filter": True,
                "sortable": True,
                "editable": False,
            },
            "columnDefs": [
                {"field": "Start", "headerName": "Start", "width": 110,
                    "editable": False, "checkboxSelection": True},
                {"field": "End", "headerName": "End",
                    "width": 100, "editable": False},
                {"field": "Text", "headerName": "Text", "flex": 3, "editable": False,
                    "tooltipValueGetter": JsCode("""function(p) {return p.value}"""),
                    "headerTooltip": "Tooltip for caption",
                 },
                {
                    "field": "Category",
                    "width": 100,
                    "headerName": "Category",
                    "editable": True,
                    "cellEditor": "agSelectCellEditor",
                    "cellEditorParams": {"values": ["", "illustration", "meme", "texte"]},
                },
                {"field": "Complement", "width": 150,
                    "headerName": "Complement", "editable": True},
                {"field": "Operation", "flex": 2, "headerName": "Operation", "editable": True,
                    "tooltipValueGetter": JsCode(
                        """function(p) {return p.value}"""
                    ),
                    "headerTooltip": "Tooltip for Operations",
                 },
            ],
            "rowSelection": "multiple",
            "tooltipShowDelay": 100,
        }
        return grid_options

    def list_videos(self):
        with st.sidebar.expander(t("movied_filter_videos")):
            st.markdown(f"**{t('movied_extensions')}:**")
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox(
                    t("movied_mp4"),
                    value=True,
                    key="filter_mp4",
                    help=t("movied_mp4")
                )
            with col2:
                st.checkbox(
                    t("movied_mkv"),
                    value=True,
                    key="filter_mkv",
                    help=t("movied_mkv")
                )

            st.markdown(f"**{t('movied_filters')}:**")
            st.checkbox(
                t("movied_exclude_edited"),
                value=True,
                key="exclude_edited",
                help=t("movied_exclude_edited")
            )
            st.checkbox(
                t("movied_exclude_chroma"),
                value=False,
                key="exclude_chroma",
                help=t("movied_exclude_chroma")
            )
            st.checkbox(
                t("movied_only_chroma"),
                value=False,
                key="show_only_chroma",
                help=t("movied_only_chroma")
            )

            if st.button(t("movied_apply_filters")):
                st.rerun()

        video_extensions = [".mp4", ".mkv", ".avi"]
        videos = []

        # Récupérer les états des filtres depuis session_state
        exclude_edited = st.session_state.get("exclude_edited", True)
        exclude_chroma = st.session_state.get("exclude_chroma", False)
        show_only_chroma = st.session_state.get("show_only_chroma", False)
        filter_mp4 = st.session_state.get("filter_mp4", True)
        filter_mkv = st.session_state.get("filter_mkv", True)

        for file in os.listdir(self.working_dir):
            file_lower = file.lower()
            file_ext = os.path.splitext(file_lower)[1]

            # Filtre par extension
            if not ((filter_mp4 and file_ext == ".mp4") or (filter_mkv and file_ext == ".mkv")):
                continue

            base_name = os.path.splitext(file_lower)[0]

            # Filtre pour ne montrer que les chroma
            if show_only_chroma and not base_name.startswith("chroma_"):
                continue

            # Filtres d'exclusion
            if exclude_edited and base_name.endswith("_edited"):
                continue
            if exclude_chroma and base_name.startswith("chroma_"):
                continue

            full_path = os.path.join(self.working_dir, file)
            vtt_path = os.path.splitext(full_path)[0] + ".vtt"
            videos.append({
                "Video": file,
                "Full Path": full_path,
                "Has Transcript": os.path.exists(vtt_path),
                # Ajout de la colonne Type (MP4/MKV)
                "Type": file_ext.upper()[1:]
            })
        return pd.DataFrame(videos)

    def display_videos(self, video_df):
        st.write(t("movied_video_list"))
        selected_video = st.dataframe(
            video_df[["Video", "Type", "Has Transcript"]],
            selection_mode="single-row",
            on_select="rerun",
            key="movied_selector",
            hide_index=True,
            height=200,
            column_config={
                "Type": st.column_config.TextColumn(t("movied_format_column")),
                "Has Transcript": st.column_config.CheckboxColumn(t("movied_transcript_column"))
            }
        )
        return selected_video

    def handle_transcript(self, selected_video, video_df, selected_model):
        """Gère l'affichage et l'édition des sous-titres avec AgGrid"""
        if not selected_video["selection"]["rows"]:
            return None, None, None

        idx = selected_video["selection"]["rows"][0]
        video_info = video_df.iloc[idx]
        vtt_path = os.path.splitext(video_info["Full Path"])[0] + ".vtt"
        video_path = video_info["Full Path"]

        # Boutons de contrôle
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(t("movied_generate_transcript")):
                with st.spinner(t("movied_processing")):
                    try:
                        generate_subtitles(video_path, selected_model)
                        subtitles_df, _ = load_subtitles_and_chapters(vtt_path)
                        st.session_state["subtitles_df"] = subtitles_df
                        st.session_state["current_vtt_path"] = vtt_path
                        save_vtt(vtt_path, subtitles_df, pd.DataFrame())
                        st.success(t("movied_success").format(
                            video=os.path.basename(video_path)))
                        st.rerun()
                    except Exception as e:
                        st.error(t("movied_error").format(error=str(e)))
                        return None, None, None

        with col2:
            if st.button(t("movied_normalize_audio")):
                with st.spinner(t("movied_normalizing")):
                    try:
                        normalize_audio(video_path, self.reference_audio_path)
                        st.rerun()
                    except Exception as e:
                        st.error(t("movied_error").format(error=str(e)))

        with col3:
            if st.button("Clean up operations"):
                if 'operations_log' in st.session_state:
                    del st.session_state.operations_log
                if 'generated_video_path' in st.session_state:
                    del st.session_state.generated_video_path
                st.rerun()

        # Chargement ou récupération des sous-titres
        if os.path.exists(vtt_path):
            if "subtitles_df" not in st.session_state or st.session_state.get("current_vtt_path") != vtt_path:
                subtitles_df, _ = load_subtitles_and_chapters(vtt_path)
                if "Category" not in subtitles_df.columns:
                    subtitles_df["Category"] = ""
                if "Complement" not in subtitles_df.columns:
                    subtitles_df["Complement"] = ""
                st.session_state["subtitles_df"] = subtitles_df
                st.session_state["current_vtt_path"] = vtt_path
            else:
                subtitles_df = st.session_state["subtitles_df"]

            st.write(t("movied_subtitles").format(video=video_info["Video"]))

            # Configuration et affichage de la grille AgGrid
            grid_options = self.setup_grid_options(subtitles_df)

            if "grid_key" not in st.session_state:
                self.refresh_grid_key()

            grid_response = AgGrid(
                subtitles_df,
                gridOptions=grid_options,
                height=400,
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True,
                update_mode=GridUpdateMode.VALUE_CHANGED | GridUpdateMode.SELECTION_CHANGED,
                key=st.session_state.grid_key
            )

            # Mise à jour du DataFrame dans session_state
            st.session_state["subtitles_df"] = grid_response['data']
            selected_rows = pd.DataFrame(grid_response['selected_rows'])

            # Boutons pour les suggestions uniquement
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Suggest All", key="suggest_all_btn"):
                    self.handle_llm_suggestions(
                        video_path, vtt_path, subtitles_df)
            with col2:
                if st.button("Suggest for Selection", key="suggest_selection_btn", disabled=selected_rows.empty):
                    self.handle_llm_suggestions_for_selection(
                        video_path, vtt_path, selected_rows)
            with col3:
                if st.button("Rafraîchir les opérations", key="refresh_ops_btn"):
                    st.session_state["subtitles_df"] = self.update_operations_in_grid(
                        st.session_state["subtitles_df"])
                    # st.rerun()

            return selected_rows, subtitles_df, vtt_path
        return None, None, None

    def handle_llm_suggestions_for_selection(self, video_path, vtt_path, selected_rows):
        """Gère les suggestions LLM pour les lignes sélectionnées uniquement"""
        if selected_rows.empty:
            st.warning("Please select at least one subtitle row.")
            return

        config = self.plugin_manager.config
        edit_prompt = config.get(self.name, {}).get(
            'edit_suggestion_prompt', self.get_config_fields()["edit_suggestion_prompt"]["default"])

        with st.spinner("Getting LLM suggestions for selection..."):
            subtitles_df = st.session_state["subtitles_df"]

            for _, row in selected_rows.iterrows():
                prompt = edit_prompt.format(
                    start=row["Start"],
                    end=row["End"],
                    text=row["Text"],
                    category=row["Category"],
                    complement=row["Complement"]
                )
                response = self.process_with_llm(
                    prompt,
                    config.get('llm', {}).get('llm_sys_prompt', ''),
                    row["Text"]
                )

                for line in response.split('\n'):
                    if line.startswith('[SUGGESTION]'):
                        try:
                            parts = line.split(maxsplit=2)
                            if len(parts) >= 2:
                                if not row["Category"]:
                                    if len(parts) >= 3:
                                        category = parts[1]
                                        complement = parts[2]
                                        # Localiser la ligne par Start et End
                                        mask = (subtitles_df["Start"] == row["Start"]) & (
                                            subtitles_df["End"] == row["End"])
                                        subtitles_df.loc[mask,
                                                         "Category"] = category
                                        subtitles_df.loc[mask,
                                                         "Complement"] = complement
                                else:
                                    complement = parts[1]
                                    current_complement = row["Complement"]
                                    new_complement = f"{current_complement}, {complement}" if current_complement else complement
                                    # Localiser la ligne par Start et End
                                    mask = (subtitles_df["Start"] == row["Start"]) & (
                                        subtitles_df["End"] == row["End"])
                                    subtitles_df.loc[mask,
                                                     "Complement"] = new_complement
                        except Exception as e:
                            st.warning(
                                f"Error parsing suggestion: {line} - {str(e)}")

            st.session_state["subtitles_df"] = subtitles_df
            st.success(
                f"Applied suggestions to {len(selected_rows)} selected rows")
            st.rerun()

    def handle_llm_suggestions(self, video_path, vtt_path, subtitles_df):
        """Gère les suggestions globales LLM pour illustrations et mèmes dans l'AgGrid"""
        config = self.plugin_manager.config

        # Récupérer les prompts depuis la configuration
        illustration_prompt = config.get(self.name, {}).get('illustration_prompt',
                                                            "Analyze the following subtitles and suggest where to add illustrations. Format: [ILLUSTRATION] start_time complement")
        meme_prompt = config.get(self.name, {}).get('meme_prompt',
                                                    "Analyze the following subtitles and suggest where to add memes. Format: [MEME] start_time complement")

        with st.spinner("Getting LLM suggestions..."):
            # Préparer le texte des sous-titres pour le LLM
            subtitles_text = "\n".join(
                f"{row['Start']} - {row['End']}: {row['Text']}"
                for _, row in subtitles_df.iterrows()
            )

            # Générer les suggestions pour les illustrations
            illustration_response = self.process_with_llm(
                illustration_prompt,
                config.get('llm', {}).get('llm_sys_prompt', ''),
                subtitles_text
            )

            # Générer les suggestions pour les mèmes
            meme_response = self.process_with_llm(
                meme_prompt,
                config.get('llm', {}).get('llm_sys_prompt', ''),
                subtitles_text
            )

            # Réinitialiser les colonnes Category et Complement pour éviter les conflits
            subtitles_df["Category"] = ""
            subtitles_df["Complement"] = ""

            # Traiter les suggestions d'illustrations
            for line in illustration_response.split('\n'):
                if line.startswith('[ILLUSTRATION]'):
                    try:
                        parts = line.split(maxsplit=2)
                        if len(parts) < 2:
                            continue
                        start_time_str = parts[1]
                        complement = parts[2] if len(parts) > 2 else ""
                        start_time_sec = time_to_seconds(start_time_str)

                        # Trouver le sous-titre correspondant
                        for idx, row in subtitles_df.iterrows():
                            start_sec = time_to_seconds(row["Start"])
                            end_sec = time_to_seconds(row["End"])
                            if start_sec <= start_time_sec <= end_sec:
                                subtitles_df.at[idx,
                                                "Category"] = "illustration"
                                subtitles_df.at[idx, "Complement"] = complement
                                break
                    except Exception as e:
                        st.warning(
                            f"Error parsing illustration suggestion: {line} - {str(e)}")

            # Traiter les suggestions de mèmes
            for line in meme_response.split('\n'):
                if line.startswith('[MEME]'):
                    try:
                        parts = line.split(maxsplit=2)
                        if len(parts) < 2:
                            continue
                        start_time_str = parts[1]
                        complement = parts[2] if len(parts) > 2 else ""
                        start_time_sec = time_to_seconds(start_time_str)

                        # Trouver le sous-titre correspondant
                        for idx, row in subtitles_df.iterrows():
                            start_sec = time_to_seconds(row["Start"])
                            end_sec = time_to_seconds(row["End"])
                            if start_sec <= start_time_sec <= end_sec:
                                subtitles_df.at[idx, "Category"] = "meme"
                                subtitles_df.at[idx, "Complement"] = complement
                                break
                    except Exception as e:
                        st.warning(
                            f"Error parsing meme suggestion: {line} - {str(e)}")

            # Mettre à jour st.session_state pour refléter les changements dans l'AgGrid
            st.session_state["subtitles_df"] = subtitles_df
            st.success("Applied global LLM suggestions to subtitles.")
            st.rerun()  # Rafraîchir l'interface pour afficher les changements

    def handle_section(self, selected_final, final_subtitles_df):
        if selected_final and selected_final["selection"]["rows"]:
            selected_indices = selected_final["selection"]["rows"]
            if not final_subtitles_df.empty and selected_indices[0] < len(final_subtitles_df):
                start_time = final_subtitles_df.iloc[selected_indices[0]]["Start"]
                end_time = final_subtitles_df.iloc[selected_indices[-1]]["End"]

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
                    "Selected final subtitle index out of bounds or DataFrame is empty.")
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
            media_files = {"images": [], "videos": [], "audio": []}
            for dir_path in self.media_dirs:
                if not os.path.exists(dir_path):
                    continue
                files = [f for f in os.listdir(
                    dir_path) if os.path.isfile(os.path.join(dir_path, f))]
                for file in files:
                    full_path = os.path.join(dir_path, file)
                    if file.lower().endswith(IMAGE_EXTENSIONS):
                        base64_url = image_to_base64(full_path)
                        if base64_url:
                            media_files["images"].append({
                                "File": file,
                                "Path": full_path,
                                "Preview": base64_url
                            })
                    elif file.lower().endswith(VIDEO_EXTENSIONS):
                        thumbnail = generate_thumbnail(full_path, 0)
                        if thumbnail:
                            media_files["videos"].append({
                                "File": file,
                                "Path": full_path,
                                "Preview": thumbnail
                            })
                    elif file.lower().endswith(AUDIO_EXTENSIONS):
                        media_files["audio"].append({
                            "File": file,
                            "Path": full_path,
                            "Preview": None
                        })
            st.session_state["media_thumbnails"] = {
                "images": pd.DataFrame(media_files["images"]),
                "videos": pd.DataFrame(media_files["videos"]),
                "audio": pd.DataFrame(media_files["audio"])
            }
            st.session_state["last_thumbnail_size"] = current_size

        # Récupérer depuis session_state
        image_df = st.session_state["media_thumbnails"]["images"]
        video_df = st.session_state["media_thumbnails"]["videos"]
        audio_df = st.session_state["media_thumbnails"]["audio"]
        return image_df, video_df, audio_df

    def show_media_preview(self, media_path):
        """Affiche une prévisualisation du média dans une colonne centrale (1/3 de la largeur)."""
        st.subheader("Preview")
        if media_path.lower().endswith(IMAGE_EXTENSIONS):
            # Ajuste à la largeur de la colonne
            st.image(media_path, use_container_width=True)
        elif media_path.lower().endswith(VIDEO_EXTENSIONS):
            st.video(media_path, format="video/mp4", autoplay=True, muted=True)
        elif media_path.lower().endswith(AUDIO_EXTENSIONS):
            st.audio(media_path)

    def verify_operations(self):
        """Verify if operations in the input text overlap."""
        operations = st.session_state.get("operations", "").strip()
        if not operations:
            st.warning("No operations to verify.")
            return

        # Parse operations into a list of tuples (start, end)
        ops_list = []
        for line in operations.split("\n"):
            parts = line.strip().split()
            if parts[0] in ["insert_video", "insert_audio"]:
                continue
            if len(parts) >= 3:  # Expect at least operation_type, start, end
                try:
                    start = self.parse_timecode(parts[1])
                    end = self.parse_timecode(parts[2])
                    ops_list.append((start, end, line))
                except ValueError:
                    st.warning(f"Invalid timecode in operation: {line}")
                    return

        # Check for overlaps
        overlaps = False
        for i in range(len(ops_list)):
            for j in range(i + 1, len(ops_list)):
                start1, end1, line1 = ops_list[i]
                start2, end2, line2 = ops_list[j]
                if start1 < end2 and start2 < end1:  # Overlap condition
                    st.warning(t("movied_overlap_warning").format(
                        start1=self.format_timecode(start1), end1=self.format_timecode(end1),
                        start2=self.format_timecode(start2), end2=self.format_timecode(end2)
                    ))
                    overlaps = True

        if not overlaps and ops_list:
            st.success(t("movied_all_ok"))

    def update_operations_in_grid(self, subtitles_df):
        """Met à jour la colonne Operation dans subtitles_df en fonction des opérations"""
        if "operations" not in st.session_state or not st.session_state["operations"]:
            subtitles_df["Operation"] = ""  # Réinitialiser si aucune opération
            return subtitles_df

        # Initialiser la colonne Operation si elle n'existe pas
        if "Operation" not in subtitles_df.columns:
            subtitles_df["Operation"] = ""

        # Réinitialiser toutes les opérations
        subtitles_df["Operation"] = ""

        # Parser les opérations
        operations = st.session_state["operations"].split("\n")
        for op in operations:
            if not op.strip():
                continue
            # Extraire le start_time (premier timecode après le type d'opération)
            parts = op.split()
            if len(parts) < 2:
                continue
            op_type = parts[0]
            start_time_str = parts[1]
            start_time_sec = time_to_milliseconds(start_time_str)

            # Trouver le sous-titre correspondant
            found = False
            for idx, row in subtitles_df.iterrows():
                start_sec = time_to_milliseconds(row["Start"])
                end_sec = time_to_milliseconds(row["End"])
                if start_sec <= start_time_sec <= end_sec:
                    # Ajouter l'opération à la colonne, en concaténant si nécessaire
                    current_op = subtitles_df.at[idx, "Operation"]
                    new_op = f"{current_op}; {op}" if current_op else op
                    subtitles_df.at[idx, "Operation"] = op
                    found = True
                    break
            if not found:
                st.warning(f"Not found : {op}")
        self.refresh_grid_key()
        return subtitles_df

    def handle_operation(self, selected_rows):
        """Gère l'ajout d'une opération basée sur les lignes sélectionnées et la commande choisie."""
        if selected_rows.empty:
            st.warning("Please select at least one subtitle row.")
            return

        # Récupérer les informations de média et texte
        media_path = st.session_state.get("movied_media_selector", None)
        text_input = st.session_state.get("text_input", "")

        # Déterminer le type de média
        media_type = None
        if media_path:
            ext = os.path.splitext(media_path)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                media_type = "image"
            elif ext in VIDEO_EXTENSIONS:
                media_type = "video"
            elif ext in AUDIO_EXTENSIONS:
                media_type = "audio"

        # Obtenir les commandes disponibles
        available_commands = self.orchestrator.get_available_commands(
            has_selection=not selected_rows.empty,
            has_text=bool(text_input.strip()),
            media_type=media_type
        )

        if not available_commands:
            st.warning("No commands available for the current context.")
            return

        # Créer la liste des libellés pour le selectbox
        command_labels = {cmd.get_label(): name for name, cmd in available_commands.items()}

        # Sélecteur de commande
        selected_command_label = st.selectbox(
            "Select Operation",
            options=list(command_labels.keys()),
            key="command_select"
        )

        # Regrouper les lignes consécutives pour une sélection multiple
        subtitles_df = st.session_state["subtitles_df"]
        selected_rows = selected_rows.sort_values("Start")
        groups = []
        current_group = [selected_rows.iloc[0]]

        for i in range(1, len(selected_rows)):
            prev_end = time_to_seconds(current_group[-1]["End"])
            curr_start = time_to_seconds(selected_rows.iloc[i]["Start"])
            if abs(curr_start - prev_end) < 0.8:  # Tolérance de 0.1s
                current_group.append(selected_rows.iloc[i])
            else:
                groups.append(current_group)
                current_group = [selected_rows.iloc[i]]
        groups.append(current_group)

        # Bouton pour ajouter l'opération
        if st.button("Add Operation", key="add_operation_btn"):
            selected_command = available_commands[command_labels[selected_command_label]]

            for group in groups:
                start_time = group[0]["Start"]
                end_time = group[-1]["End"] if selected_command.is_enabled(has_selection=True, has_text=bool(text_input), media_type=media_type) else None

                # Générer la commande par défaut
                operation = selected_command.get_default_command(
                    start_time=start_time,
                    end_time=end_time,
                    text=text_input if text_input.strip() else None,
                    media_path=media_path
                )

                self.add_to_operations(operation)

    def handle_operations(self, selected_rows, video_path, vtt_path, thumbnail_size, font, font_size):
        """Gère la sélection de médias, l'entrée de texte et les opérations."""
        if not video_path:
            st.warning("Please select a video to edit first.")
            return

        st.subheader("Media Selection and Operations")

        col1, col2 = st.columns([1, 3])
        with col1:
            media_dir_options = [t("movied_all_directories")] + self.media_dirs
            selected_dirs = st.multiselect(
                t("movied_filter_media_dir"),
                options=media_dir_options,
                default=[t("movied_all_directories")],
                key="media_dir_select",
                label_visibility="collapsed"
            )
        with col2:
            all_extensions = ALL_EXTENSIONS
            selected_extensions = st.multiselect(
                "Filter by File Extensions",
                options=all_extensions,
                default=ALL_EXTENSIONS,
                key="extension_select",
                label_visibility="collapsed"
            )

        col_selector, col_preview = st.columns([1, 1])
        if t("movied_all_directories") in selected_dirs:
            dirs_to_scan = self.media_dirs
        else:
            dirs_to_scan = [d for d in selected_dirs if d != t("movied_all_directories")]

        if not dirs_to_scan:
            st.warning("Please select at least one directory.")
            return
        if not selected_extensions:
            st.warning("Please select at least one file extension.")
            return

        with col_selector:
            initial_search = None
            if selected_rows is not None and not selected_rows.empty and "Category" in selected_rows.columns:
                valid_rows = selected_rows[selected_rows["Category"].isin(["illustration", "meme"])]
                if not valid_rows.empty:
                    initial_search = ", ".join(valid_rows["Complement"].dropna().astype(str))
            selected_media = media_selector(
                media_dirs=dirs_to_scan,
                extensions=selected_extensions,
                suffix="movied",
                initial_search=initial_search
            )

        with col_preview:
            if selected_media:
                self.show_media_preview(selected_media)
            else:
                plugin = self.plugin_manager.get_plugin('illustrator')
                plugin.run(self.plugin_manager.config)

        st.write(t("movied_text_operations"))
        text_input = st.text_input(
            t("movied_text_input"),
            key="text_input"
        )

        st.session_state["movied_media_selector"] = selected_media

        # Appel à la nouvelle méthode handle_operation
        self.handle_operation(selected_rows)

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

    def execute_operations(self, video_path, operations, font, font_size, text_background):
        with st.spinner("Processing video operations..."):
            try:
                # Préparer les arguments pour l'orchestrateur
                use_green_background = text_background == t("movied_green_background")
                text_style = st.session_state.get(
                    "text_style_select", t("movied_text_style_outline"))
                text_style = "outline" if text_style == t("movied_text_style_outline") else "box"

                kwargs = {
                    "working_dir": self.working_dir,
                    "font": font,
                    "font_size": font_size,
                    "use_green_background": use_green_background,
                    "text_style": text_style
                }

                # Obtenir la taille cible à partir du clip initial
                with VideoFileClip(video_path) as tmpclip:
                    target_size = (tmpclip.w, tmpclip.h)

                # Exécuter les opérations via l'orchestrateur
                main_clip, operation_log = self.orchestrator.execute_operations(
                    video_path, operations, target_size, **kwargs)

                # Sauvegarder le clip final
                output_path = os.path.splitext(video_path)[0] + "_edited.mp4"
                from lib.movied_commands import display_video_clip_debug
                display_video_clip_debug(main_clip)
                main_clip.write_videofile(
                    output_path, codec="libx264", audio_codec="aac")
                st.success(f"Video generated successfully at {output_path}")

                # Stocker le journal des opérations
                operations_df = pd.DataFrame(operation_log, columns=[
                    "Nature", "Details", "Start", "End", "Duration"])
                st.session_state["operations_log"] = operations_df
                st.session_state["generated_video_path"] = output_path
                st.session_state.preview_mode = False
                main_clip.close()
            except Exception as e:
                st.error(t("movied_error").format(error=str(e)))
                raise e

    def preview(self, video_path, operations, font, font_size):
        with st.spinner("Processing video operations..."):
            try:
                # Préparer les arguments pour l'orchestrateur
                text_background = st.session_state.get(
                    "text_background_select", t("movied_green_background"))
                use_green_background = text_background == t("movied_green_background")
                text_style = st.session_state.get(
                    "text_style_select", t("movied_text_style_outline"))
                text_style = "outline" if text_style == t("movied_text_style_outline") else "box"

                kwargs = {
                    "working_dir": self.working_dir,
                    "font": font,
                    "font_size": font_size,
                    "use_green_background": use_green_background,
                    "text_style": text_style
                }

                # Obtenir la taille cible à partir du clip initial
                with VideoFileClip(video_path) as tmpclip:
                    target_size = (tmpclip.w, tmpclip.h)

                # Exécuter les opérations via l'orchestrateur
                main_clip, operation_log = self.orchestrator.execute_operations(
                    video_path, operations, target_size, **kwargs)

                # Stocker le journal des opérations et le clip pour la prévisualisation
                operations_df = pd.DataFrame(operation_log, columns=[
                    "Nature", "Details", "Start", "End", "Duration"])
                st.session_state["operations_log"] = operations_df
                st.session_state.preview_mode = True
                if 'previewclip' in st.session_state:
                    st.session_state.previewclip.close()
                st.session_state.previewclip = main_clip
                st.success("Video preview generated successfully")
            except Exception as e:
                st.error(t("movied_error").format(error=str(e)))
                raise e

    def format_timecode(self, seconds):
        """Formate les secondes en timecode HH:MM:SS.mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def export_data(self, video_name):
        if "subtitles_df" not in st.session_state or st.session_state["subtitles_df"].empty:
            st.warning("No subtitles to export.")
            return

        # Prepare data for export
        subtitles_data = st.session_state["subtitles_df"].to_dict(
            orient="records")
        operations = st.session_state.get("operations", "")

        export_data = {
            "subtitles": subtitles_data,
            "operations": operations.split("\n") if operations else []
        }

        # Generate filename with current time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{video_name} - {current_time}.json"
        filepath = os.path.join(self.working_dir, filename)

        # Write to JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=4)

        st.sidebar.success(f"Data exported to {filename}")

    def _import_data(self, imported_data, video_name):
        try:
            # Extraire les opérations et les données des sous-titres
            operations = imported_data.get("operations", "")
            subtitles_data = pd.DataFrame(imported_data.get("subtitles", []))

            if subtitles_data.empty:
                st.warning("No subtitle data found in the imported file.")
                return

            # Vérifier que subtitles_df existe dans session_state
            if "subtitles_df" not in st.session_state or st.session_state["subtitles_df"].empty:
                st.warning(
                    "No current subtitles to merge with. Please load a video first.")
                return

            # Récupérer le subtitles_df actuel
            subtitles_df = st.session_state["subtitles_df"].copy()

            # Colonnes attendues
            expected_cols = ["Start", "End",
                             "Category", "Complement", "Operation"]
            # Sélectionner uniquement les colonnes présentes dans subtitles_data
            available_cols = [
                col for col in expected_cols if col in subtitles_data.columns]
            merge_data = subtitles_data[available_cols]

            # Fusionner avec subtitles_df
            # Supprimer Category, Complement, Operation si présentes
            subtitles_df = subtitles_df.drop(
                columns=available_cols[2:], errors="ignore")
            subtitles_df = subtitles_df.merge(
                merge_data,
                on=["Start", "End"],
                how="left"
            )

            # S'assurer que les colonnes Category, Complement et Operation existent
            for col in ["Category", "Complement", "Operation"]:
                if col not in subtitles_df.columns:
                    # Ajouter la colonne avec des chaînes vides
                    subtitles_df[col] = ""
                else:
                    subtitles_df[col] = subtitles_df[col].fillna(
                        "")  # Remplacer les NaN par des chaînes vides

            # Mettre à jour st.session_state
            st.session_state["subtitles_df"] = subtitles_df
            st.session_state["operations"] = "\n".join(operations)
            if 'operations_log' in st.session_state:
                del st.session_state.operations_log
            if 'generated_video_path' in st.session_state:
                del st.session_state.generated_video_path

            st.success(f"Imported data for {video_name} successfully.")
            st.rerun()  # Rafraîchir l'interface pour afficher les changements

        except Exception as e:
            st.error(f"Error importing data: {str(e)}")

    def import_data(self):
        # Use Streamlit file uploader in sidebar
        uploaded_file = st.sidebar.file_uploader(
            "Choose a JSON file", type="json", key="import_file")
        if uploaded_file and st.sidebar.button(t("movied_import")):
            try:
                # Read and parse JSON
                data = json.load(uploaded_file)
                self._import_data(data, "uploaded file")
                st.sidebar.success("Data imported successfully.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error importing data: {str(e)}")

    def import_last(self, video_name):
        # Find the most recent JSON file for the selected video
        pattern = os.path.join(self.working_dir, f"{video_name} - *.json")
        json_files = glob.glob(pattern)
        if not json_files:
            st.sidebar.warning(f"No export files found for {video_name}.")
            return

        latest_file = max(json_files, key=os.path.getctime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            imported_data = json.load(f)
        self._import_data(imported_data, video_name)

    def run(self, config):
        self.media_dirs = config.get(self.name, {}).get(
            "movied_media_dirs", t("movied_media_dirs_default")).split("\n")
        self.reference_audio_path = config.get(self.name, {}).get(
            "movied_reference_audio", "/path/to/sample.mp3")

        # Initialisation des états de filtre
        if "exclude_edited" not in st.session_state:
            st.session_state.exclude_edited = True
        if "exclude_chroma" not in st.session_state:
            st.session_state.exclude_chroma = False
        if "show_only_chroma" not in st.session_state:
            st.session_state.show_only_chroma = False
        if "filter_mp4" not in st.session_state:
            st.session_state.filter_mp4 = True
        if "filter_mkv" not in st.session_state:
            st.session_state.filter_mkv = True

        self.setup_header()
        selected_model, thumbnail_size, font, font_size, text_background, text_style, preview_buffer = self.setup_controls()

        video_df = self.list_videos()
        if video_df.empty:
            st.write("No videos found in the working directory.")
            return

        selected_video = self.display_videos(video_df)
        selected_rows, subtitles_df, vtt_path = self.handle_transcript(
            selected_video, video_df, selected_model)

        if selected_video["selection"]["rows"]:
            video_name = os.path.basename(video_df.iloc[selected_video["selection"]["rows"][0]]["Full Path"])
            current_ops = st.session_state.get("operations", "").strip()
            if not current_ops or current_ops == f"CHANGE_VIDEO {video_name}":
                st.session_state["operations"] = f"CHANGE_VIDEO {video_name}"

        # Gestion des opérations
        video_path = video_df.iloc[selected_video["selection"]["rows"][0]
                                   ]["Full Path"] if selected_video["selection"]["rows"] else None
        self.handle_operations(selected_rows, video_path,
                               vtt_path, thumbnail_size, font, font_size)

        # Zone de texte pour afficher et éditer les opérations
        operations = st.text_area(
            t("movied_operations"),
            value=st.session_state.get("operations", ""),
            height=150,
            key="operations_area"
        )
        # Met à jour les opérations avec les modifications manuelles
        st.session_state["operations"] = operations

        # Boutons de génération et vérification
        # Ajout d'une colonne pour "Ordonner"
        col1, col2, col3, col4, col5 = st.columns(5)
        if col1.button(t("movied_generate"), key="generate_btn", type="primary") and st.session_state.get("operations"):
            text_background = st.session_state.get(
                "text_background_select", t("movied_green_background"))
            self.execute_operations(
                video_path, st.session_state.operations, font, font_size, text_background)
            #st.rerun()
        if col2.button(t("movied_verify")):
            self.verify_operations()
        if col3.button("Ordonner", key="sort_ops_btn"):  # Nouveau bouton
            if st.session_state.get("operations"):
                # Convertir les opérations en liste pour trier
                ops_list = st.session_state["operations"].split("\n")
                # Filtrer les lignes vides
                ops_list = [op.strip() for op in ops_list if op.strip()]
                # Trier par timecode (deuxième élément de chaque ligne)

                def get_start_time(op):
                    parts = op.split()
                    return time_to_milliseconds(parts[1]) if len(parts) > 1 else float('inf')
                ops_list.sort(key=get_start_time)
                # Rejoindre les opérations triées
                st.session_state["operations"] = "\n".join(ops_list)
                st.rerun()  # Rafraîchir pour afficher les opérations triées
        if col4.button("Preview"):
            self.preview(video_path, st.session_state.operations,
                         font, font_size)
        if col5.button("Nouveau chapitre", key="new_chapter_btn") and selected_video["selection"]["rows"]:
                video_name = os.path.basename(video_df.iloc[selected_video["selection"]["rows"][0]]["Full Path"])
                self.add_to_operations(f"CHANGE_VIDEO {video_name}")
                st.rerun()

        # Gestion des données export/import
        with st.sidebar:
            st.header("Data Management")
            if selected_video["selection"]["rows"]:
                video_name = os.path.splitext(os.path.basename(
                    video_df.iloc[selected_video["selection"]["rows"][0]]["Full Path"]))[0]
                if st.button(t("movied_export")):
                    self.export_data(video_name)
                if st.button(t("movied_import_last")):
                    self.import_last(video_name)
            self.import_data()

        # Affichage du résultat
        if "operations_log" in st.session_state:
            st.write("Operations with Real Timecodes:")
            selected_operation = st.dataframe(
                st.session_state["operations_log"],
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="operations_log_selector"
            )

            if selected_operation["selection"]["rows"]:
                selected_row = selected_operation["selection"]["rows"][0]
                start_time_seconds = time_to_milliseconds(
                    st.session_state["operations_log"].iloc[selected_row]["Start"])/1000
                end_time_seconds = None
                if pd.notna(st.session_state["operations_log"].iloc[selected_row]["End"]):
                    end_time_seconds = time_to_milliseconds(
                        st.session_state["operations_log"].iloc[selected_row]["End"])/1000

                if st.session_state.preview_mode and video_path:
                    main_clip = st.session_state.previewclip
                    clip_duration = main_clip.duration
                    start_preview = max(0, start_time_seconds - preview_buffer)
                    if end_time_seconds is not None:
                        end_preview = min(
                            clip_duration, end_time_seconds + preview_buffer)
                    else:
                        end_preview = min(
                            clip_duration, start_time_seconds + preview_buffer)
                    try:
                        preview_clip = main_clip.subclipped(
                            start_preview, end_preview)
                        preview_clip.preview()
                        preview_clip.close()
                    except Exception as e:
                        st.error(f"Error generating preview clip: {str(e)}")
                    finally:
                        main_clip.close()
                elif "generated_video_path" in st.session_state:
                    st.write("Generated Video:")
                    _, col, _ = st.columns(3)
                    col.video(
                        st.session_state["generated_video_path"],
                        start_time=start_time_seconds,
                        autoplay=True
                    )


if __name__ == "__main__":
    st.write("Movied Plugin standalone test")
