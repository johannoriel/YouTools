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
from video_anim import replace_with_image
import json
from moviepy import VideoFileClip
from media_selector import media_selector, remote_media_selector, ALL_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS
from datetime import datetime
import glob

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
            }
        }

    def get_tabs(self):
        return [{"name": t("movied_tab"), "plugin": "movied"}]

    def setup_header(self):
        st.header(t("movied_header"))

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
                        #print(f"Impossible de lire la police {font_path} : {str(e)}")
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
                "Type": file_ext.upper()[1:]  # Ajout de la colonne Type (MP4/MKV)
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
        # Step 1: Display all subtitles with "Add" and "Forced Add" buttons
        if selected_video["selection"]["rows"]:
            idx = selected_video["selection"]["rows"][0]
            video_info = video_df.iloc[idx]
            vtt_path = os.path.splitext(video_info["Full Path"])[0] + ".vtt"

            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("movied_generate_transcript")):
                    with st.spinner(t("movied_processing")):
                        try:
                            generate_subtitles(video_info["Full Path"], selected_model)
                            subtitles_df, _ = load_subtitles_and_chapters(vtt_path)
                            st.session_state["subtitles_df"] = subtitles_df
                            st.session_state["current_vtt_path"] = vtt_path
                            st.success(t("movied_success").format(video=os.path.basename(video_info["Full Path"])))
                            st.rerun()
                        except Exception as e:
                            st.error(t("movied_error").format(error=str(e)))
                            return None, None, None

            with col2:
                if st.button(t("movied_normalize_audio")):
                    with st.spinner(t("movied_normalizing")):
                        try:
                            normalize_audio(video_info["Full Path"], self.reference_audio_path)
                            st.rerun()
                        except Exception as e:
                            st.error(t("movied_error").format(error=str(e)))

            if os.path.exists(vtt_path):
                if "subtitles_df" not in st.session_state or st.session_state.get("current_vtt_path") != vtt_path:
                    subtitles_df, _ = load_subtitles_and_chapters(vtt_path)
                    st.session_state["subtitles_df"] = subtitles_df
                    st.session_state["current_vtt_path"] = vtt_path
                else:
                    subtitles_df = st.session_state["subtitles_df"]

                st.write(t("movied_subtitles").format(video=video_info["Video"]))
                selected_subtitles = st.dataframe(
                    subtitles_df[["Start", "End", "Text"]],
                    selection_mode="multi-row",
                    on_select="rerun",
                    key="subtitle_selector",
                    hide_index=True
                )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button(t("movied_add_to_interest")) and selected_subtitles["selection"]["rows"]:
                        selected_indices = selected_subtitles["selection"]["rows"]
                        if not subtitles_df.empty and all(idx < len(subtitles_df) for idx in selected_indices):
                            new_entries = subtitles_df.iloc[selected_indices][["Start", "End", "Text"]]
                            if "interest_subtitles_df" not in st.session_state:
                                st.session_state["interest_subtitles_df"] = new_entries
                            else:
                                existing = st.session_state["interest_subtitles_df"][["Start", "End", "Text"]]
                                combined = pd.concat([existing, new_entries]).drop_duplicates(subset=["Start", "End", "Text"]).reset_index(drop=True)
                                if "Category" in st.session_state["interest_subtitles_df"].columns:
                                    combined = combined.merge(
                                        st.session_state["interest_subtitles_df"][["Start", "End", "Text", "Category", "Complement"]],
                                        on=["Start", "End", "Text"],
                                        how="left"
                                    ).fillna({"Category": "", "Complement": ""})
                                st.session_state["interest_subtitles_df"] = combined
                            st.rerun()

                with col2:
                    if st.button(t("movied_force_add")) and selected_subtitles["selection"]["rows"]:
                        selected_indices = selected_subtitles["selection"]["rows"]
                        if not subtitles_df.empty and all(idx < len(subtitles_df) for idx in selected_indices):
                            new_entries = subtitles_df.iloc[selected_indices][["Start", "End", "Text"]]
                            if "interest_subtitles_df" not in st.session_state:
                                st.session_state["interest_subtitles_df"] = new_entries
                            else:
                                combined = pd.concat([st.session_state["interest_subtitles_df"], new_entries]).reset_index(drop=True)
                                st.session_state["interest_subtitles_df"] = combined
                            st.rerun()

                with col3:
                    if st.button("Suggestions", key="llm_suggestions_btn"):
                        self.handle_llm_suggestions(video_info["Full Path"], vtt_path, subtitles_df)

                with col4:
                    if st.button("Deduplicate", key="deduplicate_step2"):
                        self.remove_duplicates_step2()

                return selected_subtitles, subtitles_df, vtt_path
            return None, None, None
        return None, None, None

    def handle_intermediate_subtitles(self, selected_subtitles, subtitles_df):
        # Step 2: Manage interest subtitles with Remove, Merge, and Sort
        if "interest_subtitles_df" not in st.session_state or st.session_state["interest_subtitles_df"].empty:
            st.write("No subtitles of interest selected yet.")
            return None, None

        intermediate_subtitles_df = st.session_state["interest_subtitles_df"].copy()

        st.write("Subtitles of Interest (Manage):")
        selected_intermediate = st.dataframe(
            intermediate_subtitles_df[["Start", "End", "Text"]],
            selection_mode="multi-row",
            on_select="rerun",
            key="intermediate_subtitle_selector",
            hide_index=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(t("movied_remove_from_interest")) and selected_intermediate["selection"]["rows"]:
                selected_indices = selected_intermediate["selection"]["rows"]
                # Remove from interest subtitles
                intermediate_subtitles_df = intermediate_subtitles_df.drop(selected_indices).reset_index(drop=True)
                st.session_state["interest_subtitles_df"] = intermediate_subtitles_df

                # Sync with edited subtitles (remove matching entries)
                if "edited_subtitles_df" in st.session_state:
                    edited_df = st.session_state["edited_subtitles_df"]
                    # Keep only rows that still exist in interest_subtitles_df
                    edited_df = edited_df.merge(
                        intermediate_subtitles_df[["Start", "End", "Text"]],
                        on=["Start", "End", "Text"],
                        how="inner"
                    ).reset_index(drop=True)
                    # Preserve Category and Complement if they exist
                    if "Category" not in edited_df.columns:
                        edited_df["Category"] = ""
                    if "Complement" not in edited_df.columns:
                        edited_df["Complement"] = ""
                    st.session_state["edited_subtitles_df"] = edited_df[["Start", "End", "Text", "Category", "Complement"]]
                else:
                    st.session_state["edited_subtitles_df"] = intermediate_subtitles_df.copy()

                st.rerun()

        with col2:
            if st.button(t("movied_merge_subtitles")) and selected_intermediate["selection"]["rows"]:
                selected_indices = sorted(selected_intermediate["selection"]["rows"])
                if len(selected_indices) > 1:
                    is_continuous = True
                    for i in range(len(selected_indices) - 1):
                        current_end = self.parse_timecode(intermediate_subtitles_df.iloc[selected_indices[i]]["End"])
                        next_start = self.parse_timecode(intermediate_subtitles_df.iloc[selected_indices[i + 1]]["Start"])
                        if current_end != next_start:
                            if abs(next_start - current_end) > 0.5:
                                is_continuous = False
                                break

                    if is_continuous:
                        start_time = intermediate_subtitles_df.iloc[selected_indices[0]]["Start"]
                        end_time = intermediate_subtitles_df.iloc[selected_indices[-1]]["End"]
                        merged_text = " ".join(intermediate_subtitles_df.iloc[selected_indices]["Text"].tolist())
                        category = intermediate_subtitles_df.iloc[selected_indices[0]].get("Category", "")
                        complement = intermediate_subtitles_df.iloc[selected_indices[0]].get("Complement", "")
                        merged_row = pd.DataFrame({
                            "Start": [start_time],
                            "End": [end_time],
                            "Text": [merged_text],
                            "Category": [category],
                            "Complement": [complement]
                        })
                        intermediate_subtitles_df = intermediate_subtitles_df.drop(selected_indices).reset_index(drop=True)
                        intermediate_subtitles_df = pd.concat([intermediate_subtitles_df, merged_row]).reset_index(drop=True)
                        st.session_state["interest_subtitles_df"] = intermediate_subtitles_df

                        # Sync with edited subtitles
                        st.session_state["edited_subtitles_df"] = intermediate_subtitles_df.copy()
                        st.rerun()
                    else:
                        st.error(t("movied_merge_error_not_continuous"))

        with col3:
            if st.button(t("movied_sort_subtitles")):
                intermediate_subtitles_df["Start_seconds"] = intermediate_subtitles_df["Start"].apply(self.parse_timecode)
                intermediate_subtitles_df = intermediate_subtitles_df.sort_values("Start_seconds").drop(columns=["Start_seconds"]).reset_index(drop=True)
                st.session_state["interest_subtitles_df"] = intermediate_subtitles_df

                # Sync with edited subtitles
                st.session_state["edited_subtitles_df"] = intermediate_subtitles_df.copy()
                st.rerun()

        return selected_intermediate, intermediate_subtitles_df

    def handle_edit_subtitles(self):
        # Step 3: Edit subtitles with st.data_editor and Refresh button
        if "interest_subtitles_df" not in st.session_state or st.session_state["interest_subtitles_df"].empty:
            st.write("No subtitles available for editing.")
            return None

        st.write(t("movied_edit_subtitles"))
        if "edited_subtitles_df" not in st.session_state:
            st.session_state["edited_subtitles_df"] = st.session_state["interest_subtitles_df"].copy()
            if "Category" not in st.session_state["edited_subtitles_df"].columns:
                st.session_state["edited_subtitles_df"]["Category"] = ""
            if "Complement" not in st.session_state["edited_subtitles_df"].columns:
                st.session_state["edited_subtitles_df"]["Complement"] = ""

        def refresh():
            current_edited = st.session_state["edited_subtitles_df"]
            new_base = st.session_state["interest_subtitles_df"].copy()
            synced_df = new_base.merge(
                current_edited[["Category", "Complement", "Start", "End", "Text"]],
                on=["Start", "End", "Text"],
                how="left",
                suffixes=("", "_edited")
            ).fillna({"Category": "", "Complement": ""})
            synced_df = synced_df[["Start", "End", "Text", "Category", "Complement"]]
            st.session_state["edited_subtitles_df"] = synced_df

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button(t("movied_refresh"), key="refresh_edit"):
                refresh()
                st.rerun()
        with col2:
            # Get the prompt for help text
            config = self.plugin_manager.config
            edit_prompt = config.get(self.name, {}).get('edit_suggestion_prompt', "")
            if st.button(t("movied_suggestions"), key="llm_edit_suggestions_btn", help=edit_prompt):
                self.handle_llm_edit_suggestions()
        with col3:
            if st.button("Deduplicate", key="deduplicate_step3"):
                self.remove_duplicates_step3()

        # Bug bypass : https://github.com/streamlit/streamlit/issues/7749
        def update():
            for idx, change in st.session_state.subtitle_editor["edited_rows"].items():
                #print(f"idx : {idx} / change : {change}")
                for label, value in change.items():
                    print(f"label : {label} / value : {value}")
                    st.session_state.edited_subtitles_df.loc[idx, label] = value
                    refresh() #needed to key idx in sync

        edited_df = st.data_editor(
            st.session_state["edited_subtitles_df"],
            column_config={
                "Start": st.column_config.TextColumn("Start", disabled=True),
                "End": st.column_config.TextColumn("End", disabled=True),
                "Text": st.column_config.TextColumn("Text", disabled=True),
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    options=["", "meme", "illustration", "texte"],
                    default=""
                ),
                "Complement": st.column_config.TextColumn("Complement", default="")
            },
            #hide_index=True,
            key="subtitle_editor",
            on_change=update
        )
        st.session_state["edited_subtitles_df"] = edited_df
        return edited_df

    def handle_final_selection(self):
        # Step 4: Final selection with single-row mode and Refresh button
        if "edited_subtitles_df" not in st.session_state or st.session_state["edited_subtitles_df"].empty:
            st.write("No edited subtitles available for final selection.")
            return None, None

        st.write(t("movied_final_selection"))
        col1, col2 = st.columns(2)
        multiple_selection = col1.checkbox(t("movied_multiple_selection"), value=False, key="multiple_selection_final")
        selection_mode = "multi-row" if multiple_selection else "single-row"
        if col2.button(t("movied_refresh"), key="refresh_final"):
            if "interest_subtitles_df" in st.session_state:
                current_edited = st.session_state["edited_subtitles_df"]
                new_base = st.session_state["interest_subtitles_df"].copy()
                synced_df = new_base.merge(
                    current_edited[["Category", "Complement", "Start", "End", "Text"]],
                    on=["Start", "End", "Text"],
                    how="left",
                    suffixes=("", "_edited")
                ).fillna({"Category": "", "Complement": ""})
                synced_df = synced_df[["Start", "End", "Text", "Category", "Complement"]]
                st.session_state["edited_subtitles_df"] = synced_df
            st.rerun()

        final_subtitles_df = st.session_state["edited_subtitles_df"]
        selected_final = st.dataframe(
            final_subtitles_df[["Category", "Complement", "Start", "End", "Text"]],
            selection_mode=selection_mode,
            on_select="rerun",
            key="final_subtitle_selector",
            hide_index=True
        )
        return selected_final, final_subtitles_df

    def remove_duplicates_step2(self):
        """Remove duplicates from interest_subtitles_df based on Start, End, and Text"""
        if "interest_subtitles_df" not in st.session_state or st.session_state["interest_subtitles_df"].empty:
            st.warning("No subtitles of interest to deduplicate.")
            return

        df = st.session_state["interest_subtitles_df"]
        initial_count = len(df)
        # Drop duplicates based on Start, End, Text
        deduplicated_df = df.drop_duplicates(subset=["Start", "End", "Text"]).reset_index(drop=True)
        st.session_state["interest_subtitles_df"] = deduplicated_df

        # Sync with edited_subtitles_df
        if "edited_subtitles_df" in st.session_state:
            edited_df = st.session_state["edited_subtitles_df"]
            synced_df = deduplicated_df.merge(
                edited_df[["Start", "End", "Text", "Category", "Complement"]],
                on=["Start", "End", "Text"],
                how="left",
                suffixes=("", "_edited")
            ).fillna({"Category": "", "Complement": ""})
            st.session_state["edited_subtitles_df"] = synced_df[["Start", "End", "Text", "Category", "Complement"]]

        removed_count = initial_count - len(deduplicated_df)
        if removed_count > 0:
            st.success(f"Removed {removed_count} duplicates from interest subtitles")
        else:
            st.info("No duplicates found in interest subtitles")
        st.rerun()

    def remove_duplicates_step3(self):
        """Remove duplicates from edited_subtitles_df based on Start, End, Text, and Category, merging Complement"""
        if "edited_subtitles_df" not in st.session_state or st.session_state["edited_subtitles_df"].empty:
            st.warning("No edited subtitles to deduplicate.")
            return

        df = st.session_state["edited_subtitles_df"]
        initial_count = len(df)

        # Clean trailing commas in Complement
        df["Complement"] = df["Complement"].apply(lambda x: x.rstrip(',') if pd.notna(x) and isinstance(x, str) else x)

        # Group by Start, End, Text, Category and merge Complement
        def merge_complements(group):
            if len(group) > 1:
                # Split complements, remove duplicates, and join back
                all_complements = []
                for comp in group["Complement"]:
                    if pd.notna(comp) and comp:
                        all_complements.extend([c.strip() for c in comp.split(",")])
                # Remove duplicates while preserving order
                unique_complements = list(dict.fromkeys([c for c in all_complements if c]))
                return ", ".join(unique_complements)
            return group["Complement"].iloc[0]

        deduplicated_df = df.groupby(["Start", "End", "Text", "Category"]).apply(
            lambda g: pd.Series({
                "Complement": merge_complements(g)
            })
        ).reset_index()

        st.session_state["edited_subtitles_df"] = deduplicated_df[["Start", "End", "Text", "Category", "Complement"]]

        # Sync back to interest_subtitles_df
        st.session_state["interest_subtitles_df"] = deduplicated_df.copy()

        removed_count = initial_count - len(deduplicated_df)
        if removed_count > 0:
            st.success(f"Removed {removed_count} duplicates from edited subtitles and merged complements")
        else:
            st.info("No duplicates found in edited subtitles")
        st.rerun()

    def handle_llm_suggestions(self, video_path, vtt_path, subtitles_df):
        """Handle LLM suggestions for illustrations and memes with configurable prompts"""
        if not os.path.exists(vtt_path):
            st.warning("Please generate a transcript first.")
            return

        # Convert VTT to plain text transcript
        transcript = ""
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if '-->' in line:
                    continue
                if line.strip() and not line.startswith('WEBVTT'):
                    transcript += line.strip() + "\n"

        # Get prompts from config (with French defaults)
        config = self.plugin_manager.config
        illustration_prompt = config.get(self.name, {}).get('illustration_prompt', """
            Analyse la transcription suivante d'une vidéo et identifie les sections qui bénéficieraient d'illustrations visuelles.
            Pour chaque section, fournis :
            1. Le timecode de début (format : HH:MM:SS.sss)
            2. Un seul mot décrivant le thème de l'illustration mais en anglais

            Retourne ta réponse dans ce format exact, une suggestion par ligne :
            [ILLUSTRATION] HH:MM:SS.sss thème

            Voici la transcription :
            {transcript}
        """)

        meme_prompt = config.get(self.name, {}).get('meme_prompt', """
            Analyse la transcription suivante d'une vidéo et identifie les sections avec des émotions fortes adaptées à des mèmes.
            Pour chaque section, fournis :
            1. Le timecode de début (format : HH:MM:SS.sss)
            2. Un seul mot décrivant l'émotion principale mais en anglais

            Retourne ta réponse dans ce format exact, une suggestion par ligne :
            [MEME] HH:MM:SS.sss émotion

            Voici la transcription :
            {transcript}
        """)

        # Process LLM suggestions
        with st.spinner("Getting LLM suggestions..."):
            # Get illustration suggestions
            illustration_response = self.process_with_llm(
                illustration_prompt.format(transcript=transcript),
                config.get('ragllm', {}).get('llm_sys_prompt', ''),
                transcript
            )

            # Get meme suggestions
            meme_response = self.process_with_llm(
                meme_prompt.format(transcript=transcript),
                config.get('ragllm', {}).get('llm_sys_prompt', ''),
                transcript
            )

            # Process responses
            new_entries = []

            # Parse illustration suggestions
            for line in illustration_response.split('\n'):
                if line.startswith('[ILLUSTRATION]'):
                    try:
                        parts = line.split(maxsplit=3)
                        if len(parts) == 3:
                            start, theme = parts[1], parts[2]
                            # Try exact match
                            matching_subs = subtitles_df[subtitles_df['Start'] == start]
                            if matching_subs.empty:
                                # Find closest previous subtitle
                                start_sec = self.parse_timecode(start)
                                subtitles_df['Start_sec'] = subtitles_df['Start'].apply(self.parse_timecode)
                                previous_subs = subtitles_df[subtitles_df['Start_sec'] <= start_sec]
                                if not previous_subs.empty:
                                    matching_subs = previous_subs.iloc[[-1]]
                                    st.warning(f"Imprecise illustration timecode {start}: using previous subtitle at {matching_subs.iloc[0]['Start']}")
                                else:
                                    st.warning(f"No matching subtitle found for illustration timecode {start}")
                                    continue

                            sub = matching_subs.iloc[0]
                            print(sub)
                            new_entries.append({
                                'Start': sub['Start'],
                                'End': sub['End'],
                                'Text': sub['Text'],
                                'Category': 'illustration',
                                'Complement': theme
                            })
                    except Exception as e:
                        st.warning(f"Error parsing illustration suggestion: {line} - {str(e)}")

            # Parse meme suggestions
            for line in meme_response.split('\n'):
                if line.startswith('[MEME]'):
                    try:
                        parts = line.split(maxsplit=3)
                        if len(parts) == 3:
                            start, emotion = parts[1], parts[2]
                            # Try exact match
                            matching_subs = subtitles_df[subtitles_df['Start'] == start]
                            if matching_subs.empty:
                                # Find closest previous subtitle
                                start_sec = self.parse_timecode(start)
                                subtitles_df['Start_sec'] = subtitles_df['Start'].apply(self.parse_timecode)
                                previous_subs = subtitles_df[subtitles_df['Start_sec'] <= start_sec]
                                if not previous_subs.empty:
                                    matching_subs = previous_subs.iloc[[-1]]
                                    st.warning(f"Imprecise meme timecode {start}: using previous subtitle at {matching_subs.iloc[0]['Start']}")
                                else:
                                    st.warning(f"No matching subtitle found for meme timecode {start}")
                                    continue

                            sub = matching_subs.iloc[0]
                            print(sub)
                            new_entries.append({
                                'Start': sub['Start'],
                                'End': sub['End'],
                                'Text': sub['Text'],
                                'Category': 'meme',
                                'Complement': emotion
                            })
                    except Exception as e:
                        st.warning(f"Error parsing meme suggestion: {line} - {str(e)}")

            # Clean up temporary column
            if 'Start_sec' in subtitles_df.columns:
                subtitles_df = subtitles_df.drop(columns=['Start_sec'])

            # Add to interest subtitles (Step 2)
            print(new_entries)
            if new_entries:
                new_df = pd.DataFrame(new_entries)
                if "interest_subtitles_df" not in st.session_state:
                    st.session_state["interest_subtitles_df"] = new_df
                else:
                    existing = st.session_state["interest_subtitles_df"]
                    #combined = pd.concat([existing, new_df]).drop_duplicates(
                    #    subset=["Start", "End", "Text"]
                    #).reset_index(drop=True)
                    combined = pd.concat([existing, new_df]).reset_index(drop=True)
                    st.session_state["interest_subtitles_df"] = combined

                # Sync with edited subtitles (Step 3)
                if "edited_subtitles_df" in st.session_state:
                    current_edited = st.session_state["edited_subtitles_df"]
                    synced_df = st.session_state["interest_subtitles_df"].merge(
                        current_edited[["Category", "Complement", "Start", "End", "Text"]],
                        on=["Start", "End", "Text"],
                        how="left",
                        suffixes=("", "_edited")
                    ).fillna({"Category": "", "Complement": ""})
                    synced_df = synced_df[["Start", "End", "Text", "Category", "Complement"]]
                    st.session_state["edited_subtitles_df"] = synced_df
                else:
                    st.session_state["edited_subtitles_df"] = st.session_state["interest_subtitles_df"].copy()

                st.success(f"Added {len(new_entries)} suggestions from LLM")
                st.rerun()
            else:
                st.info("No suggestions found by LLM")

    def handle_llm_edit_suggestions(self):
        """Handle LLM suggestions for Category and Complement in edited subtitles, appending to existing complements"""
        if "edited_subtitles_df" not in st.session_state or st.session_state["edited_subtitles_df"].empty:
            st.warning("No edited subtitles available for suggestions.")
            return

        edited_df = st.session_state["edited_subtitles_df"].copy()

        # Get prompt from config
        config = self.plugin_manager.config
        edit_prompt = config.get(self.name, {}).get('edit_suggestion_prompt', """
            Analyse la ligne suivante d'une transcription vidéo et propose :
            - Si aucune catégorie n'est fournie ("{category}" est vide) : une catégorie ('illustration', 'meme', ou 'texte') et des compléments (mots séparés par des virgules)
            - Si une catégorie est fournie ("{category}") : des compléments (mots séparés par des virgules) adaptés à la catégorie

            Entrée : {start} - {end} - "{text}" - Catégorie actuelle : "{category}" - Compléments actuels : "{complement}"

            Retourne ta réponse dans ce format exact :
            - Avec catégorie vide : [SUGGESTION] catégorie complément1, complément2, ...
            - Avec catégorie remplie : [SUGGESTION] complément1, complément2, ...

            Ne réponds qu'une seule ligne par suggestion.
        """)

        # Process each line with LLM
        with st.spinner("Getting LLM suggestions for edits..."):
            suggestions = []
            for _, row in edited_df.iterrows():
                current_category = row["Category"] if pd.notna(row["Category"]) else ""
                current_complement = row["Complement"] if pd.notna(row["Complement"]) else ""

                # Prepare prompt
                prompt = edit_prompt.format(
                    start=row["Start"],
                    end=row["End"],
                    text=row["Text"],
                    category=current_category,
                    complement=current_complement
                )
                response = self.process_with_llm(
                    prompt,
                    config.get('ragllm', {}).get('llm_sys_prompt', ''),
                    row["Text"]
                )

                # Parse response
                for line in response.split('\n'):
                    if line.startswith('[SUGGESTION]'):
                        try:
                            parts = line.split(maxsplit=2)
                            if len(parts) >= 2:
                                # Case 1: Category is empty, expect category and complements
                                if not current_category:
                                    if len(parts) >= 3:
                                        suggested_category = parts[1]
                                        suggested_complements = parts[2]
                                        new_complement = suggested_complements if not current_complement else f"{current_complement}, {suggested_complements}"
                                        suggestions.append({
                                            'Start': row['Start'],
                                            'End': row['End'],
                                            'Text': row['Text'],
                                            'Category': suggested_category,
                                            'Complement': new_complement
                                        })
                                # Case 2: Category exists, expect only complements
                                else:
                                    suggested_complements = parts[1]
                                    new_complement = suggested_complements if not current_complement else f"{current_complement}, {suggested_complements}"
                                    suggestions.append({
                                        'Start': row['Start'],
                                        'End': row['End'],
                                        'Text': row['Text'],
                                        'Category': current_category,
                                        'Complement': new_complement
                                    })
                        except Exception as e:
                            st.warning(f"Error parsing suggestion for '{row['Text']}': {line} - {str(e)}")

            # Update edited subtitles with suggestions
            if suggestions:
                suggested_df = pd.DataFrame(suggestions)
                # Merge with existing edited_df to update only relevant rows
                updated_df = edited_df.merge(
                    suggested_df[["Start", "End", "Text", "Category", "Complement"]],
                    on=["Start", "End", "Text"],
                    how="left",
                    suffixes=("_old", "")
                )
                # Keep old values where no new suggestion was provided
                for col in ["Category", "Complement"]:
                    updated_df[col] = updated_df[col].fillna(updated_df[f"{col}_old"])
                updated_df = updated_df.drop(columns=[f"{col}_old" for col in ["Category", "Complement"]])
                st.session_state["edited_subtitles_df"] = updated_df[["Start", "End", "Text", "Category", "Complement"]]

                # Sync back to interest_subtitles_df
                st.session_state["interest_subtitles_df"] = updated_df.copy()

                st.success(f"Applied {len(suggestions)} suggestions to edited subtitles")
                st.rerun()
            else:
                st.info("No suggestions provided by LLM")

    def handle_section(self, selected_final, final_subtitles_df):
        if selected_final and selected_final["selection"]["rows"]:
            selected_indices = selected_final["selection"]["rows"]
            if not final_subtitles_df.empty and selected_indices[0] < len(final_subtitles_df):
                start_time = final_subtitles_df.iloc[selected_indices[0]]["Start"]
                end_time = final_subtitles_df.iloc[selected_indices[-1]]["End"]

                col1, col2 = st.columns(2)
                with col1:
                    edited_start = st.text_input(t("movied_start_time"), start_time, key="start_time")
                with col2:
                    edited_end = st.text_input(t("movied_end_time"), end_time, key="end_time")
                return edited_start, edited_end
            else:
                st.error("Selected final subtitle index out of bounds or DataFrame is empty.")
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
            if media_path.lower().endswith(IMAGE_EXTENSIONS):
                st.image(media_path, use_container_width=True)  # Ajuste à la largeur de la colonne
            elif media_path.lower().endswith(VIDEO_EXTENSIONS):
                st.video(media_path, format="video/mp4", autoplay=True, muted=True)

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
            if parts[0] == "insert_video":
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
        all_extensions = ALL_EXTENSIONS
        selected_extensions = col2.multiselect(
            "Filter by File Extensions",
            options=all_extensions,
            default=ALL_EXTENSIONS,
            key="extension_select"
        )

        if not selected_extensions:
            st.warning("Please select at least one file extension.")
            return

        # Single media selector for images and videos
        # Use Complement as initial_search if Category is "illustration"
        initial_search = None
        if "final_subtitle_selector" in st.session_state and st.session_state["final_subtitle_selector"]["selection"]["rows"]:
            selected_idx = st.session_state["final_subtitle_selector"]["selection"]["rows"][0]
            final_df = st.session_state["edited_subtitles_df"]
            if final_df.iloc[selected_idx]["Category"] in ["illustration", "meme"]:
                initial_search = final_df.iloc[selected_idx]["Complement"]

        selected_media = media_selector(
            media_dirs=dirs_to_scan,
            extensions=selected_extensions,
            suffix="movied",
            initial_search=initial_search
        )

        # Prévisualisation si un média est sélectionné
        if selected_media:
            self.show_media_preview(selected_media)

        # Determine media type
        is_image = selected_media and any(selected_media.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
        is_video = selected_media and any(selected_media.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
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

    def _execute_operations(self, video_path, vtt_path, operations, font, font_size):
        main_clip = VideoFileClip(video_path)
        target_size = (main_clip.w, main_clip.h)
        subtitles_df, _ = load_subtitles_and_chapters(vtt_path)
        duration_offset = 0
        operation_log = []  # Liste pour stocker les opérations avec nature séparée

        text_background = st.session_state.get(
            "text_background_select", t("movied_green_background"))
        use_green_background = text_background == t(
            "movied_green_background")
        background_type = "green" if use_green_background else "video"
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
                        main_clip, start_sec, end_sec, image_path, target_size, background=background_type)
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
                        main_clip, start_sec, end_sec, video_path_replace, target_size, background=background_type)
                    subtitles_df = self.adjust_subtitles(
                        subtitles_df, end_time, duration_change)
                    duration_offset += duration_change
                    operation_log.append(
                        {"Nature": "replace_video", "Details": video_path_replace, "Start": real_start, "End": real_end})

                elif cmd == "replace_video_keep_audio":
                    video_path_replace = remaining_args
                    main_clip = replace_video_keep_audio(
                        main_clip, start_sec, end_sec, video_path_replace, target_size, background=background_type)
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
        return main_clip, operation_log, subtitles_df

    def execute_operations(self, video_path, vtt_path, operations, font, font_size):
        with st.spinner("Processing video operations..."):
            try:
                main_clip, operation_log, subtitles_df = self._execute_operations(video_path, vtt_path, operations, font, font_size)
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
                main_clip.close()
                st.rerun()
                self.alert()
            except Exception as e:
                st.error(t("movied_error").format(error=str(e)))

    def preview(self, video_path, vtt_path, operations, font, font_size):
        with st.spinner("Processing video operations..."):
            try:
                main_clip, operation_log, subtitles_df = self._execute_operations(video_path, vtt_path, operations, font, font_size)
                save_vtt(vtt_path, subtitles_df, pd.DataFrame())
                st.success(f"Video generated successfully")
                operations_df = pd.DataFrame(operation_log, columns=[
                                                "Nature", "Details", "Start", "End"])
                st.session_state["operations_log"] = operations_df
                main_clip.preview()
                main_clip.close()
                st.rerun()
                self.alert()
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
        if "edited_subtitles_df" not in st.session_state or st.session_state["edited_subtitles_df"].empty:
            st.sidebar.warning("No data to export.")
            return

        # Prepare data for export
        subtitles_data = st.session_state["edited_subtitles_df"].to_dict(orient="records")
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

    def _import_data_json(self, data):
        subtitles_data = data.get("subtitles", [])
        operations_data = data.get("operations", [])

        # Validate and convert subtitles data to DataFrame
        if subtitles_data:
            imported_df = pd.DataFrame(subtitles_data)
            required_columns = ["Start", "End", "Text"]
            optional_columns = ["Category", "Complement"]
            if all(col in imported_df.columns for col in required_columns):
                # Ensure optional columns exist
                for col in optional_columns:
                    if col not in imported_df.columns:
                        imported_df[col] = ""
                imported_df = imported_df[["Start", "End", "Text", "Category", "Complement"]]
                st.session_state["interest_subtitles_df"] = imported_df.copy()
                st.session_state["edited_subtitles_df"] = imported_df.copy()
            else:
                st.sidebar.error("Imported JSON missing required subtitle columns.")
                return

        # Import operations
        if operations_data:
            st.session_state["operations"] = "\n".join(operations_data)
        else:
            st.session_state["operations"] = ""

    def import_data(self):
        # Use Streamlit file uploader in sidebar
        uploaded_file = st.sidebar.file_uploader("Choose a JSON file", type="json", key="import_file")
        if uploaded_file and st.sidebar.button(t("movied_import")):
            try:
                # Read and parse JSON
                data = json.load(uploaded_file)
                self._import_data_json(data)
                if 'operation_log' in st.session_state:
                    del st.session_state.operations_log
                if 'generated_video_path' in st.session_state:
                    del st.session_state.generated_video_path
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
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._import_data_json(data)
                if 'operation_log' in st.session_state:
                    del st.session_state.operations_log
                if 'generated_video_path' in st.session_state:
                    del st.session_state.generated_video_path
                st.sidebar.success(f"Imported last export: {os.path.basename(latest_file)}")
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error importing last file: {str(e)}")

    def run(self, config):
        self.working_dir = config.get(self.name, {}).get("movied_workdir", t("movied_workdir_default"))
        self.media_dirs = config.get(self.name, {}).get("movied_media_dirs", t("movied_media_dirs_default")).split("\n")
        self.reference_audio_path = config.get(self.name, {}).get("movied_reference_audio", "/path/to/sample.mp3")

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
        selected_model, thumbnail_size, font, font_size, text_background, text_style = self.setup_controls()

        video_df = self.list_videos()
        if video_df.empty:
            st.write("No videos found in the working directory.")
            return

        selected_video = self.display_videos(video_df)
        selected_subtitles, subtitles_df, vtt_path = self.handle_transcript(selected_video, video_df, selected_model)
        selected_intermediate, intermediate_subtitles_df = self.handle_intermediate_subtitles(selected_subtitles, subtitles_df)
        edited_subtitles_df = self.handle_edit_subtitles()
        selected_final, final_subtitles_df = self.handle_final_selection()
        start_time, end_time = self.handle_section(selected_final, final_subtitles_df)

        # Sidebar buttons for Export and Import
        with st.sidebar:
            st.header("Data Management")
            if selected_video["selection"]["rows"]:
                video_name = os.path.splitext(os.path.basename(video_df.iloc[selected_video["selection"]["rows"][0]]["Full Path"]))[0]
                if st.button(t("movied_export")):
                    self.export_data(video_name)
                if st.button(t("movied_import_last")):
                    self.import_last(video_name)
            self.import_data()

        if selected_video["selection"]["rows"]:
            video_path = video_df.iloc[selected_video["selection"]["rows"][0]]["Full Path"]
            if selected_final and selected_final["selection"]["rows"]:
                self.handle_operations(start_time, end_time, video_path, vtt_path, thumbnail_size, font, font_size)
            else:
                self.handle_operations(None, None, video_path, vtt_path, thumbnail_size, font, font_size)
        else:
            self.handle_operations(None, None, None, None, thumbnail_size, font, font_size)

        # Generate button
        col1, col2, col3 = st.columns(3)
        if col1.button(t("movied_generate"), key="generate_btn", type="primary") and st.session_state.operations:
            self.execute_operations(video_path, vtt_path, st.session_state.operations, font, font_size)
        if col2.button(t("movied_verify")):
            self.verify_operations()
        if col3.button("Preview"):
            self.preview(video_path, vtt_path, st.session_state.operations, font, font_size)

        if "operations_log" in st.session_state:
            st.write("Operations with Real Timecodes:")
            selected_operation = st.dataframe(
                st.session_state["operations_log"],
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="operations_log_selector"
            )
            start_time_seconds = None
            if selected_operation["selection"]["rows"]:
                selected_row = selected_operation["selection"]["rows"][0]
                selected_timecode = st.session_state["operations_log"].iloc[selected_row]["Start"]
                h, m, s = map(float, selected_timecode.replace(",", ".").split(":"))
                start_time_seconds = h * 3600 + m * 60 + s

        if "generated_video_path" in st.session_state:
            st.write("Generated Video:")
            _, col, _ = st.columns(3)
            col.video(
                st.session_state["generated_video_path"],
                start_time=start_time_seconds if start_time_seconds is not None else 0,
                autoplay=True,
            )


if __name__ == "__main__":
    st.write("Movied Plugin standalone test")
