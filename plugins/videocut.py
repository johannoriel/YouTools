from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
from lib.video_utils import (
    scan_videos, load_subtitles_and_chapters, save_vtt, generate_subtitles,
    convert_to_mp4, rename_video, merge_videos, split_video, delete_videos,
    generate_thumbnail, format_time, parse_timecode_to_ms, split_by_chapters, normalize_audio
)
import pandas as pd
import os
from plugins.trimsilences import TrimsilencesPlugin
from plugins.chromakey import ChromakeyPlugin
from lib.chromakey_background import replace_background

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
    "video_chapter_start": "Chapter Start",  # Nouvelle traduction
    "video_chapter_end": "Chapter End",  # Nouvelle traduction
    "video_add_chapter": "Add Chapter",
    "video_delete_chapter": "Delete Chapter",
    "video_edit_chapter": "Edit Chapter",
    "video_chapters_label": "Chapters",
    "video_generate_thumbnails": "Generate Thumbnails",
    "video_refresh_thumbnails": "Refresh Thumbnails",
    "video_convert_to_mp4": "Convert to MP4",
    "video_rename": "Rename",
    "video_merge": "Merge Videos",
    "video_split": "Split Video",
    "video_extensions_label": "Filter by Extensions",
    "video_delete": "Delete Videos",
    "video_mute_label": "Mute videos",
    "video_auto_chapter": "Auto-Generate Chapters",
    "video_auto_chapter_processing": "Generating chapters automatically...",
    "video_auto_chapter_success": "Chapters generated successfully for {video}!",
    "video_auto_chapter_prompt": "Prompt for Auto-Chaptering",
    "video_shift_chapter_up": "Shift Chapter Up",
    "video_shift_chapter_down": "Shift Chapter Down",
    "video_shift_n": "Shift by (subtitles)",
    "video_split_by_chapters": "Split by Chapters",
    "video_move_up": "Move Up",
    "video_move_down": "Move Down",
    "video_delete_chapter_continuous": "Delete Chapter",
    "video_delete_chapter_discontinuous": "Discontinuous Delete",
    "video_create_chapter": "Create Chapter",
    "video_directory_selector": "Directory Selection",
    "video_directory_all": "All directories (recursive)",
    "video_directory_select": "Select directories",
    "video_normalize_audio": "Normalize Audio",
    "video_normalize_processing": "Normalizing audio...",
    "video_normalize_success": "Audio normalized successfully for {video}!",
})

translations["fr"].update({
    "video_tab": "Éditeur Vidéo",
    "video_header": "Éditeur Vidéo Basé sur les Sous-titres",
    "video_config_workdir": "Répertoire de travail des vidéos",
    "video_config_workdir_default": "/chemin/vers/vidéos",
    "video_list_label": "Vidéos Disponibles",
    "video_generate_subtitles": "Générer les Sous-titres",
    "video_processing": "Génération des sous-titres en cours...",
    "video_success": "Sous-titres générés avec succès pour {video} !",
    "video_error": "Une erreur s'est produite : {error}",
    "video_subtitles_label": "Sous-titres pour {video}",
    "video_model_label": "Modèle de transcription",
    "video_chapter_title": "Titre du chapitre",
    "video_chapter_start": "Début du chapitre",  # Nouvelle traduction
    "video_chapter_end": "Fin du chapitre",  # Nouvelle traduction
    "video_add_chapter": "Ajouter un chapitre",
    "video_delete_chapter": "Supprimer un chapitre",
    "video_edit_chapter": "Modifier un chapitre",
    "video_chapters_label": "Chapitres",
    "video_generate_thumbnails": "Générer les vignettes",
    "video_refresh_thumbnails": "Rafraîchir les vignettes",
    "video_convert_to_mp4": "Convertir en MP4",
    "video_rename": "Renommer",
    "video_merge": "Fusionner les vidéos",
    "video_split": "Découper la vidéo",
    "video_extensions_label": "Filtrer par extensions",
    "video_delete": "Supprimer les vidéos",
    "video_mute_label": "Vidéos muettes",
    "video_auto_chapter": "Générer les chapitres automatiquement",
    "video_auto_chapter_processing": "Génération automatique des chapitres en cours...",
    "video_auto_chapter_success": "Chapitres générés avec succès pour {video} !",
    "video_auto_chapter_prompt": "Prompt pour le chapitrage automatique",
    "video_shift_chapter_up": "Déplacer le chapitre vers le haut",
    "video_shift_chapter_down": "Déplacer le chapitre vers le bas",
    "video_shift_n": "Déplacer le chapitre de n positions",
    "video_split_by_chapters": "Découper la vidéo par chapitres",
    "video_move_up": "Remonter",
    "video_move_down": "Descendre",
    "video_delete_chapter_continuous": "Supprimer le chapitre",
    "video_delete_chapter_discontinuous": "Supprimer discontinue",
    "video_create_chapter": "Créer un chapitre",
    "video_directory_selector": "Sélection de répertoires",
    "video_directory_all": "Tous les répertoires (récursif)",
    "video_directory_select": "Sélectionner des répertoires",
    "video_normalize_audio": "Normaliser l'audio",
    "video_normalize_processing": "Normalisation de l'audio en cours...",
    "video_normalize_success": "Audio normalisé avec succès pour {video} !",
})


class VideocutPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.working_dir = None
        self.trimsilences_plugin = self.plugin_manager.get_plugin(
            'trimsilences')
        self.chromakey_plugin = self.plugin_manager.get_plugin('chromakey')

    def get_config_fields(self):
        return {
            "video_workdir": {
                "type": "text",
                "label": t("video_config_workdir"),
                "default": t("video_config_workdir_default")
            },
            "auto_chapter_prompt": {  # Nouveau champ de configuration
                "type": "textarea",
                "label": t("video_auto_chapter_prompt"),
                "default": "Based on the following subtitles, generate a list of chapters with precise timecodes (e.g., 00:00:00.000) and titles followed by a dash and a short summary. Return the result in this format:\n\n00:00:00.000 - Title - Summary\n00:05:00.000 - Title - Summary\n\nHere are the subtitles:\n"
            }
        }

    def get_tabs(self):
        return [{"name": t("video_tab"), "plugin": "videoplugin"}]

    def setup_header(self):
        st.header(t("video_header"))

    def setup_controls(self):
        with st.expander("Options"):  # Ajout d'un expander
            col_model, col_thumb, col_refresh, col_ext, col_mute, col_debug = st.columns([
                                                                                         2, 1, 1, 2, 1, 1])
            with col_model:
                model_options = ["base", "medium",
                                 "turbo", "large-v3", "large-v3-turbo"]
                selected_model = st.selectbox(
                    t("video_model_label"), model_options, index=4, key="video_model")
            with col_thumb:
                generate_thumbnails = st.checkbox(
                    t("video_generate_thumbnails"), key="video_generate_thumbnails")
            with col_refresh:
                refresh_thumbnails = st.button(t("video_refresh_thumbnails"))
            with col_ext:
                extension_options = [".mp4", ".mkv", ".ogg"]
                selected_extensions = st.multiselect(
                    t("video_extensions_label"), extension_options, default=extension_options)
            with col_mute:
                mute_videos = st.checkbox(t("video_mute_label"), value=False)
            with col_debug:
                show_end_columns = st.checkbox(
                    "Show End Columns", value=False, key="show_end_columns")  # Option pour afficher "End"
            return selected_model, generate_thumbnails, refresh_thumbnails, selected_extensions, mute_videos, show_end_columns

    def display_videos(self, video_df):
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(t("video_list_label"))
            # Initialiser ou ajuster l’ordre des vidéos dans session_state
            if "video_order" not in st.session_state or len(st.session_state["video_order"]) != len(video_df) or any(i >= len(video_df) for i in st.session_state["video_order"]):
                st.session_state["video_order"] = video_df.index.tolist()

            # Appliquer l’ordre personnalisé au DataFrame
            ordered_video_df = video_df.iloc[st.session_state["video_order"]].reset_index(
                drop=True)

            selected_videos = st.dataframe(
                ordered_video_df[["Video", "Directory",
                                  "Duration", "Has Subtitles"]],
                selection_mode="multi-row",
                on_select="rerun",
                key="video_selector",
                hide_index=True
            )
        return col1, col2, selected_videos

    def handle_video_actions(self, col1, selected_videos, video_df, selected_model, config):
        with col1:
            if selected_videos["selection"]["rows"]:
                selected_idx = selected_videos["selection"]["rows"][0]
                # Utiliser l’ordre personnalisé
                selected_video = video_df.iloc[st.session_state["video_order"][selected_idx]]
                vtt_path = os.path.splitext(
                    selected_video["Full Path"])[0] + ".vtt"
                has_chapters = selected_video["Has Subtitles"] and os.path.exists(
                    vtt_path) and "CHAPTERS" in open(vtt_path, "r", encoding="utf-8").read()
                with st.expander("Video Actions", expanded=not has_chapters):
                    col_gen, col_conv = st.columns(2)
                    with col_gen:
                        if st.button(t("video_generate_subtitles")) and selected_videos["selection"]["rows"]:
                            with st.spinner(t("video_processing")):
                                try:
                                    for idx in selected_videos["selection"]["rows"]:
                                        video_path = video_df.iloc[st.session_state["video_order"]
                                                                   [idx]]["Full Path"]
                                        with st.spinner(f"Generating subtitles for {os.path.basename(video_path)}..."):
                                            generate_subtitles(
                                                video_path, selected_model)
                                        st.success(t("video_success").format(
                                            video=os.path.basename(video_path)))
                                    st.rerun()
                                except Exception as e:
                                    st.error(
                                        t("video_error").format(error=str(e)))
                    with col_conv:
                        if st.button(t("video_convert_to_mp4")) and selected_videos["selection"]["rows"]:
                            with st.spinner("Converting videos..."):
                                for idx in selected_videos["selection"]["rows"]:
                                    video_path = video_df.iloc[st.session_state["video_order"]
                                                               [idx]]["Full Path"]
                                    if not video_path.endswith(".mp4"):
                                        convert_to_mp4(video_path)
                                st.rerun()

                    col_rename = st.columns([3, 1])
                    with col_rename[0]:
                        new_name = st.text_input(
                            "New Video Name", os.path.splitext(selected_video["Video"])[0])
                    with col_rename[1]:
                        if st.button(t("video_rename")) and new_name:
                            with st.spinner("Renaming video..."):
                                rename_video(
                                    selected_video["Full Path"], new_name)
                            st.rerun()

                    col_merge, col_delete = st.columns(2)
                    with col_merge:
                        # Préparer la liste des vidéos sélectionnées
                        selected_video_paths = [video_df.iloc[st.session_state["video_order"][idx]]["Full Path"]
                                                for idx in selected_videos["selection"]["rows"]]
                        if len(selected_video_paths) >= 2:
                            # Afficher le text_area pour réorganiser les vidéos
                            default_text = "\n".join(
                                [os.path.basename(path) for path in selected_video_paths])
                            video_order_input = st.text_area(
                                "Videos to merge (one per line, reorder as needed)",
                                default_text,
                                height=150,
                                help="Rearrange the videos by editing the list. Each line represents one video."
                            )
                        else:
                            st.write(
                                "Select at least two videos to enable merging.")
                            video_order_input = ""

                        # Bouton Fusionner
                        if st.button(t("video_merge")) and selected_videos["selection"]["rows"]:
                            if len(selected_video_paths) < 2:
                                st.error(
                                    "Please select at least two videos to merge.")
                            else:
                                # Traiter l'entrée utilisateur
                                new_order_names = [
                                    line.strip() for line in video_order_input.split("\n") if line.strip()]
                                # Vérifier que toutes les vidéos entrées sont valides
                                original_names = [os.path.basename(
                                    path) for path in selected_video_paths]
                                invalid_entries = [
                                    name for name in new_order_names if name not in original_names]
                                if invalid_entries:
                                    st.error(
                                        f"Invalid video names: {', '.join(invalid_entries)}. Please use only the selected videos.")
                                elif len(new_order_names) < 2:
                                    st.error(
                                        "At least two videos are required to merge.")
                                else:
                                    # Reconstruire la liste des chemins dans le nouvel ordre
                                    name_to_path = {os.path.basename(
                                        path): path for path in selected_video_paths}
                                    reordered_paths = [name_to_path[name]
                                                       for name in new_order_names]
                                    # Mettre à jour l'ordre dans video_df pour refléter dans st.session_state["video_order"]
                                    new_order_indices = []
                                    for path in reordered_paths:
                                        idx = video_df[video_df["Full Path"]
                                                       == path].index[0]
                                        new_order_indices.append(idx)
                                    # Mettre à jour video_order en plaçant les vidéos fusionnées en premier
                                    current_order = st.session_state["video_order"]
                                    unselected_indices = [
                                        i for i in current_order if i not in new_order_indices]
                                    st.session_state["video_order"] = new_order_indices + \
                                        unselected_indices
                                    # Lancer la fusion
                                    with st.spinner("Merging videos..."):
                                        merge_videos(
                                            reordered_paths, self.working_dir)
                                    st.rerun()

                    with col_delete:
                        # Initialiser l'état dans session_state si non présent
                        if "delete_requested" not in st.session_state:
                            st.session_state["delete_requested"] = False
                        if "videos_to_delete" not in st.session_state:
                            st.session_state["videos_to_delete"] = []

                        # Bouton pour demander la suppression
                        if st.button(t("video_delete")) and selected_videos["selection"]["rows"]:
                            st.session_state["delete_requested"] = True
                            st.session_state["videos_to_delete"] = [
                                video_df.iloc[st.session_state["video_order"][idx]]["Full Path"] for idx in selected_videos["selection"]["rows"]]
                            st.rerun()

                        # Afficher l'alerte et la confirmation si une suppression est demandée
                        if st.session_state["delete_requested"]:
                            video_names = [os.path.basename(
                                path) for path in st.session_state["videos_to_delete"]]
                            st.warning(
                                f"Are you sure you want to delete the following videos?\n\n{', '.join(video_names)}\n\nThis action cannot be undone.", icon="⚠️")
                            if st.button("Confirm Deletion"):
                                with st.spinner("Deleting videos..."):
                                    delete_videos(
                                        st.session_state["videos_to_delete"])
                                # Réinitialiser l'état après suppression
                                st.session_state["delete_requested"] = False
                                st.session_state["videos_to_delete"] = []
                                st.rerun()

                    col_auto_chapter = st.columns([1, 1])
                    with col_auto_chapter[0]:
                        if st.button(t("video_auto_chapter")) and selected_videos["selection"]["rows"]:
                            with st.spinner(t("video_auto_chapter_processing")):
                                try:
                                    subtitles_df, chapters_df = load_subtitles_and_chapters(
                                        vtt_path)
                                    if subtitles_df.empty:
                                        st.error(
                                            "No subtitles available for chapter generation.")
                                    else:
                                        subtitles_text = "\n".join(
                                            f"{row['Start']} - {row['Text']}" for _, row in subtitles_df.iterrows()
                                        )
                                        prompt = config.get(self.name, {}).get("auto_chapter_prompt", self.get_config_fields()[
                                            "auto_chapter_prompt"]["default"]) + subtitles_text
                                        sysprompt = "You are an AI assistant tasked with analyzing video subtitles and generating meaningful chapters with timecodes, titles, and summaries."
                                        response = self.process_with_llm(
                                            prompt, sysprompt, subtitles_text)
                                        new_chapters = []
                                        for line in response.split("\n"):
                                            if line.strip() and " - " in line:
                                                try:
                                                    timecode, rest = line.split(
                                                        " - ", 1)
                                                    title, summary = rest.split(
                                                        " - ", 1)
                                                    new_chapters.append(
                                                        {"Start": timecode.strip(), "End": "", "Title": f"{title} - {summary}"})
                                                except ValueError:
                                                    continue
                                        if new_chapters:
                                            new_chapters_df = pd.DataFrame(
                                                new_chapters)
                                            for i in range(len(new_chapters_df) - 1):
                                                new_chapters_df.at[i,
                                                                   "End"] = new_chapters_df.at[i + 1, "Start"]
                                            new_chapters_df.at[len(
                                                new_chapters_df) - 1, "End"] = subtitles_df["End"].iloc[-1]
                                            chapters_df = pd.concat(
                                                [chapters_df, new_chapters_df], ignore_index=True)
                                            save_vtt(
                                                vtt_path, subtitles_df, chapters_df)
                                            st.success(t("video_auto_chapter_success").format(
                                                video=os.path.basename(selected_video["Full Path"])))
                                            st.rerun()
                                        else:
                                            st.error(
                                                "No valid chapters generated by the LLM.")
                                except Exception as e:
                                    st.error(
                                        t("video_error").format(error=str(e)))

                    with col_auto_chapter[1]:
                        if st.button(t("video_split_by_chapters")) and selected_videos["selection"]["rows"]:
                            with st.spinner("Splitting video by chapters..."):
                                subtitles_df, chapters_df = load_subtitles_and_chapters(
                                    vtt_path)
                                if chapters_df.empty:
                                    st.error(
                                        "No chapters available to split the video.")
                                else:
                                    split_by_chapters(
                                        selected_video["Full Path"], os.path.dirname(selected_video["Full Path"]), chapters_df=chapters_df)
                                st.rerun()

                    # Nouveaux boutons "Move Up" et "Move Down"
                    col_move_up, col_move_down = st.columns(2)
                    with col_move_up:
                        if st.button(t("video_move_up")) and selected_videos["selection"]["rows"]:
                            if selected_idx > 0:  # Ne pas remonter si déjà en haut
                                current_order = st.session_state["video_order"]
                                new_order = current_order.copy()
                                # Échanger avec l’élément précédent
                                new_order[selected_idx], new_order[selected_idx -
                                                                   1] = new_order[selected_idx - 1], new_order[selected_idx]
                                st.session_state["video_order"] = new_order
                                st.rerun()
                    with col_move_down:
                        if st.button(t("video_move_down")) and selected_videos["selection"]["rows"]:
                            # Ne pas descendre si déjà en bas
                            if selected_idx < len(video_df) - 1:
                                current_order = st.session_state["video_order"]
                                new_order = current_order.copy()
                                # Échanger avec l’élément suivant
                                new_order[selected_idx], new_order[selected_idx +
                                                                   1] = new_order[selected_idx + 1], new_order[selected_idx]
                                st.session_state["video_order"] = new_order
                                st.rerun()

                    col1_actions, col2_actions, col3_actions = st.columns(3)
                    with col1_actions:
                        # Section Chromakey
                        background_directory = config.get(
                            'chromakey', {}).get('background_directory', '')
                        if not background_directory:
                            st.error(
                                "Background directory not configured in chromakey plugin")
                        else:
                            background_files = [f for f in os.listdir(background_directory)
                                                if f.lower().endswith(('.mp4', '.avi', '.mov'))]
                            if not background_files:
                                st.error(
                                    "No background videos found in directory")
                            else:
                                # Afficher la sélection du fond
                                selected_background = st.selectbox(
                                    "Select background video",
                                    background_files,
                                    key="chroma_background_select"
                                )

                                # Bouton Replace Green Screen
                                if st.button("Replace Green Screen") and selected_videos["selection"]["rows"]:
                                    with st.spinner("Replacing green screens..."):
                                        for idx in selected_videos["selection"]["rows"]:
                                            video_path = video_df.iloc[st.session_state["video_order"]
                                                                       [idx]]["Full Path"]
                                            try:
                                                background_path = os.path.join(
                                                    background_directory, selected_background)
                                                result_filename = f"chroma_{os.path.basename(video_path)}"
                                                result_path = os.path.join(
                                                    os.path.dirname(video_path), result_filename)
                                                target_color_rgb = config.get('chromakey', {}).get(
                                                    "default_target_color", "#00FF00")
                                                target_color_rgb = [int(target_color_rgb.lstrip('#')[
                                                                        i:i+2], 16) for i in (0, 2, 4)]
                                                replace_background(
                                                    video_path, background_path, result_path, target_color_rgb)
                                                st.success(
                                                    f"Green screen replaced for {os.path.basename(video_path)}")
                                            except Exception as e:
                                                st.error(
                                                    f"Error processing {os.path.basename(video_path)}: {str(e)}")
                                        st.rerun()

                    with col2_actions:
                        # Bouton Normalize Audio
                        if st.button(t("video_normalize_audio")) and selected_videos["selection"]["rows"]:
                            with st.spinner(t("video_normalize_processing")):
                                for idx in selected_videos["selection"]["rows"]:
                                    video_path = video_df.iloc[st.session_state["video_order"]
                                                               [idx]]["Full Path"]
                                    try:
                                        reference_audio_path = config.get("movied", {}).get(
                                            "movied_reference_audio", "")
                                        normalize_audio(
                                            video_path, reference_audio_path)
                                        st.success(t("video_normalize_success").format(
                                            video=os.path.basename(video_path)))
                                    except Exception as e:
                                        st.error(
                                            t("video_error").format(error=str(e)))
                            st.rerun()

                with col3_actions:
                    # Bouton Trim Silences
                    if st.button("Trim Silences") and selected_videos["selection"]["rows"]:
                        with st.spinner("Trimming silences..."):
                            for idx in selected_videos["selection"]["rows"]:
                                video_path = video_df.iloc[st.session_state["video_order"]
                                                           [idx]]["Full Path"]
                                try:
                                    result, reduction, original_duration, final_duration = self.trimsilences_plugin.remove_silence(
                                        video_path,
                                        config['trimsilences']['silence_threshold'],
                                        config['trimsilences']['silence_duration'],
                                        config['trimsilences']['keep_duration'],
                                        os.path.dirname(video_path)
                                    )
                                    if isinstance(result, str) and (result.startswith("Erreur") or result.startswith("Une erreur")):
                                        st.error(
                                            f"Error processing {os.path.basename(video_path)}: {result}")
                                    else:
                                        st.success(
                                            f"Trimmed {os.path.basename(video_path)} - Reduction: {reduction}% | {original_duration:.1f}s → {final_duration:.1f}s")
                                except Exception as e:
                                    st.error(
                                        f"Error processing {os.path.basename(video_path)}: {str(e)}")
                            st.rerun()

    def handle_chapters(self, col1, selected_videos, video_df, show_end_columns):
        with col1:
            if selected_videos["selection"]["rows"]:
                selected_idx = selected_videos["selection"]["rows"][0]
                selected_video = video_df.iloc[st.session_state["video_order"][selected_idx]]
                vtt_path = os.path.splitext(
                    selected_video["Full Path"])[0] + ".vtt"
                if selected_video["Has Subtitles"]:
                    subtitles_df, chapters_df = load_subtitles_and_chapters(
                        vtt_path)
                    if not chapters_df.empty:
                        st.write(t("video_chapters_label"))
                        column_order = ["Start", "End", "Title"] if show_end_columns else [
                            "Start", "Title"]
                        selected_chapters = st.dataframe(
                            chapters_df,
                            selection_mode="multi-row",
                            on_select="rerun",
                            key="chapter_selector",
                            column_order=column_order,
                            hide_index=True
                        )
                        if selected_chapters["selection"]["rows"]:
                            chapter_idx = selected_chapters["selection"]["rows"][0]
                            # Zone de saisie
                            col_start, col_end, col_title = st.columns([
                                                                       1, 1, 2])
                            with col_start:
                                new_start = st.text_input(
                                    t("video_chapter_start"), chapters_df.iloc[chapter_idx]["Start"])
                            with col_end:
                                new_end = st.text_input(
                                    t("video_chapter_end"), chapters_df.iloc[chapter_idx]["End"])
                            with col_title:
                                new_title = st.text_area(
                                    t("video_chapter_title"), chapters_df.iloc[chapter_idx]["Title"], height=100)

                            # Boutons sur une même ligne : Modifier, Supprimer, Suppression discontinue
                            col_edit, col_delete_cont, col_delete_discont = st.columns(
                                3)
                            with col_edit:
                                if st.button(t("video_edit_chapter")) and new_title and new_start and new_end:
                                    if chapter_idx > 0 and new_start != chapters_df.iloc[chapter_idx]["Start"]:
                                        chapters_df.at[chapter_idx -
                                                       1, "End"] = new_start
                                    if chapter_idx < len(chapters_df) - 1 and new_end != chapters_df.iloc[chapter_idx]["End"]:
                                        chapters_df.at[chapter_idx +
                                                       1, "Start"] = new_end
                                    chapters_df.at[chapter_idx,
                                                   "Start"] = new_start
                                    chapters_df.at[chapter_idx,
                                                   "End"] = new_end
                                    chapters_df.at[chapter_idx,
                                                   "Title"] = new_title
                                    save_vtt(
                                        vtt_path, subtitles_df, chapters_df)
                                    st.rerun()
                            with col_delete_cont:
                                if st.button(t("video_delete_chapter_continuous")):
                                    if chapter_idx > 0:
                                        # Étendre le chapitre précédent jusqu’à la fin du chapitre supprimé
                                        chapters_df.at[chapter_idx - 1,
                                                       "End"] = chapters_df.iloc[chapter_idx]["End"]
                                    # Supprimer le chapitre
                                    chapters_df = chapters_df.drop(
                                        chapter_idx).reset_index(drop=True)
                                    save_vtt(
                                        vtt_path, subtitles_df, chapters_df)
                                    st.rerun()
                            with col_delete_discont:
                                if st.button(t("video_delete_chapter_discontinuous")):
                                    # Supprimer le chapitre sans ajuster les autres (laisser un vide)
                                    chapters_df = chapters_df.drop(
                                        chapter_idx).reset_index(drop=True)
                                    save_vtt(
                                        vtt_path, subtitles_df, chapters_df)
                                    st.rerun()

                            # Boutons de décalage (inchangés)
                            col_shift_up, col_shift_down, col_shift_n = st.columns([
                                                                                   1, 1, 1])
                            with col_shift_n:
                                shift_n = st.number_input(
                                    t("video_shift_n"), min_value=1, value=1, step=1)
                            with col_shift_up:
                                if st.button(t("video_shift_chapter_up")):
                                    subtitle_idx = subtitles_df[subtitles_df["Start"] ==
                                                                chapters_df.iloc[chapter_idx]["Start"]].index[0]
                                    new_idx = max(0, subtitle_idx - shift_n)
                                    new_start = subtitles_df.iloc[new_idx]["Start"]
                                    if chapter_idx > 0:
                                        chapters_df.at[chapter_idx -
                                                       1, "End"] = new_start
                                    chapters_df.at[chapter_idx,
                                                   "Start"] = new_start
                                    save_vtt(
                                        vtt_path, subtitles_df, chapters_df)
                                    st.rerun()
                            with col_shift_down:
                                if st.button(t("video_shift_chapter_down")):
                                    subtitle_idx = subtitles_df[subtitles_df["Start"] ==
                                                                chapters_df.iloc[chapter_idx]["Start"]].index[0]
                                    new_idx = min(
                                        len(subtitles_df) - 1, subtitle_idx + shift_n)
                                    new_start = subtitles_df.iloc[new_idx]["Start"]
                                    if chapter_idx > 0:
                                        chapters_df.at[chapter_idx -
                                                       1, "End"] = new_start
                                    chapters_df.at[chapter_idx,
                                                   "Start"] = new_start
                                    save_vtt(
                                        vtt_path, subtitles_df, chapters_df)
                                    st.rerun()

    def handle_subtitles(self, col2, selected_videos, video_df, generate_thumbnails, refresh_thumbnails, mute_videos, show_end_columns):
        with col2:
            if selected_videos["selection"]["rows"]:
                selected_idx = selected_videos["selection"]["rows"][0]
                selected_video = video_df.iloc[st.session_state["video_order"][selected_idx]]
                vtt_path = os.path.splitext(
                    selected_video["Full Path"])[0] + ".vtt"

                if selected_video["Has Subtitles"]:
                    subtitles_df, chapters_df = load_subtitles_and_chapters(
                        vtt_path)

                    if "thumbnails" not in st.session_state:
                        st.session_state["thumbnails"] = {}
                    if refresh_thumbnails or (generate_thumbnails and not st.session_state["thumbnails"].get(vtt_path)):
                        st.session_state["thumbnails"][vtt_path] = {}
                        for i, row in subtitles_df.iterrows():
                            start_ms = parse_timecode_to_ms(row["Start"])
                            thumbnail = generate_thumbnail(
                                selected_video["Full Path"], start_ms)
                            st.session_state["thumbnails"][vtt_path][i] = thumbnail
                    if generate_thumbnails:
                        for i, thumbnail in st.session_state["thumbnails"].get(vtt_path, {}).items():
                            subtitles_df.at[i, "Thumbnail"] = thumbnail

                    subtitles_df["Chapitre"] = ""
                    for i, sub in subtitles_df.iterrows():
                        for _, chap in chapters_df.iterrows():
                            if sub["Start"] >= chap["Start"] and sub["End"] <= chap["End"]:
                                subtitles_df.at[i, "Chapitre"] = chap["Title"].split("\n")[
                                    0]
                                break

                    if "chapter_selector" in st.session_state and st.session_state["chapter_selector"]["selection"]["rows"]:
                        selected_chapter_indices = st.session_state["chapter_selector"]["selection"]["rows"]
                        filtered_subtitles_df = pd.concat([
                            subtitles_df[(subtitles_df["Start"] >= chapters_df.iloc[idx]["Start"]) & (
                                subtitles_df["End"] <= chapters_df.iloc[idx]["End"])]
                            for idx in selected_chapter_indices
                        ]).drop_duplicates()
                    else:
                        filtered_subtitles_df = subtitles_df

                    st.write(t("video_subtitles_label").format(
                        video=selected_video["Video"]))
                    column_order = ["Start", "End", "Chapitre", "Text", "Thumbnail"] if (show_end_columns and generate_thumbnails) else \
                        ["Start", "Chapitre", "Text", "Thumbnail"] if generate_thumbnails else \
                        ["Start", "End", "Chapitre", "Text"] if show_end_columns else \
                        ["Start", "Chapitre", "Text"]
                    selected_subtitles = st.dataframe(
                        filtered_subtitles_df,
                        selection_mode="multi-row",
                        on_select="rerun",
                        key="subtitle_selector",
                        column_config={
                            "Thumbnail": st.column_config.ImageColumn("Thumbnail", width="small") if generate_thumbnails else None,
                            "Chapitre": st.column_config.TextColumn("Chapitre", width="medium")
                        },
                        column_order=column_order,
                        hide_index=True
                    )

                    if selected_subtitles["selection"]["rows"]:
                        selected_subtitle_idx = selected_subtitles["selection"]["rows"][0]
                        selected_subtitle = filtered_subtitles_df.iloc[selected_subtitle_idx]
                        col_edit_start, col_edit_end, col_edit_text, col_edit_save = st.columns([
                                                                                                1, 1, 3, 1])
                        with col_edit_start:
                            edited_start = st.text_input(
                                "Start", selected_subtitle["Start"], key="edit_start")
                        with col_edit_end:
                            edited_end = st.text_input(
                                "End", selected_subtitle["End"], key="edit_end")
                        with col_edit_text:
                            edited_text = st.text_input(
                                "Text", selected_subtitle["Text"], key="edit_text")
                        with col_edit_save:
                            if st.button("Save Subtitle"):
                                if edited_start != selected_subtitle["Start"] and selected_subtitle_idx > 0:
                                    subtitles_df.at[selected_subtitle_idx -
                                                    1, "End"] = edited_start
                                if edited_end != selected_subtitle["End"] and selected_subtitle_idx < len(subtitles_df) - 1:
                                    subtitles_df.at[selected_subtitle_idx +
                                                    1, "Start"] = edited_end
                                subtitles_df.at[selected_subtitle_idx,
                                                "Start"] = edited_start
                                subtitles_df.at[selected_subtitle_idx,
                                                "End"] = edited_end
                                subtitles_df.at[selected_subtitle_idx,
                                                "Text"] = edited_text
                                save_vtt(vtt_path, subtitles_df, chapters_df)
                                st.rerun()

                        # Ligne avec "Découper" et "Créer chapitre"
                        col_split, col_chapter_title, col_create_chapter = st.columns([
                                                                                      1, 2, 1])
                        with col_split:
                            if st.button(t("video_split")):
                                split_time_ms = parse_timecode_to_ms(
                                    selected_subtitle["Start"])
                                split_video(
                                    selected_video["Full Path"], self.working_dir, split_time_ms=split_time_ms)
                                st.rerun()
                        with col_chapter_title:
                            chapter_title = st.text_input(
                                t("video_chapter_title"), "", key="new_chapter_title")
                        with col_create_chapter:
                            if st.button(t("video_create_chapter")) and chapter_title:
                                new_start = selected_subtitle["Start"]
                                # Trouver la position d’insertion dans chapters_df
                                insert_idx = 0
                                for i, chap in chapters_df.iterrows():
                                    if parse_timecode_to_ms(chap["Start"]) < parse_timecode_to_ms(new_start):
                                        insert_idx = i + 1
                                    else:
                                        break
                                # Ajuster le End du chapitre précédent si existant
                                if insert_idx > 0:
                                    chapters_df.at[insert_idx -
                                                   1, "End"] = new_start
                                # Définir le End du nouveau chapitre : jusqu’au chapitre suivant ou fin de la vidéo
                                if insert_idx < len(chapters_df):
                                    # Début du chapitre suivant
                                    new_end = chapters_df.iloc[insert_idx]["Start"]
                                else:
                                    # Fin de la vidéo (dernier sous-titre)
                                    new_end = subtitles_df["End"].iloc[-1]
                                # Insérer le nouveau chapitre
                                new_chapter = pd.DataFrame(
                                    [{"Start": new_start, "End": new_end, "Title": chapter_title}], index=[insert_idx])
                                chapters_df = pd.concat(
                                    [chapters_df.iloc[:insert_idx], new_chapter, chapters_df.iloc[insert_idx:]]).reset_index(drop=True)
                                save_vtt(vtt_path, subtitles_df, chapters_df)
                                st.rerun()

                        if selected_subtitles["selection"]["rows"] and len(selected_subtitles["selection"]["rows"]) >= 2:
                            start_idx = selected_subtitles["selection"]["rows"][0]
                            end_idx = selected_subtitles["selection"]["rows"][-1]
                            start_time = filtered_subtitles_df.iloc[start_idx]["Start"]
                            end_time = filtered_subtitles_df.iloc[end_idx]["End"]
                            chapter_title_multi = st.text_input(
                                t("video_chapter_title"), "", key="multi_chapter_title")
                            if st.button(t("video_add_chapter")) and chapter_title_multi:
                                new_chapter = pd.DataFrame(
                                    [{"Start": start_time, "End": end_time, "Title": chapter_title_multi}])
                                chapters_df = pd.concat(
                                    [chapters_df, new_chapter], ignore_index=True)
                                save_vtt(vtt_path, subtitles_df, chapters_df)
                                st.rerun()

                if selected_video["Full Path"].endswith(".mp4"):
                    col_left, col_video, col_right = st.columns([1, 2, 1])
                    with col_video:
                        if selected_video["Has Subtitles"] and selected_subtitles["selection"]["rows"]:
                            selected_subtitle_idx = selected_subtitles["selection"]["rows"][0]
                            selected_subtitle = filtered_subtitles_df.iloc[selected_subtitle_idx]
                            st.video(
                                selected_video["Full Path"],
                                start_time=selected_subtitle["Start"],
                                end_time=selected_subtitle["End"],
                                autoplay=True,
                                muted=mute_videos
                            )
                        else:
                            st.video(
                                selected_video["Full Path"],
                                autoplay=True,
                                muted=mute_videos
                            )
                else:
                    st.write("Video preview only available for .mp4 files.")
            else:
                st.write("Select a video to view subtitles.")

    def get_selected_directories(self, working_dir):
        """Gère la sélection des répertoires avec des options spéciales"""
        # Liste tous les répertoires immédiats
        immediate_subdirs = [d for d in os.listdir(working_dir)
                             if os.path.isdir(os.path.join(working_dir, d))]

        # Options spéciales
        options = [
            "ALL_DIRS_RECURSIVE",  # Tous les répertoires (récursif)
            "ROOT_DIR_ONLY"       # Répertoire racine seulement
        ] + immediate_subdirs

        # Traductions pour l'affichage
        display_names = [
            "All directories (recursive)",
            "Root directory only"
        ] + immediate_subdirs

        # Créer un mapping entre noms affichés et valeurs réelles
        options_map = dict(zip(display_names, options))

        # Sélection multiple avec les options spéciales
        selected = st.sidebar.multiselect(
            "Select directories to include",
            display_names,
            default=[display_names[0]] if display_names else None
        )

        if not selected:
            st.warning("Please select at least one directory")
            return None

        # Convertir les sélections en valeurs réelles
        real_selections = [options_map[s] for s in selected]

        # Construire la liste des répertoires à scanner
        dirs_to_scan = []

        if "ALL_DIRS_RECURSIVE" in real_selections:
            # Mode récursif complet
            return [(working_dir, True)]
        elif "ROOT_DIR_ONLY" in real_selections:
            # Juste le répertoire racine
            dirs_to_scan.append((working_dir, False))

        # Ajouter les répertoires sélectionnés individuellement
        for selection in real_selections:
            if selection not in ["ALL_DIRS_RECURSIVE", "ROOT_DIR_ONLY"]:
                dir_path = os.path.join(working_dir, selection)
                dirs_to_scan.append((dir_path, False))

        return dirs_to_scan

    def run(self, config):
        self.working_dir = config.get(self.name, {}).get(
            "video_workdir", t("video_config_workdir_default"))

        self.setup_header()
        selected_model, generate_thumbnails, refresh_thumbnails, selected_extensions, mute_videos, show_end_columns = self.setup_controls()

        # Nouveau sélecteur de répertoires
        st.sidebar.markdown(f"**{t('video_directory_selector')}**")
        dirs_to_scan = self.get_selected_directories(self.working_dir)

        if not dirs_to_scan:
            return

        # Scanner chaque répertoire selon les paramètres
        video_dfs = []
        for dir_path, recursive in dirs_to_scan:
            video_df = scan_videos(
                dir_path, selected_extensions, recursive=recursive)
            video_dfs.append(video_df)

        video_df = pd.concat(video_dfs).reset_index(
            drop=True) if video_dfs else pd.DataFrame()

        if video_df.empty:
            st.write("No videos found with the selected extensions.")
            return

        col1, col2, selected_videos = self.display_videos(video_df)
        self.handle_video_actions(
            col1, selected_videos, video_df, selected_model, config)
        self.handle_chapters(col1, selected_videos, video_df, show_end_columns)
        self.handle_subtitles(col2, selected_videos, video_df, generate_thumbnails,
                              refresh_thumbnails, mute_videos, show_end_columns)


if __name__ == "__main__":
    st.write("Video Plugin standalone test")
