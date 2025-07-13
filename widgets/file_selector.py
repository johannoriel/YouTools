from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import os
from pathlib import Path
from datetime import datetime

# Ajout des traductions
translations["en"].update({
    "file_selector_title": "File Selector",
    "file_selector_filter_label": "Filter files (substring)",
    "file_selector_extensions_label": "Select file extensions",
    "file_selector_show_details": "Show file size and date",
    "file_selector_show_video_info": "Show video duration and resolution",
    "file_selector_select_files": "Select files to process",
    "file_selector_file_column": "File",
    "file_selector_size_column": "Size (MB)",
    "file_selector_date_column": "Last Modified",
    "file_selector_duration_column": "Duration",
    "file_selector_resolution_column": "Resolution",
})

translations["fr"].update({
    "file_selector_title": "Sélecteur de fichiers",
    "file_selector_filter_label": "Filtrer les fichiers (sous-chaîne)",
    "file_selector_extensions_label": "Sélectionner les extensions de fichiers",
    "file_selector_show_details": "Afficher la taille et la date des fichiers",
    "file_selector_show_video_info": "Afficher la durée et la résolution des vidéos",
    "file_selector_select_files": "Sélectionner les fichiers à traiter",
    "file_selector_file_column": "Fichier",
    "file_selector_size_column": "Taille (Mo)",
    "file_selector_date_column": "Dernière modification",
    "file_selector_duration_column": "Durée",
    "file_selector_resolution_column": "Résolution",
})

class FileSelectorWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.work_directory = self.work_dir()

    def get_video_metadata(self, file_path: str) -> tuple[str, str]:
        """Calcule la durée et la résolution d'une vidéo avec mise en cache."""
        if 'video_metadata' not in st.session_state:
            st.session_state.video_metadata = {}

        if file_path in st.session_state.video_metadata:
            return st.session_state.video_metadata[file_path]

        try:
            from moviepy import VideoFileClip
            video = VideoFileClip(file_path)
            duration = video.duration
            duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}"  # Format MM:SS
            resolution = video.size
            resolution_str = f"{resolution[0]}x{resolution[1]}"
            video.close()
            st.session_state.video_metadata[file_path] = (duration_str, resolution_str)
            return duration_str, resolution_str
        except Exception as e:
            return f"Erreur ({e})", f"Erreur ({e})"

    def list_files(self, extensions: list, mode: str = "simple") -> list:
        """Liste les fichiers dans le répertoire de travail avec les extensions spécifiées."""
        files = []
        for file in os.listdir(self.work_directory):
            file_path = os.path.join(self.work_directory, file)
            if os.path.isfile(file_path):
                if mode == "video" and Path(file).suffix.lower() in extensions:
                    files.append((file, file_path))
                elif mode == "simple" and (not extensions or Path(file).suffix.lower() in extensions):
                    files.append((file, file_path))
        return sorted(files, key=lambda x: x[0].lower())

    def display(self, mode: str = "simple", allowed_extensions: list = None):
            """Affiche les fichiers et renvoie la liste des fichiers sélectionnés."""
            st.subheader(t("file_selector_title"))

            # Liste des extensions par défaut selon le mode
            if mode == "video":
                default_extensions = ['.mp4', '.ogg', '.mov', '.avi', '.mkv']
            else:
                default_extensions = ['.mp4', '.ogg', '.mov', '.avi', '.mkv', '.txt', '.jpg', '.png', '.pdf']

            allowed_extensions = allowed_extensions or default_extensions

            # Créer des colonnes pour aligner les filtres sur une seule ligne
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

            # Sélection des extensions
            with col1:
                selected_extensions = st.multiselect(
                    t("file_selector_extensions_label"),
                    options=allowed_extensions,
                    default=allowed_extensions,
                    key=f"{self.prefix}_extensions"
                )

            # Filtre de recherche
            with col2:
                filter_text = st.text_input(
                    t("file_selector_filter_label"),
                    key=f"{self.prefix}_filter"
                )

            # Case à cocher pour les détails (taille et date)
            with col3:
                show_details = st.checkbox(
                    t("file_selector_show_details"),
                    key=f"{self.prefix}_show_details"
                )

            # Case à cocher pour les informations vidéo (durée et résolution)
            show_video_info = False
            if mode == "video":
                with col4:
                    show_video_info = st.checkbox(
                        t("file_selector_show_video_info"),
                        key=f"{self.prefix}_show_video_info"
                    )

            # Lister les fichiers
            files = self.list_files(selected_extensions, mode)
            filtered_files = [(file, path) for file, path in files if filter_text.lower() in file.lower()]

            # Préparer les données pour l'affichage
            file_data = []
            for file, full_path in filtered_files:
                file_info = {"File": file, "Full Path": full_path}
                if show_details:
                    file_size = os.path.getsize(full_path) / (1024 * 1024)  # Taille en MB
                    file_date = datetime.fromtimestamp(os.path.getmtime(full_path)).strftime('%Y-%m-%d %H:%M:%S')
                    file_info["Size (MB)"] = f"{file_size:.2f}"
                    file_info["Last Modified"] = file_date
                if show_video_info and mode == "video":
                    duration, resolution = self.get_video_metadata(full_path)
                    file_info["Duration"] = duration
                    file_info["Resolution"] = resolution
                file_data.append(file_info)

            # Configurer les colonnes pour la grille
            column_config = {
                "File": st.column_config.TextColumn(t("file_selector_file_column")),
                "Full Path": None,  # Cacher la colonne Full Path
            }
            if show_details:
                column_config.update({
                    "Size (MB)": st.column_config.TextColumn(t("file_selector_size_column")),
                    "Last Modified": st.column_config.TextColumn(t("file_selector_date_column")),
                })
            if show_video_info and mode == "video":
                column_config.update({
                    "Duration": st.column_config.TextColumn(t("file_selector_duration_column")),
                    "Resolution": st.column_config.TextColumn(t("file_selector_resolution_column")),
                })

            # Afficher la grille
            selected_rows = st.dataframe(
                file_data,
                column_config=column_config,
                use_container_width=True,
                height=400,
                selection_mode="multi-row",
                on_select="rerun",
                key=f"{self.prefix}_file_selection"
            )

            # Récupérer les fichiers sélectionnés
            selected_indices = selected_rows.get('selection', {}).get('rows', [])
            selected_files = [file_data[i]["Full Path"] for i in selected_indices]

            return selected_files
