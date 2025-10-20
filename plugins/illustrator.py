from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
import os
import shutil
from widgets.media_selector import media_selector, remote_media_selector, ALL_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS
from lib.assets_api import PexelsAPI, GoogleImageAPI, DuckDuckGoImageAPI, VlipsyAPI, asset_memory_download, asset_download
from io import BytesIO
from lib.youtube_api import YoutubeAPI
import re
from lib.video_utils import normalize_audio


# Constantes pour les types de média
MEDIA_TYPE_ALL = "All"
MEDIA_TYPE_IMAGES = "Images"
MEDIA_TYPE_VIDEOS = "Videos"
MEDIA_TYPE_AUDIO = "Audio"

# Traductions
translations["en"].update({
    "illustrator_tab": "Illustrator",
    "illustrator_header": "Video Assets Manager",
    "illustrator_current_tab": "Current Assets",
    "illustrator_stored_tab": "Stored Assets",
    "illustrator_search_tab": "Search New Assets",
    "illustrator_current_dir": "Current Assets Directory",
    "illustrator_stored_dir": "Stored Assets Directory",
    "illustrator_delete_all": "Delete All",
    "illustrator_delete": "Delete",
    "illustrator_add_to_current": "Add to Current Assets",
    "illustrator_search_api": "Select API",
    "illustrator_search_keywords": "Search Keywords",
    "illustrator_search_button": "Search",
    "illustrator_destination_folder": "Destination Folder",
    "illustrator_create_folder": "Create New Folder",
    "illustrator_download": "Download Selected",
    "illustrator_no_assets": "No assets found",
    "illustrator_config_current_dir": "Current Assets Directory",
    "illustrator_config_stored_dir": "Stored Assets Directory",
    "illustrator_refresh": "Refresh",
    "illustrator_media_type": "Media Type",
    "illustrator_photos": "Photos",
    "illustrator_videos": "Videos",
    "illustrator_both": "Both",
    "All": "All",
    "Images": "Images",
    "Videos": "Videos",
    "Audio": "Audio",
    "Filter by type": "Filter by type",
    "download_to_stored": "Download to Stored Assets",
    "download_to_current": "Download to Current Assets",
    "download_to_both": "Download to Both",
    "illustrator_youtube_tab": "YouTube CC Videos",
    "illustrator_youtube_search": "Search YouTube CC Videos",
    "illustrator_youtube_duration": "Duration",
    "illustrator_youtube_select": "Select Videos",
    "illustrator_youtube_cut": "Cut Video Segments",
    "illustrator_youtube_start": "Start Time (s)",
    "illustrator_youtube_end": "End Time (s)",
    "illustrator_youtube_process": "Process Selected",
    "illustrator_youtube_download": "Download Segments"
})

translations["fr"].update({
    "illustrator_tab": "Illustrateur",
    "illustrator_header": "Gestionnaire d'Assets Vidéo",
    "illustrator_current_tab": "Assets Actuels",
    "illustrator_stored_tab": "Assets Stockés",
    "illustrator_search_tab": "Rechercher Nouveaux Assets",
    "illustrator_current_dir": "Répertoire des Assets Actuels",
    "illustrator_stored_dir": "Répertoire des Assets Stockés",
    "illustrator_delete_all": "Tout Supprimer",
    "illustrator_delete": "Supprimer",
    "illustrator_add_to_current": "Ajouter aux Assets Actuels",
    "illustrator_search_api": "Sélectionner une API",
    "illustrator_search_keywords": "Mots-clés de Recherche",
    "illustrator_search_button": "Rechercher",
    "illustrator_destination_folder": "Dossier de Destination",
    "illustrator_create_folder": "Créer un Nouveau Dossier",
    "illustrator_download": "Télécharger la Sélection",
    "illustrator_no_assets": "Aucun asset trouvé",
    "illustrator_config_current_dir": "Répertoire des Assets Actuels",
    "illustrator_config_stored_dir": "Répertoire des Assets Stockés",
    "illustrator_refresh": "Rafraîchir",
    "illustrator_media_type": "Type de média",
    "illustrator_photos": "Photos",
    "illustrator_videos": "Vidéos",
    "illustrator_both": "Les deux",
    "All": "Tous",
    "Images": "Images",
    "Videos": "Vidéos",
    "Audio": "Audio",
    "Filter by type": "Filtrer par type",
    "download_to_stored": "Télécharger vers Assets Stockés",
    "download_to_current": "Télécharger vers Assets Actuels",
    "download_to_both": "Télécharger vers les Deux",
    "illustrator_youtube_tab": "Vidéos YouTube CC",
    "illustrator_youtube_search": "Rechercher vidéos CC YouTube",
    "illustrator_youtube_duration": "Durée",
    "illustrator_youtube_select": "Sélectionner vidéos",
    "illustrator_youtube_cut": "Découper les vidéos",
    "illustrator_youtube_start": "Temps début (s)",
    "illustrator_youtube_end": "Temps fin (s)",
    "illustrator_youtube_process": "Traiter sélection",
    "illustrator_youtube_download": "Télécharger segments"
})


class IllustratorPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.apis = {
            "pexels": PexelsAPI(),
            "google": GoogleImageAPI(),
            "duckduckgo": DuckDuckGoImageAPI(),
            "vlipsy": VlipsyAPI()
        }

    def get_config_fields(self):
        """Configuration du plugin"""
        return {
            "illustrator_current_dir": {
                "type": "text",
                "label": t("illustrator_config_current_dir"),
                "default": "~/Vidéos/Assets"
            },
            "illustrator_stored_dir": {
                "type": "text",
                "label": t("illustrator_config_stored_dir"),
                "default": "~/Vidéos/OBS/Illustrations"
            },
            "pexels_api_key": {
                "type": "text",
                "label": "Pexels API Key",
                "default": ""
            },
            "google_cx": {
                "type": "text",
                "label": "Google Custom Search Engine ID",
                "default": "",
                "help": "Required for Google Image Search"
            },
            "vlipsy_api_key": {
                "type": "text",
                "label": "Vlipsy API Key",
                "default": "vl_hFxn07bG43d0n9t"
            }
        }

    def get_tabs(self):
        """Définition des onglets"""
        return [
            {"name": t("illustrator_current_tab"),
             "plugin": "illustratorplugin", "tab": "current"},
            {"name": t("illustrator_stored_tab"),
             "plugin": "illustratorplugin", "tab": "stored"},
            {"name": "Pexels",  # Changed from search tab to specific API tabs
             "plugin": "illustratorplugin", "tab": "pexels"},
            {"name": "Google",
             "plugin": "illustratorplugin", "tab": "google"},
            {"name": "DuckDuckGo",
             "plugin": "illustratorplugin", "tab": "duckduckgo"},
            {"name": "Vlipsy",  # Nouvel onglet
             "plugin": "illustratorplugin", "tab": "vlipsy"},
            {"name": t("illustrator_youtube_tab"),
             "plugin": "illustratorplugin", "tab": "youtube"}
        ]

    def expand_path(self, path):
        """Convertit les chemins avec ~ en chemins absolus"""
        return os.path.expanduser(path)

    def get_subdirectories(self, base_dir):
        """Liste les sous-répertoires d'un répertoire de base"""
        expanded_dir = self.expand_path(base_dir)
        if not os.path.exists(expanded_dir):
            return []
        return [d for d in os.listdir(expanded_dir) if os.path.isdir(os.path.join(expanded_dir, d))]

    def copy_to_current(self, src_path):
        """Copie un fichier vers le répertoire des assets courants"""
        current_dir = self.expand_path(self.config.get(self.name, {}).get(
            "illustrator_current_dir", t("illustrator_config_default_current")))
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        filename = os.path.basename(src_path)
        dest_path = os.path.join(current_dir, filename)
        shutil.copy2(src_path, dest_path)
        return dest_path

    def delete_asset(self, path):
        """Supprime un asset"""
        try:
            os.remove(path)
            return True
        except Exception as e:
            st.error(f"Error deleting file: {str(e)}")
            return False

    def show_media_preview(self, media_data, media_type=None):
        """Affiche une prévisualisation du média dans une colonne centrale"""
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:  # Colonne centrale pour la prévisualisation
            st.subheader("Preview")

            # Cas des BytesIO (données en mémoire)
            if isinstance(media_data, BytesIO):
                if media_type == 'photo':
                    st.image(media_data)
                elif media_type == 'video':
                    st.video(media_data, format="video/mp4",
                             autoplay=True, muted=True)

            # Cas des chemins de fichiers locaux
            elif isinstance(media_data, str):
                if media_data.lower().endswith(IMAGE_EXTENSIONS):
                    st.image(media_data)
                elif media_data.lower().endswith(VIDEO_EXTENSIONS):
                    st.video(media_data, format="video/mp4",
                             autoplay=True, muted=True)
                elif media_data.lower().endswith(AUDIO_EXTENSIONS):
                    st.audio(media_data)

            # Cas des résultats de recherche (dictionnaire)
            elif isinstance(media_data, dict):
                if media_data['original_data']['type'] == 'photo':
                    st.image(media_data['url'])
                else:
                    st.video(
                        media_data['url'], format="video/mp4", autoplay=True, muted=True)

    def folder_selector_with_creation(self, base_dir, key=None):
        """Sélection de dossier avec option de création de nouveau dossier"""
        subdirs = self.get_subdirectories(base_dir)
        selected_subdir = st.selectbox(
            t("illustrator_destination_folder"),
            subdirs + ["[Create New Folder]"],
            key=f"folder_selector_{key}" if key else None
        )

        if selected_subdir == "[Create New Folder]":
            new_folder = st.text_input(
                t("illustrator_create_folder"),
                key=f"new_folder_{key}" if key else None
            )
            if new_folder and st.button("Create", key=f"create_{key}" if key else None):
                new_path = os.path.join(base_dir, new_folder)
                os.makedirs(new_path, exist_ok=True)
                st.success(f"Folder created: {new_path}")
                st.rerun()
            return None

        return selected_subdir

    def run_current_assets_tab(self, config):
        """Onglet des assets courants"""
        st.header(t("illustrator_current_tab"))
        current_dir = self.expand_path(config.get(self.name, {}).get(
            "illustrator_current_dir", t("illustrator_config_default_current")))

        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
            st.info(f"Created directory: {current_dir}")
            return

        # Filtre par type de média
        media_types = {
            MEDIA_TYPE_ALL: ALL_EXTENSIONS,
            MEDIA_TYPE_IMAGES: IMAGE_EXTENSIONS,
            MEDIA_TYPE_VIDEOS: VIDEO_EXTENSIONS,
            MEDIA_TYPE_AUDIO: AUDIO_EXTENSIONS
        }
        selected_type = st.selectbox(
            "Filter by type", list(media_types.keys()))
        media_extensions = media_types[selected_type]

        media_files = [f for f in os.listdir(current_dir) if os.path.splitext(f)[
            1].lower() in media_extensions]

        if not media_files:
            st.info(t("illustrator_no_assets"))
            return

        # Affichage des assets avec option de suppression
        selected = media_selector(
            [current_dir], media_extensions, "current", st)

        # Boutons de suppression et rafraîchissement sur la même ligne
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(t("illustrator_delete_all")):
                for f in media_files:
                    self.delete_asset(os.path.join(current_dir, f))
                st.rerun()
        with col2:
            if selected and st.button(t("illustrator_delete")):
                if self.delete_asset(selected):
                    st.rerun()
        with col3:
            if st.button(t("illustrator_refresh")):
                st.rerun()

        return selected

    def run_stored_assets_tab(self, config):
        """Onglet des assets stockés"""
        st.header(t("illustrator_stored_tab"))
        stored_dir = self.expand_path(config.get(self.name, {}).get(
            "illustrator_stored_dir", t("illustrator_config_default_stored")))

        if not os.path.exists(stored_dir):
            os.makedirs(stored_dir)
            st.info(f"Created directory: {stored_dir}")
            return

        # Filtre par type de média
        media_types = {
            MEDIA_TYPE_ALL: ALL_EXTENSIONS,
            MEDIA_TYPE_IMAGES: IMAGE_EXTENSIONS,
            MEDIA_TYPE_VIDEOS: VIDEO_EXTENSIONS,
            MEDIA_TYPE_AUDIO: AUDIO_EXTENSIONS
        }

        # Sélection du sous-répertoire et type de média sur la même ligne
        col1, col2 = st.columns(2)
        with col1:
            subdirs = self.get_subdirectories(stored_dir)
            selected_subdir = st.selectbox(
                "Select folder", subdirs + ["[Create New Folder]"], key="stored_folder")
        with col2:
            selected_type = st.selectbox(
                "Filter by type", list(media_types.keys()), key="stored_filter")

        media_extensions = media_types[selected_type]

        if selected_subdir == "[Create New Folder]":
            new_folder = st.text_input(t("illustrator_create_folder"))
            if new_folder and st.button("Create"):
                new_path = os.path.join(stored_dir, new_folder)
                os.makedirs(new_path, exist_ok=True)
                st.success(f"Folder created: {new_path}")
                st.rerun()
            return

        # Affichage des médias du sous-répertoire sélectionné
        if selected_subdir:
            selected_dir = os.path.join(stored_dir, selected_subdir)
            selected_media = media_selector(
                [selected_dir], media_extensions, "stored", st)

            if selected_media and st.button(t("illustrator_add_to_current")):
                dest_path = self.copy_to_current(selected_media)
                st.success(f"Added to current assets: {dest_path}")

            return selected_media

    def _save_media_options(self, media_buffer, media_name, media_type, stored_dir, current_dir, prefix=""):
        """Affiche les options de sauvegarde communes pour tous les moteurs de recherche"""
        st.markdown("---")
        st.subheader("Save Options")

        # Nettoyer le media_name pour ne garder que lettres, chiffres, - et _
        cleaned_media_name = re.sub(r'[^\w\-_]', '-', media_name)
        cleaned_media_name = re.sub(r'-+', '-', cleaned_media_name)
        cleaned_media_name = cleaned_media_name.strip('-')

        # Récupérer le dernier mot-clé utilisé pour cette recherche spécifique
        search_keyword = st.session_state.get(
            f"last_search_keyword_{prefix}", "")

        # Ajouter le mot-clé de recherche si disponible et non présent
        if (search_keyword and
                search_keyword.lower() not in cleaned_media_name.lower()):
            # Nettoyer le mot-clé pour le nom de fichier
            clean_keyword = re.sub(r'[^\w\-_]', '-', search_keyword)
            clean_keyword = re.sub(r'-+', '-', clean_keyword).strip('-')
            cleaned_media_name = f"{cleaned_media_name}_{clean_keyword}"

        ext = '.mp4' if media_type == 'video' else '.jpg'
        reference_audio_path = self.config.get(
            "movied", {}).get("movied_reference_audio", "")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button(t("download_to_current"), key=f"download_current_{prefix}_{cleaned_media_name}"):
                try:
                    os.makedirs(current_dir, exist_ok=True)
                    filename = f"{cleaned_media_name}{ext}"
                    filepath = os.path.join(current_dir, filename)

                    with open(filepath, 'wb') as f:
                        media_buffer.seek(0)
                        f.write(media_buffer.read())

                    if media_type == 'video':
                        with st.spinner("Normalizing audio..."):
                            normalize_audio(
                                filepath, reference_audio_path, make_backup=False)

                    st.success(f"Added to current assets: {filepath}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with col2:
            selected_subdir = self.folder_selector_with_creation(
                stored_dir, f"save_{prefix}_{cleaned_media_name}")

        with col3:
            if selected_subdir and st.button(t("download_to_stored"), key=f"download_stored_{prefix}_{cleaned_media_name}"):
                try:
                    filename = f"{cleaned_media_name}{ext}"
                    filepath = os.path.join(
                        stored_dir, selected_subdir, filename)

                    with open(filepath, 'wb') as f:
                        media_buffer.seek(0)
                        f.write(media_buffer.read())

                    if media_type == 'video':
                        with st.spinner("Normalizing audio..."):
                            normalize_audio(
                                filepath, reference_audio_path, make_backup=False)

                    st.success(f"Saved to stored assets: {filepath}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with col4:
            if selected_subdir and st.button(t("download_to_both"), key=f"download_to_both_{prefix}_{cleaned_media_name}"):
                try:
                    # Save to stored
                    filename = f"{cleaned_media_name}{ext}"
                    stored_path = os.path.join(
                        stored_dir, selected_subdir, filename)
                    with open(stored_path, 'wb') as f:
                        media_buffer.seek(0)
                        f.write(media_buffer.read())

                    if media_type == 'video':
                        with st.spinner("Normalizing audio..."):
                            normalize_audio(
                                stored_path, reference_audio_path, make_backup=False)

                    # Save to current
                    os.makedirs(current_dir, exist_ok=True)
                    current_path = os.path.join(current_dir, filename)
                    with open(current_path, 'wb') as f:
                        media_buffer.seek(0)
                        f.write(media_buffer.read())

                    if media_type == 'video':
                        with st.spinner("Normalizing audio..."):
                            normalize_audio(
                                current_path, reference_audio_path, make_backup=False)

                    st.success(
                        f"Saved to stored assets: {stored_path}\n"
                        f"Added to current assets: {current_path}"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    def _handle_search_results(self, api_name, results, config, prefix="", search_keyword=""):
        """Gère l'affichage et la sélection des résultats de recherche (commun à tous les moteurs)"""
        # Stocker le mot-clé utilisé pour cette recherche
        st.session_state[f"last_search_keyword_{prefix}"] = search_keyword

        stored_dir = self.expand_path(config.get(self.name, {}).get(
            "illustrator_stored_dir", t("illustrator_config_default_stored")))
        current_dir = self.expand_path(config.get(self.name, {}).get(
            "illustrator_current_dir", t("illustrator_config_default_current")))

        # Format results for the media selector
        formatted_results = []
        for item in results:
            formatted_results.append({
                'url': item['url'],
                'name': item.get('name', f"Media {item['id']}"),
                'date': item.get('date', 0),
                'original_data': item
            })

        # Store results in session state with prefix
        st.session_state[f"search_results_{prefix}"] = formatted_results
        st.session_state[f"selected_item_{prefix}"] = None
        st.session_state[f"media_buffer_{prefix}"] = None
        st.session_state[f"media_type_{prefix}"] = None

    def _display_search_results(self, api_name, prefix=""):
        """Affiche les résultats de recherche et gère la prévisualisation"""
        if f"search_results_{prefix}" not in st.session_state or not st.session_state[f"search_results_{prefix}"]:
            return

        # Sélection du média
        new_selection = remote_media_selector(
            st.session_state[f"search_results_{prefix}"],
            f"search_{prefix}_{api_name}"
        )

        # Si la sélection a changé, réinitialiser le buffer
        if new_selection != st.session_state[f"selected_item_{prefix}"]:
            st.session_state[f"selected_item_{prefix}"] = new_selection
            st.session_state[f"media_buffer_{prefix}"] = None
            st.session_state[f"media_type_{prefix}"] = None

        # Téléchargement pour prévisualisation
        if st.session_state[f"selected_item_{prefix}"] and not st.session_state[f"media_buffer_{prefix}"]:
            with st.spinner("Downloading for preview..."):
                try:
                    url = st.session_state[f"selected_item_{prefix}"]['original_data']
                    st.write(url['original_url'])
                    buffer, media_type = asset_memory_download(url)
                    st.session_state[f"media_buffer_{prefix}"] = buffer
                    st.session_state[f"media_type_{prefix}"] = media_type
                except Exception as e:
                    st.error(f"Preview download error: {str(e)}")
                    # raise e

        # Prévisualisation
        if st.session_state[f"media_buffer_{prefix}"]:
            self.show_media_preview(
                st.session_state[f"media_buffer_{prefix}"],
                st.session_state[f"media_type_{prefix}"]
            )

        # Options de sauvegarde
        if st.session_state[f"selected_item_{prefix}"] and st.session_state[f"media_buffer_{prefix}"]:
            stored_dir = self.expand_path(self.config.get(self.name, {}).get(
                "illustrator_stored_dir", t("illustrator_config_default_stored")))
            current_dir = self.expand_path(self.config.get(self.name, {}).get(
                "illustrator_current_dir", t("illustrator_config_default_current")))

            self._save_media_options(
                st.session_state[f"media_buffer_{prefix}"],
                st.session_state[f"selected_item_{prefix}"]['name'],
                st.session_state[f"selected_item_{prefix}"]['original_data']['type'],
                stored_dir,
                current_dir,
                prefix
            )

    def run_pexels_tab(self, config):
        """Onglet de recherche Pexels"""
        st.header("Pexels Search")

        if not config.get(self.name, {}).get("pexels_api_key"):
            st.error("API key for Pexels is not configured")
            return

        # Initialisation des variables de session
        if 'search_results' not in st.session_state:
            st.session_state.search_results = None

        # Options de recherche
        col1, col2 = st.columns(2)
        with col1:
            keywords = st.text_input(
                t("illustrator_search_keywords"),
                key="pexels_keywords",
                on_change=lambda: setattr(
                    st.session_state, 'pexels_search_triggered', True)
            )
        with col2:
            media_type = st.selectbox(
                t("illustrator_media_type"),
                ["photos", "videos", "both"],
                format_func=lambda x: t(f"illustrator_{x}")
            )

        # Recherche soit avec Enter soit avec le bouton
        if st.button(t("illustrator_search_button"), key="pexels_search") or getattr(st.session_state, 'pexels_search_triggered', False):
            st.session_state.pexels_search_triggered = False
            if keywords:
                with st.spinner("Searching Pexels..."):
                    try:
                        results = []
                        if media_type in ["photos", "both"]:
                            photos = self.apis["pexels"].search(
                                remove_quotes(keywords),
                                config.get(self.name, {}).get(
                                    "pexels_api_key"),
                                "photos"
                            )
                            results.extend(photos)
                        if media_type in ["videos", "both"]:
                            videos = self.apis["pexels"].search(
                                remove_quotes(keywords),
                                config.get(self.name, {}).get(
                                    "pexels_api_key"),
                                "videos"
                            )
                            results.extend(videos)

                        self._handle_search_results(
                            "pexels", results, config, prefix="pexels", search_keyword=keywords)
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                        raise e

        # Affichage des résultats
        self._display_search_results("pexels", prefix="pexels")

    def run_google_tab(self, config):
        """Onglet de recherche Google"""
        st.header("Google Search")

        if not config.get('common', {}).get('youtube_api_key'):
            st.error(
                "Google API key (YouTube API key) is not configured in common settings")
            return
        if not config.get(self.name, {}).get('google_cx'):
            st.error("Google Custom Search Engine ID (cx) is not configured")
            return

        # Initialisation des variables de session
        if 'search_results' not in st.session_state:
            st.session_state.search_results = None

        # Options de recherche
        keywords = st.text_input(
            t("illustrator_search_keywords"),
            key="google_keywords",
            on_change=lambda: setattr(
                st.session_state, 'google_search_triggered', True)
        )

        # Recherche soit avec Enter soit avec le bouton
        if st.button(t("illustrator_search_button"), key="google_search") or getattr(st.session_state, 'google_search_triggered', False):
            st.session_state.google_search_triggered = False
            if keywords:
                with st.spinner("Searching Google..."):
                    try:
                        results = self.apis["google"].search(
                            remove_quotes(keywords),
                            config.get('common', {}).get('youtube_api_key'),
                            config.get(self.name, {}).get('google_cx')
                        )
                        self._handle_search_results(
                            "google", results, config, prefix="google", search_keyword=keywords)
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                        raise e

        # Affichage des résultats
        self._display_search_results("google", prefix="google")

    def run_duckduckgo_tab(self, config):
        """Onglet de recherche DuckDuckGo"""
        st.header("DuckDuckGo Search")

        # Initialisation des variables de session
        if 'search_results' not in st.session_state:
            st.session_state.search_results = None

        # Options de recherche
        keywords = st.text_input(
            t("illustrator_search_keywords"),
            key="duckduckgo_keywords",
            on_change=lambda: setattr(
                st.session_state, 'duckduckgo_search_triggered', True)
        )

        # Recherche soit avec Enter soit avec le bouton
        if st.button(t("illustrator_search_button"), key="duckduckgo_search") or getattr(st.session_state, 'duckduckgo_search_triggered', False):
            st.session_state.duckduckgo_search_triggered = False
            if keywords:
                with st.spinner("Searching DuckDuckGo..."):
                    try:
                        results = self.apis["duckduckgo"].search(
                            remove_quotes(keywords)
                        )
                        self._handle_search_results(
                            "duckduckgo", results, config, prefix="duckduckgo", search_keyword=keywords)
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                        raise e

        # Affichage des résultats
        self._display_search_results("duckduckgo", prefix="duckduckgo")

    def run_youtube_assets_tab(self, config):
        """Onglet de recherche d'assets vidéo sur YouTube"""
        st.header(t("illustrator_youtube_tab"))
        stored_dir = self.expand_path(config.get(self.name, {}).get(
            "illustrator_stored_dir", t("illustrator_config_default_stored")))
        current_dir = self.expand_path(config.get(self.name, {}).get(
            "illustrator_current_dir", t("illustrator_config_default_current")))

        # Initialisation des variables de session
        if 'youtube_results' not in st.session_state:
            st.session_state.youtube_results = []
        if 'selected_youtube_video' not in st.session_state:
            st.session_state.selected_youtube_video = None
        if 'youtube_video_buffer' not in st.session_state:
            st.session_state.youtube_video_buffer = None
        if 'processed_segment' not in st.session_state:
            st.session_state.processed_segment = None

        # Initialisation de l'API YouTube
        youtube_api = YoutubeAPI(config)

        # Options de recherche
        col1, col2 = st.columns(2)
        with col1:
            keywords = st.text_input(
                t("illustrator_search_keywords"), key="youtube_keywords")
        with col2:
            creative_commons = st.checkbox("Creative Commons only", value=True)

        if st.button(t("illustrator_youtube_search")):
            with st.spinner("Searching YouTube..."):
                try:
                    st.session_state.youtube_results = youtube_api.search_assets(
                        keywords,
                        creative_commons=creative_commons
                    )
                    st.session_state.selected_youtube_video = None
                    st.session_state.youtube_video_buffer = None
                    st.session_state.processed_segment = None
                except Exception as e:
                    st.error(f"Search error: {str(e)}")

        # Affichage des résultats
        if st.session_state.youtube_results:

            # Sélection de la vidéo avec le media_selector standard
            st.session_state.selected_youtube_video = remote_media_selector(
                st.session_state.youtube_results,
                "youtube",
                st
            )

            # Options de découpage et traitement
            if st.session_state.selected_youtube_video:
                st.markdown("---")
                st.subheader(t("illustrator_youtube_cut"))

                # Afficher la durée totale de la vidéo
                total_duration = st.session_state.selected_youtube_video['original_data']['duration']
                st.write(
                    f"Durée totale: {youtube_api._format_duration(total_duration)}")

                col1, col2 = st.columns(2)
                with col1:
                    start_time = st.text_input(
                        t("illustrator_youtube_start"),
                        value="00:00:00.000",
                        help="Format HH:MM:SS.mmm ou MM:SS.mmm ou SS.mmm"
                    )
                with col2:
                    end_time = st.text_input(
                        t("illustrator_youtube_end"),
                        value="00:00:05.000",
                        help="Format HH:MM:SS.mmm ou MM:SS.mmm ou SS.mmm"
                    )

                if st.button(t("illustrator_youtube_process")):
                    try:
                        # Convertir les timecodes en secondes
                        start_seconds = youtube_api._timecode_to_seconds(
                            start_time)
                        end_seconds = youtube_api._timecode_to_seconds(
                            end_time)

                        # Validation et ajustement des timecodes
                        if end_seconds == 0 or end_seconds > total_duration:
                            end_seconds = total_duration

                        if start_seconds >= end_seconds:
                            st.error(
                                "Le temps de fin doit être après le temps de début")
                        else:
                            with st.spinner("Processing video..."):
                                try:
                                    # Télécharger la vidéo complète
                                    st.session_state.youtube_video_buffer = youtube_api.download_asset(
                                        st.session_state.selected_youtube_video
                                    )

                                    # Découper le segment
                                    st.session_state.processed_segment = youtube_api.process_video_segment(
                                        st.session_state.youtube_video_buffer,
                                        start_seconds,
                                        end_seconds
                                    )
                                    st.success(
                                        f"Video segment processed! ({youtube_api._format_duration(end_seconds - start_seconds)})")
                                except Exception as e:
                                    st.error(
                                        f"Error processing video: {str(e)}")
                                    raise e
                    except ValueError as e:
                        st.error(f"Format de timecode invalide : {str(e)}")

                # Prévisualisation du segment
                if st.session_state.processed_segment:
                    st.markdown("---")
                    st.video(st.session_state.processed_segment,
                             format="video/mp4")
                    st.subheader("Save Options")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button(t("download_to_current"), key="youtube_download_current"):
                            self._save_youtube_segment(
                                current_dir, None, st.session_state.selected_youtube_video)

                    with col2:
                        selected_subdir = self.folder_selector_with_creation(
                            stored_dir, "youtube_save")

                    with col3:
                        if selected_subdir and st.button(t("download_to_stored"), key="youtube_download_stored"):
                            self._save_youtube_segment(
                                stored_dir, selected_subdir, st.session_state.selected_youtube_video)

                    with col4:
                        if selected_subdir and st.button(t("download_to_both"), key="youtube_download_both"):
                            self._save_youtube_segment(
                                current_dir, None, st.session_state.selected_youtube_video)
                            self._save_youtube_segment(
                                stored_dir, selected_subdir, st.session_state.selected_youtube_video)

    def _save_youtube_segment(self, base_dir: str, subdir: str, video_data: dict) -> str:
        """Sauvegarde un segment vidéo YouTube avec comme nom le titre de la vidéo"""
        try:
            target_dir = os.path.join(base_dir, subdir) if subdir else base_dir
            os.makedirs(target_dir, exist_ok=True)

            # Créer un nom de fichier propre à partir du titre de la vidéo
            title = video_data['original_data']['title']
            # Limite à 100 caractères
            clean_title = re.sub(r'[^\w\-_\. ]', '_', title)[:100]

            # Récupérer le dernier mot-clé utilisé pour YouTube
            search_keyword = st.session_state.get(
                'last_search_keyword_youtube', '')
            if (search_keyword and
                    search_keyword.lower() not in clean_title.lower()):
                clean_keyword = re.sub(r'[^\w\-_]', '_', search_keyword)
                clean_keyword = re.sub(r'_+', '_', clean_keyword).strip('_')
                clean_title = f"{clean_title}_{clean_keyword}"

            filename = f"{clean_title}.mp4"
            filepath = os.path.join(target_dir, filename)

            with open(filepath, 'wb') as f:
                f.write(st.session_state.processed_segment.getvalue())

            # Normalisation audio pour les vidéos YouTube
            reference_audio_path = self.config.get(
                "movied", {}).get("movied_reference_audio", "")
            with st.spinner("Normalizing audio..."):
                normalize_audio(filepath, reference_audio_path,
                                make_backup=False)

            st.success(f"Saved to {filepath}")
            return filepath
        except Exception as e:
            st.error(f"Error saving video: {str(e)}")
            return None

    def run_vlipsy_tab(self, config):
        """Onglet de recherche Vlipsy"""
        st.header("Vlipsy Search")

        # Vérification de la clé API
        vlipsy_api_key = config.get(self.name, {}).get(
            "vlipsy_api_key", "vl_hFxn07bG43d0n9t")
        if not vlipsy_api_key:
            st.error("API key for Vlipsy is not configured")
            return

        # Initialisation des variables de session
        if 'vlipsy_results' not in st.session_state:
            st.session_state.vlipsy_results = None

        # Options de recherche
        keywords = st.text_input(
            t("illustrator_search_keywords"),
            key="vlipsy_keywords",
            on_change=lambda: setattr(
                st.session_state, 'vlipsy_search_triggered', True)
        )

        # Recherche soit avec Enter soit avec le bouton
        if st.button(t("illustrator_search_button"), key="vlipsy_search") or getattr(st.session_state, 'vlipsy_search_triggered', False):
            st.session_state.vlipsy_search_triggered = False
            if keywords:
                with st.spinner("Searching Vlipsy..."):
                    try:
                        results = self.apis["vlipsy"].search(
                            remove_quotes(keywords)
                        )
                        self._handle_search_results(
                            "vlipsy", results, config, prefix="vlipsy", search_keyword=keywords)
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                        raise e

        # Affichage des résultats
        self._display_search_results("vlipsy", prefix="vlipsy")

    def run(self, config):
        """Logique principale du plugin"""
        self.config = config
        st.header(t("illustrator_header"))

        # Ajout de la recherche globale
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                global_search_query = st.text_input(
                    "Recherche globale (appuyez sur Entrée ou cliquez le bouton)",
                    key="global_search",
                    help="Recherche sur tous les moteurs simultanément"
                )
            with col2:
                if st.button("Lancer la recherche globale", width='stretch'):
                    st.session_state.global_search_triggered = True

        # Si recherche globale déclenchée
        if (st.session_state.get('global_search_triggered')):
            st.session_state.global_search_triggered = False
            if global_search_query:
                with st.spinner("Lancement des recherches globales..."):
                    # Stocker la requête pour l'ajout aux noms de fichiers
                    st.session_state.last_global_search = global_search_query
                    st.session_state.pexels_keywords = global_search_query
                    st.session_state.google_keywords = global_search_query
                    st.session_state.duckduckgo_keywords = global_search_query
                    st.session_state.vlipsy_keywords = global_search_query
                    st.session_state.youtube_keywords = global_search_query
                    # Lancer les recherches sur tous les onglets
                    try:
                        st.write("Global search runing...")
                        # Pexels
                        if config.get(self.name, {}).get("pexels_api_key"):
                            results = []
                            st.write("Pexel search...")
                            photos = self.apis["pexels"].search(
                                remove_quotes(global_search_query),
                                config.get(self.name, {}).get(
                                    "pexels_api_key"),
                                "photos"
                            )
                            results.extend(photos)
                            videos = self.apis["pexels"].search(
                                remove_quotes(global_search_query),
                                config.get(self.name, {}).get(
                                    "pexels_api_key"),
                                "videos"
                            )
                            results.extend(videos)
                            self._handle_search_results(
                                "pexels", results, config, prefix="pexels", search_keyword=global_search_query)

                        # Google
                        if (config.get('common', {}).get('youtube_api_key') and
                                config.get(self.name, {}).get('google_cx')):
                            st.write("Google search...")
                            results = self.apis["google"].search(
                                remove_quotes(global_search_query),
                                config.get('common', {}).get(
                                    'youtube_api_key'),
                                config.get(self.name, {}).get('google_cx')
                            )
                            self._handle_search_results(
                                "google", results, config, prefix="google", search_keyword=global_search_query)

                        # DuckDuckGo
                        st.write("DuckDuckGo search...")
                        results = self.apis["duckduckgo"].search(
                            remove_quotes(global_search_query)
                        )
                        self._handle_search_results(
                            "duckduckgo", results, config, prefix="duckduckgo", search_keyword=global_search_query)

                        # Vlipsy
                        if config.get(self.name, {}).get("vlipsy_api_key"):
                            st.write("Vlipsy search...")
                            results = self.apis["vlipsy"].search(
                                remove_quotes(global_search_query)
                            )
                            self._handle_search_results(
                                "vlipsy", results, config, prefix="vlipsy", search_keyword=global_search_query)

                        # YouTube
                        st.write("Youtube search...")
                        youtube_api = YoutubeAPI(config)
                        st.session_state.youtube_results = youtube_api.search_assets(
                            global_search_query,
                            creative_commons=True
                        )

                    except Exception as e:
                        st.error(
                            f"Erreur lors de la recherche globale: {str(e)}")

        # Navigation par onglets
        tabs = st.tabs([
            t("illustrator_current_tab"),
            t("illustrator_stored_tab"),
            "Pexels",
            "Google",
            "DuckDuckGo",
            "Vlipsy",
            t("illustrator_youtube_tab")
        ])

        media = None
        with tabs[0]:
            media = self.run_current_assets_tab(config)
        with tabs[1]:
            media = self.run_stored_assets_tab(config)
        with tabs[2]:
            self.run_pexels_tab(config)
        with tabs[3]:
            self.run_google_tab(config)
        with tabs[4]:
            self.run_duckduckgo_tab(config)
        with tabs[5]:
            self.run_vlipsy_tab(config)  # Nouvelle méthode
        with tabs[6]:
            self.run_youtube_assets_tab(config)
        if media:
            self.show_media_preview(media)
