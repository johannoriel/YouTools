from global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
import os
import shutil
from media_selector import media_selector, remote_media_selector
from assets_api import PexelsAPI, CanvaAPI
from io import BytesIO

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
    "both": "Both",
    "All": "All",
    "Images": "Images",
    "Videos": "Videos",
    "Audio": "Audio",
    "Filter by type": "Filter by type",
    "download_to_stored": "Download to Stored Assets",
    "download_to_current": "Download to Current Assets",
    "download_to_both": "Download to Both",
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
    "both": "Les deux",
    "All": "Tous",
    "Images": "Images",
    "Videos": "Vidéos",
    "Audio": "Audio",
    "Filter by type": "Filtrer par type",
    "download_to_stored": "Télécharger vers Assets Stockés",
    "download_to_current": "Télécharger vers Assets Actuels",
    "download_to_both": "Télécharger vers les Deux",
})


class IllustratorPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.apis = {
            "pexels": PexelsAPI(),
            "canva": CanvaAPI()
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
            "canva_api_key": {
                "type": "text",
                "label": "Canva API Key",
                "default": ""
            }
        }

    def get_tabs(self):
        """Définition des onglets"""
        return [
            {"name": t("illustrator_current_tab"),
             "plugin": "illustratorplugin", "tab": "current"},
            {"name": t("illustrator_stored_tab"),
             "plugin": "illustratorplugin", "tab": "stored"},
            {"name": t("illustrator_search_tab"),
             "plugin": "illustratorplugin", "tab": "search"}
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
                    st.video(media_data, format="video/mp4", autoplay=True)

            # Cas des chemins de fichiers locaux
            elif isinstance(media_data, str):
                if media_data.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    st.image(media_data)
                elif media_data.lower().endswith(('.mp4', '.mov', '.avi')):
                    st.video(media_data, format="video/mp4", autoplay=True)
                elif media_data.lower().endswith(('.mp3', '.wav')):
                    st.audio(media_data)

            # Cas des résultats de recherche (dictionnaire)
            elif isinstance(media_data, dict):
                if media_data['original_data']['type'] == 'photo':
                    st.image(media_data['url'])
                else:
                    st.video(media_data['url'], format="video/mp4", autoplay=True)

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
            "All": ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov', '.avi', '.mp3', '.wav'],
            "Images": ['.jpg', '.jpeg', '.png', '.gif'],
            "Videos": ['.mp4', '.mov', '.avi'],
            "Audio": ['.mp3', '.wav']
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

        # Prévisualisation
        if selected:
            self.show_media_preview(selected)

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
            "All": ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov', '.avi', '.mp3', '.wav'],
            "Images": ['.jpg', '.jpeg', '.png', '.gif'],
            "Videos": ['.mp4', '.mov', '.avi'],
            "Audio": ['.mp3', '.wav']
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

            # Prévisualisation
            if selected_media:
                self.show_media_preview(selected_media)

    def run_search_assets_tab(self, config):
        """Onglet de recherche de nouveaux assets"""
        st.header(t("illustrator_search_tab"))
        stored_dir = self.expand_path(config.get(self.name, {}).get(
            "illustrator_stored_dir", t("illustrator_config_default_stored")))
        current_dir = self.expand_path(config.get(self.name, {}).get(
            "illustrator_current_dir", t("illustrator_config_default_current")))

        # Initialisation des variables de session
        if 'search_results' not in st.session_state:
            st.session_state.search_results = None
        if 'selected_item' not in st.session_state:
            st.session_state.selected_item = None
        if 'media_buffer' not in st.session_state:
            st.session_state.media_buffer = None
        if 'media_type' not in st.session_state:
            st.session_state.media_type = None

        # Configuration des API
        api_keys = {
            "pexels": config.get(self.name, {}).get("pexels_api_key", ""),
            "canva": config.get(self.name, {}).get("canva_api_key", "")
        }

        # Sélection de l'API et type de média
        col1, col2 = st.columns(2)
        with col1:
            selected_api = st.selectbox(
                t("illustrator_search_api"), list(self.apis.keys()))
        with col2:
            media_type = st.selectbox(
                t("illustrator_media_type"),
                ["photos", "videos", "both"],
                format_func=lambda x: t(f"illustrator_{x}")
            )

        if not api_keys[selected_api]:
            st.error(f"API key for {selected_api} is not configured")
            return

        # Recherche
        keywords = st.text_input(t("illustrator_search_keywords"))
        if st.button(t("illustrator_search_button")) and keywords:
            with st.spinner("Searching..."):
                try:
                    results = []
                    if media_type in ["photos", "both"]:
                        photos = self.apis[selected_api].search(
                            remove_quotes(keywords),
                            api_keys[selected_api],
                            "photos"
                        )
                        results.extend(photos)
                    if media_type in ["videos", "both"]:
                        videos = self.apis[selected_api].search(
                            remove_quotes(keywords),
                            api_keys[selected_api],
                            "videos"
                        )
                        results.extend(videos)

                    formatted_results = []
                    for item in results:
                        formatted_results.append({
                            'url': item['url'],
                            'name': item.get('name', f"Media {item['id']}"),
                            'date': item.get('date', 0),
                            'original_data': item
                        })
                    st.session_state.search_results = formatted_results
                    # Réinitialiser la sélection quand on fait une nouvelle recherche
                    st.session_state.selected_item = None
                    st.session_state.media_buffer = None
                    st.session_state.media_type = None
                except Exception as e:
                    st.error(f"Search error: {str(e)}")

        # Affichage des résultats
        if st.session_state.search_results:
            # Sélection du dossier de destination pour stored assets
            subdirs = self.get_subdirectories(stored_dir)
            selected_subdir = st.selectbox(
                t("illustrator_destination_folder"),
                subdirs + ["[Create New Folder]"]
            )

            if selected_subdir == "[Create New Folder]":
                new_folder = st.text_input(t("illustrator_create_folder"))
                if new_folder and st.button("Create"):
                    new_path = os.path.join(stored_dir, new_folder)
                    os.makedirs(new_path, exist_ok=True)
                    st.success(f"Folder created: {new_path}")
                    st.rerun()
                return

            # Sélection du média
            new_selection = remote_media_selector(
                st.session_state.search_results, "search")

            # Si la sélection a changé, réinitialiser le buffer
            if new_selection != st.session_state.selected_item:
                st.session_state.selected_item = new_selection
                st.session_state.media_buffer = None
                st.session_state.media_type = None

            # Téléchargement pour prévisualisation
            if st.session_state.selected_item and not st.session_state.media_buffer:
                with st.spinner("Downloading for preview..."):
                    try:
                        buffer, media_type = self.apis[selected_api].memory_download(
                            st.session_state.selected_item['original_data']
                        )
                        st.session_state.media_buffer = buffer
                        st.session_state.media_type = media_type
                    except Exception as e:
                        st.error(f"Preview download error: {str(e)}")

            # Prévisualisation
            if st.session_state.media_buffer:
                self.show_media_preview(
                    st.session_state.media_buffer,
                    st.session_state.media_type
                )

            # Boutons de téléchargement
            if st.session_state.selected_item and selected_subdir and st.session_state.media_buffer:
                media_type = st.session_state.selected_item['original_data']['type']
                ext = '.mp4' if media_type == 'video' else '.jpg'

                st.markdown("---")
                st.subheader("Save Options")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(t("download_to_stored")):
                        try:
                            filename = f"{st.session_state.selected_item['name']}{ext}"
                            filepath = os.path.join(stored_dir, selected_subdir, filename)

                            # On réécrit le buffer dans le fichier
                            with open(filepath, 'wb') as f:
                                f.write(st.session_state.media_buffer.getvalue())
                            st.success(f"Saved to stored assets: {filepath}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                with col2:
                    if st.button(t("download_to_current")):
                        try:
                            current_dir = self.expand_path(self.config.get(self.name, {}).get(
                                "illustrator_current_dir", t("illustrator_config_default_current")))
                            os.makedirs(current_dir, exist_ok=True)

                            filename = f"{st.session_state.selected_item['name']}{ext}"
                            filepath = os.path.join(current_dir, filename)

                            with open(filepath, 'wb') as f:
                                f.write(st.session_state.media_buffer.getvalue())
                            st.success(f"Added to current assets: {filepath}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                with col3:
                    if st.button(t("download_to_both")):
                        try:
                            # Save to stored
                            filename = f"{st.session_state.selected_item['name']}{ext}"
                            stored_path = os.path.join(stored_dir, selected_subdir, filename)
                            with open(stored_path, 'wb') as f:
                                f.write(st.session_state.media_buffer.getvalue())

                            # Save to current
                            current_dir = self.expand_path(self.config.get(self.name, {}).get(
                                "illustrator_current_dir", t("illustrator_config_default_current")))
                            os.makedirs(current_dir, exist_ok=True)
                            current_path = os.path.join(current_dir, filename)
                            with open(current_path, 'wb') as f:
                                f.write(st.session_state.media_buffer.getvalue())

                            st.success(
                                f"Saved to stored assets: {stored_path}\n"
                                f"Added to current assets: {current_path}"
                            )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

    def run(self, config):
        """Logique principale du plugin"""
        self.config = config
        st.header(t("illustrator_header"))

        # Récupération de l'onglet actif
        active_tab = st.session_state.get("illustrator_active_tab", "current")

        # Navigation par onglets
        tabs = st.tabs([
            t("illustrator_current_tab"),
            t("illustrator_stored_tab"),
            t("illustrator_search_tab")
        ])

        with tabs[0]:
            self.run_current_assets_tab(config)
        with tabs[1]:
            self.run_stored_assets_tab(config)
        with tabs[2]:
            self.run_search_assets_tab(config)


if __name__ == "__main__":
    st.write("Illustrator Plugin standalone test")
