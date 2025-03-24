from global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
import requests
import os
import shutil

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "pexels_tab": "Pexels Search",
    "pexels_header": "Search Images and Videos on Pexels",
    "pexels_keywords_label": "Enter keywords (comma-separated)",
    "pexels_per_page_label": "Results per page (max 80)",
    "pexels_language_label": "Select languages (optional, multiple)",
    "pexels_size_label": "Select size",
    "pexels_orientation_label": "Select orientation",
    "pexels_search_button": "Search Pexels",
    "pexels_searching": "Searching Pexels...",
    "pexels_images_title": "Images",
    "pexels_videos_title": "Videos",
    "pexels_success": "Search completed! Found {count} items.",
    "pexels_error": "An error occurred: {error}",
    "pexels_download_button": "Download",
    "pexels_download_success": "File downloaded to {path}",
    "pexels_config_api_key": "Pexels API Key",
    "pexels_config_directory": "Download Directory",
    "pexels_config_default_dir": "./downloads",
    "pexels_prev_page": "Previous Page",
    "pexels_next_page": "Next Page",
})

translations["fr"].update({
    "pexels_tab": "Recherche Pexels",
    "pexels_header": "Rechercher des images et vidéos sur Pexels",
    "pexels_keywords_label": "Entrez des mots-clés (séparés par des virgules)",
    "pexels_per_page_label": "Résultats par page (max 80)",
    "pexels_language_label": "Sélectionnez les langues (optionnel, multiple)",
    "pexels_size_label": "Sélectionnez la taille",
    "pexels_orientation_label": "Sélectionnez l'orientation",
    "pexels_search_button": "Rechercher sur Pexels",
    "pexels_searching": "Recherche sur Pexels en cours...",
    "pexels_images_title": "Images",
    "pexels_videos_title": "Vidéos",
    "pexels_success": "Recherche terminée ! {count} éléments trouvés.",
    "pexels_error": "Une erreur s'est produite : {error}",
    "pexels_download_button": "Télécharger",
    "pexels_download_success": "Fichier téléchargé dans {path}",
    "pexels_config_api_key": "Clé API Pexels",
    "pexels_config_directory": "Répertoire de téléchargement",
    "pexels_config_default_dir": "./téléchargements",
    "pexels_prev_page": "Page précédente",
    "pexels_next_page": "Page suivante",
})


class PexelsPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.languages = [
            'en-US', 'pt-BR', 'es-ES', 'ca-ES', 'de-DE', 'it-IT', 'fr-FR', 'sv-SE',
            'id-ID', 'pl-PL', 'ja-JP', 'zh-TW', 'zh-CN', 'ko-KR', 'th-TH', 'nl-NL',
            'hu-HU', 'vi-VN', 'cs-CZ', 'da-DK', 'fi-FI', 'uk-UA', 'el-GR', 'ro-RO',
            'nb-NO', 'sk-SK', 'tr-TR', 'ru-RU'
        ]
        self.sizes = ["large", "medium", "small"]
        self.orientations = ["landscape", "portrait", "square"]

    def get_config_fields(self):
        """Définit les champs de configuration du plugin."""
        return {
            "pexels_api_key": {
                "type": "text",
                "label": t("pexels_config_api_key"),
                "default": ""
            },
            "pexels_download_directory": {
                "type": "text",
                "label": t("pexels_config_directory"),
                "default": t("pexels_config_default_dir")
            }
        }

    def get_tabs(self):
        """Définit les onglets du plugin dans l'interface."""
        return [{"name": t("pexels_tab"), "plugin": "pexelsplugin"}]

    def search_pexels(self, keywords, api_key, media_type="photos", page=1, per_page=15, size=None, orientation=None, locales=None):
        """Effectue une recherche sur l'API Pexels."""
        url = "https://api.pexels.com/v1/search" if media_type == "photos" else "https://api.pexels.com/videos/search"
        headers = {"Authorization": api_key}
        params = {
            "query": keywords,
            "page": page,
            "per_page": per_page,
        }
        if size:
            params["size"] = size
        if orientation:
            params["orientation"] = orientation
        if locales:
            # Pexels ne semble accepter qu'une seule locale à la fois pour l'instant
            params["locale"] = locales[0]
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"API Error: {response.status_code} - {response.text}")

    def get_media_by_id(self, media_id, api_key, media_type="photo"):
        """Récupère un média spécifique par son ID."""
        url = f"https://api.pexels.com/v1/photos/{media_id}" if media_type == "photo" else f"https://api.pexels.com/videos/videos/{media_id}"
        headers = {"Authorization": api_key}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"API Error: {response.status_code} - {response.text}")

    def download_media(self, url, filename, directory):
        """Télécharge un fichier média depuis une URL avec répertoire expansé."""
        # Expansion du ~ en chemin absolu
        expanded_directory = os.path.expanduser(directory)
        if not os.path.exists(expanded_directory):
            os.makedirs(expanded_directory)
        filepath = os.path.join(expanded_directory, filename)
        response = requests.get(url, stream=True)
        response.raw.decode_content = True  # Gérer le décodage du contenu
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            return filepath
        else:
            raise Exception(f"Download failed: {response.status_code}")

    def run(self, config):
        """Logique principale du plugin."""
        st.header(t("pexels_header"))

        # Récupération de la configuration
        api_key = config.get(self.name, {}).get("pexels_api_key", "")
        download_dir = config.get(self.name, {}).get(
            "pexels_download_directory", t("pexels_config_default_dir"))

        if not api_key:
            st.error(
                "Pexels API key is not configured. Please set it in the configuration.")
            return

        # Options de recherche
        keywords = st.text_input(t("pexels_keywords_label"), value="nature")
        per_page = st.number_input(
            t("pexels_per_page_label"), min_value=1, max_value=80, value=15)
        selected_locales = st.multiselect(
            t("pexels_language_label"), self.languages)
        size = st.selectbox(t("pexels_size_label"), [
                            ""] + self.sizes, format_func=lambda x: "Any" if x == "" else x.capitalize())
        orientation = st.selectbox(t("pexels_orientation_label"), [
                                   ""] + self.orientations, format_func=lambda x: "Any" if x == "" else x.capitalize())

        # Gestion de la pagination dans la session state
        if "pexels_page" not in st.session_state:
            st.session_state.pexels_page = 1

        # Bouton de recherche
        if st.button(t("pexels_search_button")):
            with st.spinner(t("pexels_searching")):
                try:
                    cleaned_keywords = remove_quotes(keywords)
                    # Recherche des images
                    images_data = self.search_pexels(
                        cleaned_keywords, api_key, "photos", st.session_state.pexels_page, per_page,
                        size if size else None, orientation if orientation else None, selected_locales if selected_locales else None
                    )
                    # Recherche des vidéos
                    videos_data = self.search_pexels(
                        cleaned_keywords, api_key, "videos", st.session_state.pexels_page, per_page,
                        size if size else None, orientation if orientation else None, selected_locales if selected_locales else None
                    )

                    st.session_state.images = images_data.get("photos", [])
                    st.session_state.videos = videos_data.get("videos", [])
                    st.session_state.total_images = images_data.get(
                        "total_results", 0)
                    st.session_state.total_videos = videos_data.get(
                        "total_results", 0)
                    st.session_state.next_page_images = images_data.get(
                        "next_page", None)
                    st.session_state.prev_page_images = images_data.get(
                        "prev_page", None)
                    st.session_state.next_page_videos = videos_data.get(
                        "next_page", None)
                    st.session_state.prev_page_videos = videos_data.get(
                        "prev_page", None)

                    st.success(t("pexels_success").format(
                        count=len(st.session_state.images) + len(st.session_state.videos)))

                except Exception as e:
                    st.error(t("pexels_error").format(error=str(e)))

        # Affichage des résultats si disponibles
        if "images" in st.session_state and "videos" in st.session_state:
            col1, col2 = st.columns(2)

            # Colonne des images
            with col1:
                st.subheader(t("pexels_images_title"))
                for img in st.session_state.images:
                    st.image(img["src"]["medium"],
                             caption=f"Photo by {img['photographer']}")
                    if st.button(t("pexels_download_button"), key=f"dl_img_{img['id']}"):
                        try:
                            media_data = self.get_media_by_id(
                                img["id"], api_key, "photo")
                            url = media_data["src"]["original"]
                            filename = f"{img['id']}.jpg"
                            filepath = self.download_media(
                                url, filename, download_dir)
                            st.success(
                                t("pexels_download_success").format(path=filepath))
                        except Exception as e:
                            st.error(t("pexels_error").format(error=str(e)))

            # Colonne des vidéos
            with col2:
                st.subheader(t("pexels_videos_title"))
                for vid in st.session_state.videos:
                    st.image(vid["image"],
                             caption=f"Video by {vid['user']['name']}")
                    if st.button(t("pexels_download_button"), key=f"dl_vid_{vid['id']}"):
                        try:
                            media_data = self.get_media_by_id(
                                vid["id"], api_key, "video")
                            # Premier fichier disponible
                            url = media_data["video_files"][0]["link"]
                            filename = f"{vid['id']}.mp4"
                            filepath = self.download_media(
                                url, filename, download_dir)
                            st.success(
                                t("pexels_download_success").format(path=filepath))
                        except Exception as e:
                            st.error(t("pexels_error").format(error=str(e)))

            # Pagination
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.session_state.prev_page_images or st.session_state.prev_page_videos:
                    if st.button(t("pexels_prev_page")):
                        st.session_state.pexels_page -= 1
                        st.rerun()
            with col_next:
                if st.session_state.next_page_images or st.session_state.next_page_videos:
                    if st.button(t("pexels_next_page")):
                        st.session_state.pexels_page += 1
                        st.rerun()


if __name__ == "__main__":
    st.write("Pexels Plugin standalone test")
