from global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes, list_all_video_files

import requests
from moviepy.editor import VideoFileClip
import os

# Mise à jour des traductions
translations["en"].update({
    "platform_tab": "Multi-Platform Upload",
    "platform_header": "Upload Video to Multiple Platforms",
    "select_platforms": "Select platforms to upload to",
    "upload_button": "Upload to Selected Platforms",
    "tiktok_api_key": "TikTok API Key",
    "instagram_api_key": "Instagram API Key",
})

translations["fr"].update({
    "platform_tab": "Upload Multi-Plateforme",
    "platform_header": "Uploader une Vidéo sur Plusieurs Plateformes",
    "select_platforms": "Sélectionner les plateformes pour l'upload",
    "upload_button": "Uploader sur les Plateformes Sélectionnées",
    "tiktok_api_key": "Clé API TikTok",
    "instagram_api_key": "Clé API Instagram",
})


class PlatformPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)

    def get_config_fields(self):
        return {
            "tiktok_api_key": {
                "type": "password",
                "label": t("tiktok_api_key"),
                "default": ""
            },
            "instagram_api_key": {
                "type": "password",
                "label": t("instagram_api_key"),
                "default": ""
            },
        }

    def get_tabs(self):
        return [{"name": t("platform_tab"), "plugin": "platform"}]

    def upload_video_tiktok(self, filename, title, description):
        # Vérification de la durée de la vidéo
        clip = VideoFileClip(filename)
        duration = clip.duration
        clip.close()

        if duration > 60:
            st.warning("TikTok video must be 60 seconds or less. Please trim your video.")
            return None

        # Préparation des données pour l'upload
        url = "https://open-api.tiktok.com/share/video/upload/"
        headers = {
            "Authorization": f"Bearer {self.config['platform']['tiktok_api_key']}"
        }
        data = {
            "title": title,
            "description": description,
        }
        files = {
            "video": open(filename, "rb")
        }

        try:
            response = requests.post(url, headers=headers, data=data, files=files)
            response.raise_for_status()
            result = response.json()
            if result.get("data") and result["data"].get("share_id"):
                return result["data"]["share_id"]
            else:
                st.error(f"TikTok upload failed: {result.get('error', 'Unknown error')}")
                return None
        except requests.RequestException as e:
            st.error(f"Error during TikTok upload: {str(e)}")
            return None
        finally:
            files["video"].close()

    def upload_video_instagram(self, filename, title, description):
        # Vérification de la durée de la vidéo
        clip = VideoFileClip(filename)
        duration = clip.duration
        clip.close()

        if duration > 60:
            st.warning("Instagram video must be 60 seconds or less for a single post. Please trim your video or consider using IGTV for longer videos.")
            return None

        # Préparation des données pour l'upload
        url = "https://graph.instagram.com/me/media"
        params = {
            "access_token": self.config['platform']['instagram_api_key'],
            "caption": f"{title}\n\n{description}",
            "media_type": "VIDEO",
        }

        try:
            # Étape 1 : Créer le conteneur de média
            response = requests.post(url, params=params)
            response.raise_for_status()
            result = response.json()
            if not result.get("id"):
                st.error(f"Instagram upload failed: {result.get('error', 'Unknown error')}")
                return None

            creation_id = result["id"]

            # Étape 2 : Upload de la vidéo
            upload_url = f"https://graph.instagram.com/{creation_id}"
            files = {
                "video": open(filename, "rb")
            }
            upload_params = {
                "access_token": self.config['platform']['instagram_api_key'],
            }

            response = requests.post(upload_url, files=files, params=upload_params)
            response.raise_for_status()

            # Étape 3 : Publier le média
            publish_url = f"https://graph.instagram.com/{creation_id}?access_token={self.config['platform']['instagram_api_key']}&fields=id,media_type,media_url,username,timestamp"
            response = requests.get(publish_url)
            response.raise_for_status()
            result = response.json()

            if result.get("id"):
                return result["id"]
            else:
                st.error(f"Instagram publish failed: {result.get('error', 'Unknown error')}")
                return None

        except requests.RequestException as e:
            st.error(f"Error during Instagram upload: {str(e)}")
            return None
        finally:
            if 'files' in locals() and 'video' in files:
                files["video"].close()

    def run(self, config):
        st.header(t("platform_header"))

        work_directory = config['common']['work_directory']
        video_files = list_all_video_files(work_directory)
        if not video_files:
            st.warning(t("transcript_no_videos"))
            return

        selected_video = st.selectbox(
            t("directpublish_select_video"),
            options=[v[0] for v in video_files],
            index=0  # Sélectionne par défaut la vidéo la plus récente
        )
        selected_video_path = next(v[1] for v in video_files if v[0] == selected_video)


        platforms = st.multiselect(
            t("select_platforms"),
            ["YouTube", "TikTok", "Instagram"]
        )

        if st.button(t("upload_button")):
            for platform in platforms:
                if platform == "TikTok":
                    video_id = self.upload_video_tiktok(selected_video_path, title, description)
                    st.success(f"TikTok upload success: {video_id}")
                elif platform == "Instagram":
                    post_id = self.upload_video_instagram(selected_video_path, title, description)
                    st.success(f"Instagram upload success: {post_id}")
