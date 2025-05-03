from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
import os
from datetime import datetime as date
import requests
import jwt
import markdown2

translations["en"].update({
    "ghost_tab": "Ghost Publisher",
    "ghost_header": "Publish to Ghost",
    "ghost_title_label": "Post Title",
    "ghost_input_label": "Enter your Markdown content",
    "ghost_publish_button": "Publish to Ghost",
    "ghost_publish_checkbox": "Publish immediately",
    "ghost_processing": "Publishing to Ghost...",
    "ghost_success": "Published successfully! Post ID: {result}",
    "ghost_error": "Publishing failed: {error}",
    "ghost_config_api_key": "Ghost API Key",
    "ghost_config_url": "Ghost Admin URL",
    "ghost_config_api_key_default": "your_api_key",
    "ghost_config_url_default": "https://your-site.com/ghost/api/admin/",
})

translations["fr"].update({
    "ghost_tab": "Publicateur Ghost",
    "ghost_header": "Publier sur Ghost",
    "ghost_title_label": "Titre du post",
    "ghost_input_label": "Entrez votre contenu Markdown",
    "ghost_publish_button": "Publier sur Ghost",
    "ghost_publish_checkbox": "Publier immédiatement",
    "ghost_processing": "Publication sur Ghost...",
    "ghost_success": "Publié avec succès ! ID du post : {result}",
    "ghost_error": "Échec de la publication : {error}",
    "ghost_config_api_key": "Clé API Ghost",
    "ghost_config_url": "URL Admin Ghost",
    "ghost_config_api_key_default": "votre_clé_api",
    "ghost_config_url_default": "https://votre-site.com/ghost/api/admin/",
})

class PromoteghostPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)

    def get_config_fields(self):
        return {
            "ghost_api_key": {
                "type": "text",
                "label": t("ghost_config_api_key"),
                "default": t("ghost_config_api_key_default")
            },
            "ghost_url": {
                "type": "text",
                "label": t("ghost_config_url"),
                "default": t("ghost_config_url_default")
            }
        }

    def get_tabs(self):
        return [{"name": t("ghost_tab"), "plugin": "ghostplugin"}]

    def run(self, config):
        st.header(t("ghost_header"))
        post_title = st.text_input(t("ghost_title_label"), value="New Post")
        markdown_content = st.text_area(t("ghost_input_label"), height=300)
        publish_immediately = st.checkbox(t("ghost_publish_checkbox"))

        if st.button(t("ghost_publish_button")):
            with st.spinner(t("ghost_processing")):
                try:
                    api_key = config.get(self.name, {}).get("ghost_api_key", "")
                    url = config.get(self.name, {}).get("ghost_url", "")
                    if not api_key or not url:
                        raise ValueError("API key or URL not configured")

                    id, secret = api_key.split(':')
                    iat = int(date.now().timestamp())
                    header = {'alg': 'HS256', 'typ': 'JWT', 'kid': id}
                    payload = {
                        'iat': iat,
                        'exp': iat + 5 * 60,
                        'aud': '/admin/'
                    }
                    token = jwt.encode(payload, bytes.fromhex(secret), algorithm='HS256', headers=header)

                    headers = {'Authorization': f'Ghost {token}'}
                    html_content = markdown2.markdown(markdown_content)
                    body = {
                        'posts': [{
                            'title': post_title,
                            'html': html_content,
                            'status': 'published' if publish_immediately else 'draft',
                        }]
                    }
                    response = requests.post(f"{url}posts/", json=body, headers=headers)
                    response.raise_for_status()

                    post_id = response.json().get('posts', [{}])[0].get('id', 'N/A')
                    st.success(t("ghost_success").format(result=post_id))

                except Exception as e:
                    st.error(t("ghost_error").format(error=str(e)))
