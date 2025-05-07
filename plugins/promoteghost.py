from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
import os
from datetime import datetime as date
import requests
import jwt
import markdown2
from lib.social_api import GhostAPI

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
        self.ghost_api = GhostAPI(plugin_manager.config)

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
            },
            "randompost_prompt": {
                        "type": "textarea",
                        "label": t("randompost_prompt"),
                        "default": """Génère un article de blog en Markdown basé sur le produit suivant :
            Titre : {title}
            Contenu : {content}
            Mots-clés : {keywords}

            L'article doit être structuré avec une introduction, 3 sections principales et une conclusion. Utilise un ton professionnel et intègre les mots-clés naturellement."""
                    }
        }

    def get_tabs(self):
        return [{"name": t("ghost_tab"), "plugin": "ghostplugin"}]

    def run(self, config):
        tab1, tab2 = st.tabs([t("ghost_tab"), t("randompost_tab")])
        with tab1:
            st.header(t("ghost_header"))
            post_title = st.text_input(t("ghost_title_label"), value="New Post")
            markdown_content = st.text_area(t("ghost_input_label"), height=300)
            publish_immediately = st.checkbox(t("ghost_publish_checkbox"))

            if st.button(t("ghost_publish_button")):
                with st.spinner(t("ghost_processing")):
                    try:
                        html_content = markdown2.markdown(markdown_content)
                        response = self.ghost_api.post(post_title, html_content, publish_immediately)
                        if response:
                            post_id = response.get('posts', [{}])[0].get('id', 'N/A')
                            st.success(t("ghost_success").format(result=post_id))
                        else:
                            st.error(t("ghost_error").format(error="Unknown error"))
                    except Exception as e:
                        st.error(t("ghost_error").format(error=str(e)))
        with tab2:
            from widgets.random_ghost import RandomGhostPostWidget
            RandomGhostPostWidget("randompost", "randompost", self.plugin_manager).display(config)
