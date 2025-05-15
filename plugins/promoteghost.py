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
from widgets.random_article import RandomArticleWidget

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
    "ghost_image_label": "Feature Image",
    "ghost_load_generated": "Load Generated Article"
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
    "ghost_image_label": "Image de mise en avant",
    "ghost_load_generated": "Charger l'Article Généré"
})

class PromoteghostPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.ghost_api = GhostAPI(plugin_manager.config)
        self.work_dir = self.work_dir()

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
                "label": t("randomarticle_prompt"),
                "default": """Génère un article de blog en Markdown basé sur le produit suivant :
            Titre : {title}
            Contenu : {content}
            Mots-clés : {keywords}

            L'article doit être structuré avec une introduction, 3 sections principales et une conclusion. Utilise un ton professionnel et intègre les mots-clés naturellement."""
            }
        }

    def get_tabs(self):
        return [{"name": t("ghost_tab"), "plugin": "ghostplugin"}, {"name": t("randomarticle_tab"), "plugin": "randomarticle"}]

    def run(self, config):
        tab1, tab2 = st.tabs([t("ghost_tab"), t("randomarticle_tab")])
        with tab1:
            st.header(t("ghost_header"))

            # Initialize default values
            default_title = "New Post"
            default_content = ""
            default_image_path = None

            # Check for generated article
            article_path = os.path.join(self.work_dir, "article.md")
            image_path = os.path.join(self.work_dir, "image.png")
            if os.path.exists(article_path):
                with open(article_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Extract title (assuming first line is # Title)
                    lines = content.split("\n", 1)
                    if lines[0].startswith("# "):
                        default_title = lines[0][2:].strip()
                        default_content = lines[1] if len(lines) > 1 else ""
                    else:
                        default_content = content
                if os.path.exists(image_path):
                    default_image_path = image_path

            # Load generated article button
            if os.path.exists(article_path) and st.button(t("ghost_load_generated"), key="load_generated"):
                st.session_state["ghost_title"] = default_title
                st.session_state["ghost_content"] = default_content
                st.session_state["ghost_image_path"] = default_image_path

            # Input fields with session state to persist edits
            post_title = st.text_input(
                t("ghost_title_label"),
                value=st.session_state.get("ghost_title", default_title),
                key="ghost_title"
            )
            markdown_content = st.text_area(
                t("ghost_input_label"),
                value=st.session_state.get("ghost_content", default_content),
                height=300,
                key="ghost_content"
            )

            # Image display and upload
            st.subheader(t("ghost_image_label"))
            uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="ghost_image_upload")
            selected_image_path = st.session_state.get("ghost_image_path", default_image_path)

            if uploaded_image:
                selected_image_path = os.path.join(self.work_dir, "uploaded_image.png")
                with open(selected_image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                st.session_state["ghost_image_path"] = selected_image_path

            if selected_image_path:
                st.image(selected_image_path, caption="Selected Image", use_container_width=True)

            publish_immediately = st.checkbox(t("ghost_publish_checkbox"), key="ghost_publish_immediately")

            if st.button(t("ghost_publish_button"), key="ghost_publish"):
                with burners:
                    try:
                        html_content = markdown2.markdown(markdown_content)
                        response = self.ghost_api.post(
                            post_title,
                            html_content,
                            publish_immediately,
                            feature_image=selected_image_path
                        )
                        if response:
                            post_id = response.get('posts', [{}])[0].get('id', 'N/A')
                            st.success(t("ghost_success").format(result=post_id))
                            # Clear session state after successful publish
                            for key in ["ghost_title", "ghost_content", "ghost_image_path"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                        else:
                            st.error(t("ghost_error").format(error="Unknown error"))
                    except Exception as e:
                        st.error(t("ghost_error").format(error=str(e)))
        with tab2:
            RandomArticleWidget("randomarticle", "randomarticle", self.plugin_manager).display(config)
