from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
import os, re
from datetime import datetime as date
import requests
import jwt
import markdown2
from lib.social_api import GhostAPI, LinkedinAPI
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
    "ghost_load_generated": "Load Generated Article",
    "ghost_include_url": "Include Source URL",
    "linkedin_tab": "LinkedIn Publisher",
    "linkedin_header": "Publish to LinkedIn",
    "linkedin_publish_button": "Publish to LinkedIn",
    "linkedin_processing": "Publishing to LinkedIn...",
    "linkedin_success": "Published successfully! Post ID: {result}",
    "linkedin_error": "Publishing failed: {error}"
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
    "ghost_load_generated": "Charger l'Article Généré",
    "ghost_include_url": "Inclure l'URL Source",
    "linkedin_tab": "Publicateur LinkedIn",
    "linkedin_header": "Publier sur LinkedIn",
    "linkedin_publish_button": "Publier sur LinkedIn",
    "linkedin_processing": "Publication sur LinkedIn...",
    "linkedin_success": "Publié avec succès ! ID du post : {result}",
    "linkedin_error": "Échec de la publication : {error}"
})

class PromoteghostPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.ghost_api = GhostAPI(plugin_manager.config)
        self.linkedin_api = LinkedinAPI(plugin_manager.config)
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
        return [
            {"name": t("ghost_tab"), "plugin": "ghostplugin"},
            {"name": t("randomarticle_tab"), "plugin": "randomarticle"},
            {"name": t("linkedin_tab"), "plugin": "linkedinplugin"}
        ]

    def _markdown_to_text(self, markdown_content: str) -> str:
        """
        Converts markdown content to plain text, stripping formatting and headers.
        :param markdown_content: Markdown string.
        :return: Plain text string.
        """
        # Convert markdown to HTML using markdown2
        html = markdown2.markdown(markdown_content)
        # Strip HTML tags and clean up
        text = re.sub(r'<[^>]+>', '', html)  # Remove HTML tags
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove extra newlines
        text = text.strip()  # Remove leading/trailing whitespace
        return text

    def run(self, config):
        tab1, tab2, tab3 = st.tabs([t("ghost_tab"), t("randomarticle_tab"), t("linkedin_tab")])

        # Ghost Publisher Tab
        with tab1:
            st.header(t("ghost_header"))

            default_title = "New Post"
            default_content = ""
            default_image_path = None
            default_url = ""

            article_path = os.path.join(self.work_dir, "article.md")
            image_path = os.path.join(self.work_dir, "image.png")
            url_path = os.path.join(self.work_dir, "url.txt")
            if os.path.exists(article_path):
                with open(article_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n", 1)
                    if lines[0].startswith("# "):
                        default_title = lines[0][2:].strip()
                        default_content = lines[1] if len(lines) > 1 else ""
                    else:
                        default_content = content
                if os.path.exists(image_path):
                    default_image_path = image_path
                if os.path.exists(url_path):
                    with open(url_path, "r", encoding="utf-8") as f:
                        default_url = f.read().strip()

            if os.path.exists(article_path) and st.button(t("ghost_load_generated"), key="ghost_load_generated"):
                st.session_state["ghost_title"] = default_title
                st.session_state["ghost_content"] = default_content
                st.session_state["ghost_image_path"] = default_image_path
                st.session_state["ghost_url"] = default_url

            post_title = st.text_input(
                t("ghost_title_label"),
                value=st.session_state.get("ghost_title", default_title),
                key="ghost_title"
            )
            include_url = st.checkbox(t("ghost_include_url"), value=False, key="ghost_include_url")
            markdown_content = st.text_area(
                t("ghost_input_label"),
                value=st.session_state.get("ghost_content", default_content) + (f"\n\nSource: {st.session_state.get('ghost_url', default_url)}" if include_url and st.session_state.get('ghost_url', default_url) else ""),
                height=300,
                key="ghost_content"
            )

            st.subheader(t("ghost_image_label"))
            uploaded_image = st.file_uploader("Upload an image (Ghost)", type=["png", "jpg", "jpeg"], key="ghost_image_upload")
            selected_image_path = st.session_state.get("ghost_image_path", default_image_path)

            if uploaded_image:
                selected_image_path = os.path.join(self.work_dir, "ghost_uploaded_image.png")
                with open(selected_image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                st.session_state["ghost_image_path"] = selected_image_path

            if selected_image_path:
                st.image(selected_image_path, caption="Selected Image (Ghost)", use_container_width=True)

            publish_immediately = st.checkbox(t("ghost_publish_checkbox"), key="ghost_publish_immediately")

            if st.button(t("ghost_publish_button"), key="ghost_publish"):
                with st.spinner(t("ghost_processing")):
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
                            for key in ["ghost_title", "ghost_content", "ghost_image_path", "ghost_url"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                        else:
                            st.error(t("ghost_error").format(error="Unknown error"))
                    except Exception as e:
                        st.error(t("ghost_error").format(error=str(e)))

        # Random Article Tab
        with tab2:
            from widgets.random_article import RandomArticleWidget
            RandomArticleWidget("randomarticle", "randomarticle", self.plugin_manager).display(config)

        # LinkedIn Publisher Tab
        with tab3:
            st.header(t("linkedin_header"))

            default_title = "New Post"
            default_content = ""
            default_image_path = None
            default_url = ""

            article_path = os.path.join(self.work_dir, "article.md")
            image_path = os.path.join(self.work_dir, "image.png")
            url_path = os.path.join(self.work_dir, "url.txt")
            if os.path.exists(article_path):
                with open(article_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n", 1)
                    if lines[0].startswith("# "):
                        default_title = lines[0][2:].strip()
                        default_content = lines[1] if len(lines) > 1 else ""
                    else:
                        default_content = content
                if os.path.exists(image_path):
                    default_image_path = image_path
                if os.path.exists(url_path):
                    with open(url_path, "r", encoding="utf-8") as f:
                        default_url = f.read().strip()

            if os.path.exists(article_path) and st.button(t("ghost_load_generated"), key="linkedin_load_generated"):
                st.session_state["linkedin_title"] = default_title
                st.session_state["linkedin_content"] = default_content
                st.session_state["linkedin_image_path"] = default_image_path
                st.session_state["linkedin_url"] = default_url

            post_title = st.text_input(
                t("ghost_title_label"),
                value=st.session_state.get("linkedin_title", default_title),
                key="linkedin_title"
            )
            include_url = st.checkbox(t("ghost_include_url"), value=False, key="linkedin_include_url")
            markdown_content = st.text_area(
                t("ghost_input_label"),
                value=st.session_state.get("linkedin_content", default_content) + (f"\n\nSource: {st.session_state.get('linkedin_url', default_url)}" if include_url and st.session_state.get('linkedin_url', default_url) else ""),
                height=300,
                key="linkedin_content"
            )

            st.subheader(t("ghost_image_label"))
            uploaded_image = st.file_uploader("Upload an image (LinkedIn)", type=["png", "jpg", "jpeg", "gif"], key="linkedin_image_upload")
            selected_image_path = st.session_state.get("linkedin_image_path", default_image_path)

            if uploaded_image:
                selected_image_path = os.path.join(self.work_dir, "linkedin_uploaded_image.png")
                with open(selected_image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                st.session_state["linkedin_image_path"] = selected_image_path

            if selected_image_path:
                st.image(selected_image_path, caption="Selected Image (LinkedIn)", use_container_width=True)

            include_image = st.checkbox("Include Image in Post", value=True if selected_image_path else False, key="linkedin_include_image")
            source_url = st.session_state.get('linkedin_url', default_url)
            st.write(source_url)

            if st.button(t("linkedin_publish_button"), key="linkedin_publish"):
                with st.spinner(t("linkedin_processing")):
                    try:
                        # Convert markdown to plain text
                        plain_content = self._markdown_to_text(markdown_content)

                        feature_image = selected_image_path if include_image and selected_image_path else None
                        response = self.linkedin_api.post_article(
                            post_title,
                            plain_content,
                            source_url=source_url,
                            feature_image=feature_image
                        )
                        if response:
                            post_id = response.get('id', 'N/A')
                            st.success(t("linkedin_success").format(result=post_id))
                            for key in ["linkedin_title", "linkedin_content", "linkedin_image_path", "linkedin_url"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                        else:
                            st.error(t("linkedin_error").format(error="Unknown error"))
                    except Exception as e:
                        st.error(t("linkedin_error").format(error=str(e)))
