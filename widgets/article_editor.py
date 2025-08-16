from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from streamlit_ace import st_ace
from streamlit_lexical import streamlit_lexical
import os
import markdown2

translations["en"].update({
    "editor_title_label": "Post Title",
    "editor_input_label": "Enter your Markdown content",
    "editor_load_generated": "Load Generated Article",
    "editor_include_url": "Include Source URL",
    "editor_image_label": "Feature Image",
    "editor_publish_checkbox": "Publish immediately",
    "editor_include_image": "Include Image in Post",
    "editor_translate_button": "Translate to French",
    "translate_prompt": "Traduisez l'article suivant en français sans ajouter de commentaires, fournissez uniquement le texte traduit :"
})

translations["fr"].update({
    "editor_title_label": "Titre du post",
    "editor_input_label": "Entrez votre contenu Markdown",
    "editor_load_generated": "Charger l'Article Généré",
    "editor_include_url": "Inclure l'URL Source",
    "editor_image_label": "Image de mise en avant",
    "editor_publish_checkbox": "Publier immédiatement",
    "editor_include_image": "Inclure l'image dans le post",
    "editor_translate_button": "Traduire en anglais",
    "translate_prompt": "Translate the following article into English without adding any comments, just provide the translated text:"
})

class ArticleEditorWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.work_dir = self.work_dir()

    def display(self):
        # Default values
        default_title = "New Post"
        default_content = ""
        default_image_path = None
        default_url = ""

        # Load existing article data from disk
        article_path = os.path.join(self.work_dir, "article.md")
        image_path = os.path.join(self.work_dir, "image.png")
        url_path = os.path.join(self.work_dir, "url.txt")
        if os.path.exists(article_path):
            try:
                with open(article_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n", 1)
                    if lines[0].startswith("# "):
                        default_title = lines[0][2:].strip()
                        default_content = lines[1] if len(lines) > 1 else ""
                    else:
                        default_content = content
            except Exception as e:
                st.warning(f"Failed to read article.md: {str(e)}")
                default_content = ""
        if os.path.exists(image_path):
            default_image_path = image_path
        if os.path.exists(url_path):
            try:
                with open(url_path, "r", encoding="utf-8") as f:
                    default_url = f.read().strip()
            except Exception as e:
                st.warning(f"Failed to read url.txt: {str(e)}")
                default_url = ""

        # Button to load generated article
        if os.path.exists(article_path) and st.button(t("editor_load_generated"), key=f"{self.prefix}_load_generated"):
            st.session_state[f"{self.prefix}_title"] = default_title
            st.session_state[f"{self.prefix}_content"] = default_content
            st.session_state[f"{self.prefix}_image_path"] = default_image_path
            st.session_state[f"{self.prefix}_url"] = default_url
        content_value = st.session_state.get(f"{self.prefix}_content", default_content) or ""
        source_url = st.session_state.get(f"{self.prefix}_url", default_url) or ""


        if st.button(t("editor_translate_button"), key=f"{self.prefix}_translate"):
            if content_value:
                prompt = t("translate_prompt")
                translated_content = self.process_with_llm(prompt, content_value)
                st.session_state[f"{self.prefix}_content"] = translated_content
                st.session_state[f"{self.prefix}_title"] = self.process_with_llm(t("translate_prompt"),st.session_state.get(f"{self.prefix}_title", default_title))

        # Input fields
        post_title = st.text_input(
            t("editor_title_label"),
            value=st.session_state.get(f"{self.prefix}_title", default_title),
            key=f"{self.prefix}_title"
        )

        if st.button("URL", key=f"{self.prefix}_url_button"):
            st.session_state[f"{self.prefix}_content"] += f"\n\nSource: {source_url}"

        col1, col2 = st.columns(2)
        markdown_content = col1.text_area(
            label="Markdown Content",
            value=content_value,
            height=300,
            key=f"{self.prefix}_content"
        )
        cont = col2.container(height=300)
        cont.markdown(markdown_content)

        # Image uploader
        st.subheader(t("editor_image_label"))
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            key=f"{self.prefix}_image_upload"
        )
        selected_image_path = st.session_state.get(f"{self.prefix}_image_path", default_image_path)

        if uploaded_image:
            selected_image_path = os.path.join(self.work_dir, f"{self.prefix}_uploaded_image.png")
            with open(selected_image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            st.session_state[f"{self.prefix}_image_path"] = selected_image_path

        if selected_image_path:
            st.image(selected_image_path, caption="Selected Image", width=300)

        # Include image checkbox
        include_image = st.checkbox(
            t("editor_include_image"),
            value=True if selected_image_path else False,
            key=f"{self.prefix}_include_image"
        )

        # Publish immediately checkbox
        publish_immediately = st.checkbox(t("editor_publish_checkbox"), key=f"{self.prefix}_publish_immediately")

        # Convert markdown to HTML for publishing
        html_content = markdown2.markdown(markdown_content) if markdown_content else ""

        return {
            "title": post_title,
            "html_content": html_content,
            "markdown_content": markdown_content,
            "image_path": selected_image_path if include_image else None,
            "publish_immediately": publish_immediately
        }
