from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import os
import markdown2
import uuid

translations["en"].update({
    "editor_title_label": "Post Title",
    "editor_input_label": "Enter your Markdown content",
    "editor_load_generated": "Load Generated Article",
    "editor_load_draft": "Load Selected Draft",
    "editor_include_url": "Include Source URL",
    "editor_image_label": "Feature Image",
    "editor_publish_checkbox": "Publish immediately",
    "editor_include_image": "Include Image in Post",
    "editor_translate_button": "Translate to French",
    "editor_append_translation": "Prepend English Translation",
    "translate_prompt": "Traduisez l'article suivant en français sans ajouter de commentaires, fournissez uniquement le texte traduit :",
    "editor_save_button": "Save Article",
    "editor_save_success": "Article saved successfully!",
    "editor_draft_list": "Drafts",
    "editor_refresh": "Refresh",
    "editor_extract_title": "Extract Title"
})

translations["fr"].update({
    "editor_title_label": "Titre du post",
    "editor_input_label": "Entrez votre contenu Markdown",
    "editor_load_generated": "Charger l'Article Généré",
    "editor_load_draft": "Charger le Brouillon Sélectionné",
    "editor_include_url": "Inclure l'URL Source",
    "editor_image_label": "Image de mise en avant",
    "editor_publish_checkbox": "Publier immédiatement",
    "editor_include_image": "Inclure l'image dans le post",
    "editor_translate_button": "Traduire en anglais",
    "editor_append_translation": "Ajouter la traduction en anglais",
    "translate_prompt": "Translate the following article into English without adding any comments, just provide the translated text:",
    "editor_save_button": "Sauvegarder l'Article",
    "editor_save_success": "Article sauvegardé avec succès !",
    "editor_draft_list": "Brouillons",
    "editor_refresh": "Rafraîchir",
    "editor_extract_title": "Extraire le Titre"
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

        # Load existing article data from disk only for explicit loading
        article_path = os.path.join(self.work_dir, "article.md")
        image_path = os.path.join(self.work_dir, "illustration.jpg")
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

        # Initialize session state only if not already set
        if f"{self.prefix}_content" not in st.session_state:
            st.session_state[f"{self.prefix}_content"] = ""
        if f"{self.prefix}_title" not in st.session_state:
            st.session_state[f"{self.prefix}_title"] = default_title
        if f"{self.prefix}_url" not in st.session_state:
            st.session_state[f"{self.prefix}_url"] = default_url
        if f"{self.prefix}_image_path" not in st.session_state:
            st.session_state[f"{self.prefix}_image_path"] = default_image_path

        # Draft list, load draft button, and refresh button on same line
        col_draft, col_load_draft, col_refresh = st.columns([3, 1, 1])
        with col_draft:
            draft_files = [f for f in os.listdir(self.work_dir) if f.endswith('.md')]
            selected_draft = st.selectbox(t("editor_draft_list"), draft_files, key=f"{self.prefix}_draft_list")
        with col_load_draft:
            if st.button(t("editor_load_draft"), key=f"{self.prefix}_load_draft") and selected_draft:
                try:
                    with open(os.path.join(self.work_dir, selected_draft), "r", encoding="utf-8") as f:
                        content = f.read()
                        lines = content.split("\n", 1)
                        if lines[0].startswith("# "):
                            st.session_state[f"{self.prefix}_title"] = lines[0][2:].strip()
                            st.session_state[f"{self.prefix}_content"] = lines[1] if len(lines) > 1 else ""
                        else:
                            st.session_state[f"{self.prefix}_content"] = content
                except Exception as e:
                    st.warning(f"Failed to load draft: {str(e)}")
        with col_refresh:
            if st.button(t("editor_refresh"), key=f"{self.prefix}_refresh"):
                st.rerun()

        # Button to load generated article
        if os.path.exists(article_path) and st.button(t("editor_load_generated"), key=f"{self.prefix}_load_generated"):
            st.session_state[f"{self.prefix}_title"] = default_title
            st.session_state[f"{self.prefix}_content"] = default_content
            st.session_state[f"{self.prefix}_image_path"] = default_image_path
            st.session_state[f"{self.prefix}_url"] = default_url

        # Translation buttons, extract title button, and URL button on same line
        col_trans1, col_trans2, col_extract, col_url = st.columns(4)
        with col_trans1:
            if st.button(t("editor_translate_button"), key=f"{self.prefix}_translate"):
                if st.session_state[f"{self.prefix}_content"]:
                    prompt = t("translate_prompt")
                    translated_content = self.process_with_llm(prompt, st.session_state[f"{self.prefix}_content"])
                    st.session_state[f"{self.prefix}_content"] = translated_content
                    st.session_state[f"{self.prefix}_title"] = self.process_with_llm(t("translate_prompt"), st.session_state[f"{self.prefix}_title"])
        with col_trans2:
            if st.button(t("editor_append_translation"), key=f"{self.prefix}_append_translation"):
                if st.session_state[f"{self.prefix}_content"]:
                    prompt = t("translate_prompt")
                    translated_content = self.process_with_llm(prompt, st.session_state[f"{self.prefix}_content"])
                    st.session_state[f"{self.prefix}_content"] = f"*(Article en Français ci-dessous)*\n{translated_content}\n\n---\n{st.session_state[f'{self.prefix}_content']}"
                    st.session_state[f"{self.prefix}_title"] = self.process_with_llm(t("translate_prompt"), st.session_state[f"{self.prefix}_title"])
        with col_extract:
            if st.button(t("editor_extract_title"), key=f"{self.prefix}_extract_title"):
                if st.session_state[f"{self.prefix}_content"]:
                    lines = st.session_state[f"{self.prefix}_content"].split("\n", 1)
                    if lines[0].startswith("# "):
                        st.session_state[f"{self.prefix}_title"] = lines[0][2:].strip()
                        st.session_state[f"{self.prefix}_content"] = lines[1] if len(lines) > 1 else ""
        with col_url:
            if st.button("URL", key=f"{self.prefix}_url_button"):
                st.session_state[f"{self.prefix}_content"] += f"\n\nSource: {st.session_state[f'{self.prefix}_url']}"

        # Input fields
        post_title = st.text_input(
            t("editor_title_label"),
            key=f"{self.prefix}_title"
        )

        # Inject Overtype for Markdown text area
        # TODO : BUGGY
        overtype_script = """
        <script src="https://unpkg.com/overtype"></script>
        <style>
            .editor {
                width: 100%;
                height: 300px;
                resize: vertical;
            }
        </style>
        <script>
        """
        overtype_script += f"""            const textarea = document.querySelector('div[class="st-key-{self.prefix}_content"] textarea');"""
        overtype_script += """
            console.log("Markdown editor handling");
            if (textarea) {
                textarea.classList.add('editor');
                new OverType('.editor');
            } else {
                console.log("Markdown editor not found");
            }
        </script>
        """
        st.markdown(overtype_script, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        markdown_content = col1.text_area(
            label="Markdown Content",
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

        # Save button, publish checkbox, and include image checkbox on same line
        col_save, col_publish, col_include = st.columns(3)
        with col_save:
            if st.button(t("editor_save_button"), key=f"{self.prefix}_save"):
                try:
                    with open(article_path, "w", encoding="utf-8") as f:
                        f.write(f"# {post_title}\n{markdown_content}")
                    with open(url_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state[f"{self.prefix}_url"])
                    st.success(t("editor_save_success"))
                except Exception as e:
                    st.error(f"Failed to save article: {str(e)}")
        with col_publish:
            publish_immediately = st.checkbox(t("editor_publish_checkbox"), key=f"{self.prefix}_publish_immediately")
        with col_include:
            include_image = st.checkbox(
                t("editor_include_image"),
                value=True if selected_image_path else False,
                key=f"{self.prefix}_include_image"
            )

        # Convert markdown to HTML for publishing
        html_content = markdown2.markdown(markdown_content) if markdown_content else ""

        return {
            "title": post_title,
            "html_content": html_content,
            "markdown_content": markdown_content,
            "url": default_url,
            "image_path": selected_image_path if include_image else None,
            "publish_immediately": publish_immediately
        }
