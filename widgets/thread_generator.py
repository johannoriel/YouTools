from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import os

# Traductions spécifiques au widget
translations["en"].update({
    "social_header": "Generate & Post Content",
    "social_transcript": "Transcript",
    "social_url": "Video URL",
    "social_prompt": "LLM Prompt",
    "social_generate": "Generate Posts",
    "social_generating": "Generating posts...",
    "auto_generate_meme_with_posts": "Generate Meme with Posts",
    "meme_preview": "Meme Preview and Edit",
    "auto_selected_meme": "Selected Meme",
    "meme_text0_label": "Top Text",
    "meme_text1_label": "Bottom Text",
    "meme_generating": "Generating meme...",
    "meme_success": "Meme generated successfully at {path}",
    "meme_generation_error": "Failed to generate meme: {error}"
})

translations["fr"].update({
    "social_header": "Générer & Poster du Contenu",
    "social_transcript": "Transcription",
    "social_url": "URL de la vidéo",
    "social_prompt": "Prompt LLM",
    "social_generate": "Générer les posts",
    "social_generating": "Génération des posts...",
    "auto_generate_meme_with_posts": "Générer un mème avec les posts",
    "meme_preview": "Prévisualisation et édition du mème",
    "auto_selected_meme": "Mème sélectionné",
    "meme_text0_label": "Texte du haut",
    "meme_text1_label": "Texte du bas",
    "meme_generating": "Génération du mème...",
    "meme_success": "Mème généré avec succès à {path}",
    "meme_generation_error": "Échec de la génération du mème : {error}"
})

class ThreadGeneratorWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.memegen_plugin = self.plugin_manager.get_plugin('memegen')

    def display(self, config):
        st.header(t("social_header"))
        work_dir = config['common']['work_directory']

        # Recap transcription and video URL
        transcript_path = os.path.join(work_dir, "transcript.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                transcript = f.read()
            st.text_area(t("social_transcript"), transcript, height=100, disabled=True)
        else:
            transcript = st.text_area(t("social_transcript"), height=200)

        url_path = os.path.join(work_dir, "url.txt")
        if os.path.exists(url_path):
            with open(url_path, 'r') as f:
                url = f.read().strip()
            st.text_input(t("social_url"), url, disabled=True)
        else:
            url = st.text_input(t("social_url"))

        # LLM Prompt dans un expander
        with st.expander(t("social_prompt"), expanded=False):
            prompt = st.text_area(t("social_prompt"),
                                value=config['social']['default_prompt'],
                                key=f"{self.prefix}_prompt")
            if st.button("Save Prompt", key=f"{self.prefix}_save_prompt"):
                config = self.plugin_manager.config
                config['social']['default_prompt'] = prompt
                self.plugin_manager.save_config(config)
                st.success("Prompt saved successfully")

        # Checkbox for generating meme with posts
        generate_meme_with_posts = st.checkbox(t("auto_generate_meme_with_posts"),
                                             key=f"{self.prefix}_generate_meme")

        # Use LLM to generate response
        if st.button(t("social_generate"), key=f"{self.prefix}_generate") and transcript:
            with st.spinner(t("social_generating")):
                llm_response = self.process_with_llm(
                    prompt,
                    config.get('llm', {}).get('llm_sys_prompt', ''),
                    transcript
                )

                # Parse posts and save to thread.txt
                posts = []
                for post in llm_response.split('TWEET:')[1:]:
                    clean_post = post.strip().split('---')[0].strip()
                    if clean_post:
                        posts.append(clean_post)

                # Add URL suffix if present
                if url:
                    url_suffix = config['social']['url_suffix_template'].format(url=url)
                    posts.append(url_suffix)

                # Save to thread.txt
                thread_path = os.path.join(work_dir, "thread.txt")
                with open(thread_path, 'w') as f:
                    f.write('\n---\n'.join(posts))

                # Generate meme if checkbox is checked
                if generate_meme_with_posts and self.memegen_plugin:
                    meme_suggestion, error = self.memegen_plugin.generate_meme_from_theme(
                        config, transcript
                    )
                    if meme_suggestion:
                        st.session_state[f"{self.prefix}_auto_meme_suggestion"] = meme_suggestion
                    elif error:
                        st.error(f"Failed to generate meme: {error}")

        # Meme preview
        if f"{self.prefix}_auto_meme_suggestion" in st.session_state and generate_meme_with_posts:
            st.subheader(t("meme_preview"))
            suggestion = st.session_state[f"{self.prefix}_auto_meme_suggestion"]

            meme_titles = [meme["name"] for meme in self.memegen_plugin.memes]
            default_meme_index = meme_titles.index(
                suggestion["meme_name"]) if suggestion["meme_name"] in meme_titles else 0
            selected_meme_name = st.selectbox(
                t("auto_selected_meme"),
                options=meme_titles,
                index=default_meme_index,
                key=f"{self.prefix}_auto_meme_selectbox"
            )

            if selected_meme_name != suggestion["meme_name"]:
                suggestion["meme_name"] = selected_meme_name
                suggestion["template_id"] = next(
                    (m["id"] for m in self.memegen_plugin.memes if m["name"] == selected_meme_name),
                    self.memegen_plugin.memes[0]["id"]
                )

            edited_text0 = st.text_input(
                t("meme_text0_label"), suggestion["text0"], key=f"{self.prefix}_auto_text0")
            edited_text1 = st.text_input(
                t("meme_text1_label"), suggestion["text1"], key=f"{self.prefix}_auto_text1")
            st.image(suggestion["meme_path"], caption="Generated Meme")

            if st.button("Regenerate Meme", key=f"{self.prefix}_regenerate_meme"):
                with st.spinner(t("meme_generating")):
                    meme_path, error = self.memegen_plugin.generate_meme(
                        config, suggestion["template_id"], edited_text0, edited_text1, "impact", 50
                    )
                    if meme_path:
                        st.session_state[f"{self.prefix}_auto_meme_suggestion"]["meme_path"] = meme_path
                        st.image(meme_path, caption="Updated Meme")
                        st.success(t("meme_success").format(path=meme_path))
                    elif error:
                        st.error(t("meme_generation_error").format(error=error))
