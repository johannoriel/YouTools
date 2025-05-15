from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from typing import Dict, Any
from widgets.subject_selector import SubjectSelectorWidget  # Updated to use SubjectSelectorWidget
from widgets.image_generator import ImageGeneratorWidget
import os

translations["en"].update({
    "randomarticle_tab": "Random Article",
    "randomarticle_header": "Generate Random Article",
    "randomarticle_generate": "Generate Article",
    "randomarticle_generating": "Generating article...",
    "randomarticle_no_product": "No subject selected. Please select a subject to continue.",
    "randomarticle_selected_product": "Selected Subject",
    "randomarticle_keywords": "Keywords",
    "randomarticle_content": "Content",
    "randomarticle_title": "Article Title",
    "randomarticle_prompt": "LLM Prompt for Random Article",
    "randomarticle_generate_image": "Generate Image"
})

translations["fr"].update({
    "randomarticle_tab": "Article Aléatoire",
    "randomarticle_header": "Générer un Article Aléatoire",
    "randomarticle_generate": "Générer l'Article",
    "randomarticle_generating": "Génération de l'article...",
    "randomarticle_no_product": "Aucun sujet sélectionné. Veuillez sélectionner un sujet pour continuer.",
    "randomarticle_selected_product": "Sujet Sélectionné",
    "randomarticle_keywords": "Mots-clés",
    "randomarticle_content": "Contenu",
    "randomarticle_title": "Titre de l'Article",
    "randomarticle_prompt": "Prompt LLM pour l'Article Aléatoire",
    "randomarticle_generate_image": "Générer une Image"
})

class RandomArticleWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.image_generator = ImageGeneratorWidget("image_generator", f"{self.prefix}_image", plugin_manager)

    def generate_article(self, config: Dict[str, Any], product: Dict[str, Any], generate_image: bool) -> Dict[str, str]:
        from widgets.prompt_sequence import PromptSequenceWidget
        llm = PromptSequenceWidget("post", "post", self.plugin_manager)
        prompt = config['promoteghost']['randompost_prompt']
        llm_response = llm.prompt_sequence(prompt, product, debug=True)
        if llm_response.startswith('```markdown\n'):
            llm_response = llm_response[len('```markdown\n'):]
        if llm_response.endswith('```\n'):
            llm_response = llm_response[:-len('```\n')]

        result = {
            'title': product['title'],
            'content': llm_response,
            'image_prompt': '',
            'image_path': ''
        }

        # Generate image if enabled
        if generate_image:
            image_prompt_sys = (
                "You are an expert in creating concise prompts for image generation. "
                "Based on the article title and content, generate a vivid, descriptive prompt for an image that visually represents the article. "
                "Focus on style, colors, lighting, and composition. Return only the prompt."
            )
            image_prompt = self.process_with_llm(
                f"Title: {product['title']}\nContent: {llm_response[:500]}",
                sysprompt=image_prompt_sys
            )
            image_path = self.image_generator.generate_image(image_prompt, "16:9")
            work_dir = self.work_dir()
            output_image_path = os.path.join(work_dir, "image.png")
            os.makedirs(work_dir, exist_ok=True)
            with open(output_image_path, "wb") as f:
                with open(image_path, "rb") as img_file:
                    f.write(img_file.read())
            result['image_prompt'] = image_prompt
            result['image_path'] = output_image_path

        # Save article
        work_dir = self.work_dir()
        output_article_path = os.path.join(work_dir, "article.md")
        os.makedirs(work_dir, exist_ok=True)
        with open(output_article_path, "w", encoding="utf-8") as f:
            f.write(f"# {product['title']}\n\n{llm_response}")

        return result

    def display(self, config: Dict[str, Any]):
        st.header(t("randomarticle_header"))

        product_selector = SubjectSelectorWidget("subject_selector", f"{self.prefix}_subject_selector", self.plugin_manager)
        selected_product = product_selector.display()

        if not selected_product:
            st.warning(t("randomarticle_no_product"))
            return

        st.write(f"**{t('randomarticle_selected_product')}**: {selected_product['title']}")
        st.write(f"**{t('randomarticle_keywords')}**: {selected_product['keywords']}")
        st.markdown(f"**{t('randomarticle_content')}**: {selected_product['content'][:1000]}{'...' if len(selected_product['content']) > 1000 else ''}")

        prompt = st.text_area(
            t("randomarticle_prompt"),
            value=config['promoteghost']['randompost_prompt'],
            key=f"{self.prefix}_prompt",
            height=150
        )
        config['promoteghost']['randompost_prompt'] = prompt

        generate_image = st.checkbox(t("randomarticle_generate_image"), value=True, key=f"{self.prefix}_generate_image")

        if st.button(t("randomarticle_generate"), key=f"{self.prefix}_generate"):
            with st.spinner(t("randomarticle_generating")):
                article = self.generate_article(config, selected_product, generate_image)
                st.success(f"Article {'and image ' if generate_image else ''}generated successfully in {self.work_dir()}!")
                st.markdown(f"**{t('randomarticle_title')}**: {article['title']}")
                st.markdown(f"**{t('randomarticle_content')}**:\n{article['content'][:1000]}{'...' if len(article['content']) > 1000 else ''}")
                if generate_image and article['image_path']:
                    st.image(article['image_path'], caption="Generated Image", use_container_width=True)
