from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from typing import Dict, Any
from widgets.image_generator import ImageGeneratorWidget
import os

translations["en"].update({
    "randomarticle_tab": "Random Article",
    "randomarticle_header": "Generate Random Article",
    "randomarticle_generate": "Generate Article",
    "randomarticle_generating": "Generating article...",
    "randomarticle_no_product": "No product selected. Please select a product to continue.",
    "randomarticle_selected_product": "Selected Product",
    "randomarticle_keywords": "Keywords",
    "randomarticle_content": "Content",
    "randomarticle_title": "Article Title",
    "randomarticle_prompt": "LLM Prompt for Random Article"
})

translations["fr"].update({
    "randomarticle_tab": "Article Aléatoire",
    "randomarticle_header": "Générer un Article Aléatoire",
    "randomarticle_generate": "Générer l'Article",
    "randomarticle_generating": "Génération de l'article...",
    "randomarticle_no_product": "Aucun produit sélectionné. Veuillez sélectionner un produit pour continuer.",
    "randomarticle_selected_product": "Produit Sélectionné",
    "randomarticle_keywords": "Mots-clés",
    "randomarticle_content": "Contenu",
    "randomarticle_title": "Titre de l'Article",
    "randomarticle_prompt": "Prompt LLM pour l'Article Aléatoire"
})

class RandomArticleWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.image_generator = ImageGeneratorWidget("image_generator", f"{self.prefix}_image", plugin_manager)

    def generate_article(self, config: Dict[str, Any], product: Dict[str, Any]) -> Dict[str, str]:
        from widgets.prompt_sequence import PromptSequenceWidget
        llm = PromptSequenceWidget("post", "post", self.plugin_manager)
        prompt = config['promoteghost']['randompost_prompt']
        llm_response = llm.prompt_sequence(prompt, product, debug=True)
        if llm_response.startswith('```markdown\n'):
            llm_response = llm_response[len('```markdown\n'):]
        if llm_response.endswith('```\n'):
            llm_response = llm_response[:-len('```\n')]

        # Generate image prompt
        image_prompt_sys = (
            "You are an expert in creating concise prompts for image generation. "
            "Based on the article title and content, generate a vivid, descriptive prompt for an image that visually represents the article. "
            "Focus on style, colors, lighting, and composition. Return only the prompt."
        )
        image_prompt = self.process_with_llm(
            f"Title: {product['title']}\nContent: {llm_response[:500]}",
            sysprompt=image_prompt_sys
        )

        # Generate and save image
        image_path = self.image_generator.generate_image(image_prompt, "16:9")
        work_dir = self.work_dir()
        output_image_path = os.path.join(work_dir, "image.png")
        os.makedirs(work_dir, exist_ok=True)
        with open(output_image_path, "wb") as f:
            with open(image_path, "rb") as img_file:
                f.write(img_file.read())

        # Save article
        output_article_path = os.path.join(work_dir, "article.md")
        with open(output_article_path, "w", encoding="utf-8") as f:
            f.write(f"# {product['title']}\n\n{llm_response}")

        return {
            'title': product['title'],
            'content': llm_response,
            'image_prompt': image_prompt,
            'image_path': output_image_path
        }

    def display(self, config: Dict[str, Any]):
        st.header(t("randomarticle_header"))

        #from widgets.product_selector import ProductSelectorWidget
        #product_selector = ProductSelectorWidget("product_selector", f"{self.prefix}_product_selector", self.plugin_manager)
        from widgets.subject_selector import SubjectSelectorWidget
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

        if st.button(t("randomarticle_generate"), key=f"{self.prefix}_generate"):
            with st.spinner(t("randomarticle_generating")):
                article = self.generate_article(config, selected_product)
                st.success(f"Article and image generated successfully in {self.work_dir()}!")
                st.markdown(f"**{t('randomarticle_title')}**: {article['title']}")
                st.markdown(f"**{t('randomarticle_content')}**:\n{article['content'][:1000]}{'...' if len(article['content']) > 1000 else ''}")
                st.image(article['image_path'], caption="Generated Image", use_container_width=True)
