from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from typing import Dict, Any
from lib.social_api import GhostAPI
from widgets.product_selector import ProductSelectorWidget
from widgets.image_generator import ImageGeneratorWidget
import markdown2

translations["en"].update({
    "randompost_tab": "Random Post",
    "randompost_header": "Generate Random Ghost Post",
    "randompost_generate": "Generate Post",
    "randompost_generating": "Generating post...",
    "randompost_preview": "Preview Post",
    "randompost_publish": "Publish to Ghost",
    "randompost_no_product": "No product selected. Please select a product to continue.",
    "randompost_selected_product": "Selected Product",
    "randompost_keywords": "Keywords",
    "randompost_content": "Content",
    "randompost_title": "Post Title",
    "randompost_prompt": "LLM Prompt for Random Post"
})

translations["fr"].update({
    "randompost_tab": "Article Aléatoire",
    "randompost_header": "Générer un Article Ghost Aléatoire",
    "randompost_generate": "Générer l'Article",
    "randompost_generating": "Génération de l'article...",
    "randompost_preview": "Prévisualiser l'Article",
    "randompost_publish": "Publier sur Ghost",
    "randompost_no_product": "Aucun produit sélectionné. Veuillez sélectionner un produit pour continuer.",
    "randompost_selected_product": "Produit Sélectionné",
    "randompost_keywords": "Mots-clés",
    "randompost_content": "Contenu",
    "randompost_title": "Titre de l'Article",
    "randompost_prompt": "Prompt LLM pour l'Article Aléatoire"
})

class RandomGhostPostWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.ghost_api = GhostAPI(plugin_manager.config)
        self.image_generator = ImageGeneratorWidget("image_generator", f"{self.prefix}_image", plugin_manager)
        self._initialize_session_state()

    def _initialize_session_state(self):
        if f'{self.prefix}_generated_post' not in st.session_state:
            st.session_state[f'{self.prefix}_generated_post'] = {'title': '', 'content': '', 'image_prompt': '', 'image_path': ''}
        if f'{self.prefix}_publish_immediately' not in st.session_state:
            st.session_state[f'{self.prefix}_publish_immediately'] = False

    def generate_post(self, config: Dict[str, Any], product: Dict[str, Any]) -> Dict[str, str]:
        from widgets.prompt_sequence import PromptSequenceWidget
        llm = PromptSequenceWidget("post", "post", self.plugin_manager)
        prompt = config['promoteghost']['randompost_prompt']
        llm_response = llm.prompt_sequence(prompt, product, debug=True)
        if llm_response.startswith('```markdown\n'):
            llm_response = llm_response[len('```markdown\n'):]
        if llm_response.endswith('```\n'):
            llm_response = llm_response[:-len('```\n')]

        # Generate image pre-prompt
        image_prompt_sys = (
            "You are an expert in creating concise prompts for image generation. "
            "Based on the post title and content, generate a vivid, descriptive prompt for an image that visually represents the post. "
            "Focus on style, colors, lighting, and composition. Return only the prompt."
        )
        image_prompt = self.process_with_llm(
            f"Title: {product['title']}\nContent: {llm_response[:500]}",
            sysprompt=image_prompt_sys
        )

        # Generate image
        image_path = self.image_generator.generate_image(image_prompt, "16:9")

        return {
            'title': product['title'],
            'content': llm_response,
            'image_prompt': image_prompt,
            'image_path': image_path
        }

    def display(self, config: Dict[str, Any]):
        st.header(t("randompost_header"))

        product_selector = ProductSelectorWidget("product_selector", f"{self.prefix}_product_selector", self.plugin_manager)
        selected_product = product_selector.display()

        if not selected_product:
            st.warning(t("randompost_no_product"))
            return

        st.write(f"**{t('randompost_selected_product')}**: {selected_product['title']}")
        st.write(f"**{t('randompost_keywords')}**: {selected_product['keywords']}")
        st.markdown(f"**{t('randompost_content')}**: {selected_product['content'][:1000]}{'...' if len(selected_product['content']) > 1000 else ''}")

        prompt = st.text_area(
            t("randompost_prompt"),
            value=config['promoteghost']['randompost_prompt'],
            key=f"{self.prefix}_prompt",
            height=150
        )
        config['promoteghost']['randompost_prompt'] = prompt

        if st.button(t("randompost_generate"), key=f"{self.prefix}_generate"):
            with st.spinner(t("randompost_generating")):
                post = self.generate_post(config, selected_product)
                st.session_state[f'{self.prefix}_generated_post'] = post

        if st.session_state[f'{self.prefix}_generated_post']['content']:
            st.subheader(t("randompost_preview"))
            edited_title = st.text_input(
                t("randompost_title"),
                value=st.session_state[f'{self.prefix}_generated_post']['title'],
                key=f"{self.prefix}_title"
            )
            edited_content = st.text_area(
                t("randompost_content"),
                value=st.session_state[f'{self.prefix}_generated_post']['content'],
                key=f"{self.prefix}_content",
                height=300
            )
            st.subheader("Image Preview")
            edited_image_prompt = st.text_area(
                "Image Prompt",
                value=st.session_state[f'{self.prefix}_generated_post']['image_prompt'],
                key=f"{self.prefix}_image_prompt",
                height=100
            )
            if st.session_state[f'{self.prefix}_generated_post']['image_path']:
                st.image(st.session_state[f'{self.prefix}_generated_post']['image_path'], caption="Generated Image", use_container_width=True)

            if st.button("Regenerate Image", key=f"{self.prefix}_regenerate_image"):
                with st.spinner("Regenerating image..."):
                    new_image_path = self.image_generator.generate_image(edited_image_prompt, "16:9")
                    st.session_state[f'{self.prefix}_generated_post']['image_path'] = new_image_path
                    st.session_state[f'{self.prefix}_generated_post']['image_prompt'] = edited_image_prompt

            st.session_state[f'{self.prefix}_generated_post'].update({
                'title': edited_title,
                'content': edited_content,
                'image_prompt': edited_image_prompt
            })

            publish_immediately = st.checkbox(
                t("ghost_publish_checkbox"),
                value=st.session_state.get(f'{self.prefix}_publish_immediately', False),
                key=f"{self.prefix}_publish_immediately"
            )

            if st.button(t("randompost_publish"), key=f"{self.prefix}_publish"):
                with st.spinner(t("ghost_processing")):
                    try:
                        html_content = markdown2.markdown(edited_content)
                        image_path = st.session_state[f'{self.prefix}_generated_post']['image_path']
                        response = self.ghost_api.post(
                            edited_title,
                            html_content,
                            publish_immediately,
                            feature_image=image_path
                        )
                        if response:
                            post_id = response.get('posts', [{}])[0].get('id', 'N/A')
                            st.success(t("ghost_success").format(result=post_id))
                        else:
                            st.error(t("ghost_error").format(error="Unknown error"))
                    except Exception as e:
                        st.error(t("ghost_error").format(error=str(e)))
