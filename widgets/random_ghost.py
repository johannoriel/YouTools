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
        self._initialize_session_state()

    def _initialize_session_state(self):
        if f'{self.prefix}_generated_post' not in st.session_state:
            st.session_state[f'{self.prefix}_generated_post'] = {'title': '', 'content': ''}
        if f'{self.prefix}_publish_immediately' not in st.session_state:
            st.session_state[f'{self.prefix}_publish_immediately'] = False

    def generate_post(self, config: Dict[str, Any], product: Dict[str, Any]) -> Dict[str, str]:
        from widgets.prompt_sequence import PromptSequenceWidget
        llm = PromptSequenceWidget("post", "post", self.plugin_manager)
        prompt = config['promoteghost']['randompost_prompt'].format(
            title=product['title'],
            content=product['content'],
            keywords=product['keywords']
        )
        llm_response = llm.prompt_sequence(prompt, product, debug=True)
        if llm_response.startswith('```markdown\n'):
            llm_response = llm_response[len('```markdown\n'):]
        if llm_response.endswith('```\n'):
            llm_response = llm_response[:-len('```\n')]
        return {
            'title': product['title'],
            'content': llm_response
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
            st.session_state[f'{self.prefix}_generated_post'] = {
                'title': edited_title,
                'content': edited_content
            }

            # Utiliser la valeur du checkbox directement sans modifier session_state après instantiation
            publish_immediately = st.checkbox(
                t("ghost_publish_checkbox"),
                value=st.session_state.get(f'{self.prefix}_publish_immediately', False),
                key=f"{self.prefix}_publish_immediately"
            )

            if st.button(t("randompost_publish"), key=f"{self.prefix}_publish"):
                with st.spinner(t("ghost_processing")):
                    try:
                        html_content = markdown2.markdown(edited_content)
                        response = self.ghost_api.post(edited_title, html_content, publish_immediately)
                        if response:
                            post_id = response.get('posts', [{}])[0].get('id', 'N/A')
                            st.success(t("ghost_success").format(result=post_id))
                        else:
                            st.error(t("ghost_error").format(error="Unknown error"))
                    except Exception as e:
                        st.error(t("ghost_error").format(error=str(e)))
