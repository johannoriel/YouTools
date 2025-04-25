# widgets/product_editor.py
from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from streamlit_lexical import streamlit_lexical

class ProductEditorWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def display(self, product=None, button_label="Save"):
        title = st.text_input(t("products_title_label"), value=product['title'] if product is not None else "", key=f"{self.prefix}_title")
        url = st.text_input(t("products_url_label"), value=product['url'] or "" if product is not None else "", key=f"{self.prefix}_url")
        keywords = st.text_input(t("products_keywords_label"), value=product['keywords'] or "" if product is not None else "", key=f"{self.prefix}_keywords")
        type_product = st.text_input(t("products_type_label"), value=product['type'] or "" if product is not None else "", key=f"{self.prefix}_type")
        description = streamlit_lexical(
            value=product['description'] or "" if product is not None else "",
            placeholder=t("products_description_label"),
            height=400,
            key=f"{self.prefix}_desc"
        )
        content = st.text_area(t("products_content_label"), value=product['content'] or "" if product is not None else "", height=200, key=f"{self.prefix}_content")

        return {
            "title": title,
            "url": url,
            "keywords": keywords,
            "type": type_product,
            "description": description,
            "content": content,
            "button": st.button(button_label, key=f"{self.prefix}_save")
        }
