# widgets/product_editor.py
from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from streamlit_lexical import streamlit_lexical
from lib.products_db import ProductsDB
import os


class ProductEditorWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.db = ProductsDB()

    def display(self, product=None, button_label="Save"):
        title = st.text_input(t("products_title_label"), value=product['title'] if product is not None else "", key=f"{self.prefix}_title")
        url = st.text_input(t("products_url_label"), value=product['url'] or "" if product is not None else "", key=f"{self.prefix}_url")
        keywords = st.text_input(t("products_keywords_label"), value=product['keywords'] or "" if product is not None else "", key=f"{self.prefix}_keywords")
        type_product = st.text_input(t("products_type_label"), value=product['type'] or "" if product is not None else "", key=f"{self.prefix}_type")
        source = st.text_input(t("products_source_label"), value=product['source'] or "" if product is not None else "", key=f"{self.prefix}_source")
        goal = streamlit_lexical(
            value=product['goal'] or "" if product is not None else "",
            placeholder=t("products_goal_label"),
            height=200,
            key=f"{self.prefix}_goal"
        )
        # Get all products for multiselect
        all_products = self.db.get_all_products()
        product_options = {p['id']: f"{p['id']} - {p['title']}" for p in all_products if product is None or p['id'] != product.get('id')}
        selected_related = []
        if product is not None and isinstance(product['related'], str) and product['related'].strip():
            selected_related = [int(id) for id in product['related'].split(',') if id and id.isdigit()]
        related = st.multiselect(
            t("products_related_label"),
            options=list(product_options.keys()),
            format_func=lambda x: product_options[x],
            default=selected_related,
            key=f"{self.prefix}_related"
        )
        description = streamlit_lexical(
            value=product['description'] or "" if product is not None else "",
            placeholder=t("products_description_label"),
            height=200,
            key=f"{self.prefix}_desc"
        )
        content = streamlit_lexical(
            value=product['content'] or "" if product is not None else "",
            placeholder=t("products_content_label"),
            height=200,
            key=f"{self.prefix}_content"
        )

        return {
            "title": title,
            "url": url,
            "keywords": keywords,
            "type": type_product,
            "source": source,
            "goal": goal,
            "related": ",".join(map(str, related)),
            "description": description,
            "content": content,
            "button": st.button(button_label, key=f"{self.prefix}_save")
        }
