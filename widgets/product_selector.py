from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from typing import List, Dict, Any, Optional
from lib.products_db import ProductsDB
import random

# Ajout des traductions spécifiques au widget
translations["en"].update({
    "productselector_select_types": "Select Product Types",
    "productselector_select_all_types": "Select All Types",
    "productselector_select_product": "Select a Product",
    "productselector_random_product": "Choose Random Product",
    "productselector_no_products": "No products found for selected types.",
})

translations["fr"].update({
    "productselector_select_types": "Sélectionner les types de produits",
    "productselector_select_all_types": "Sélectionner tous les types",
    "productselector_select_product": "Sélectionner un produit",
    "productselector_random_product": "Choisir un produit aléatoire",
    "productselector_no_products": "Aucun produit trouvé pour les types sélectionnés.",
})

class ProductSelectorWidget(Widget):
    def __init__(self, name: str, prefix: str, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self._initialize_session_state()

    def _initialize_session_state(self):
        if f'{self.prefix}_selected_types' not in st.session_state:
            st.session_state[f'{self.prefix}_selected_types'] = []
        if f'{self.prefix}_selected_product' not in st.session_state:
            st.session_state[f'{self.prefix}_selected_product'] = None

    def _parse_product_types(self, products: List[Dict[str, Any]]) -> List[str]:
        """Extract unique product types from the 'type' field, splitting by commas or spaces."""
        all_types = set()
        for product in products:
            if product['type']:
                # Split by commas or spaces, remove empty strings and strip whitespace
                types = [t.strip() for t in product['type'].replace(',', ' ').split() if t.strip()]
                all_types.update(types)
        return sorted(list(all_types))

    def _filter_products_by_types(self, products: List[Dict[str, Any]], selected_types: List[str]) -> List[Dict[str, Any]]:
        """Filter products based on selected types."""
        if not selected_types:
            return products
        filtered = []
        for product in products:
            if product['type']:
                product_types = [t.strip() for t in product['type'].replace(',', ' ').split() if t.strip()]
                if any(t in selected_types for t in product_types):
                    filtered.append(product)
        return filtered

    def display(self) -> Optional[Dict[str, Any]]:
        """Display the product selector and return the selected product."""

        with st.container(key=f"{self.prefix}_container", border=True):

            # Initialize ProductsDB
            products_db = ProductsDB()
            products = products_db.get_all_products()

            if not products:
                st.warning(t("productselector_no_products"))
                return None

            # Get all unique product types
            all_types = self._parse_product_types(products)

            if not all_types:
                st.warning("No product types found in the database.")
                return None

            # Multiselect for product types
            if not st.session_state[f'{self.prefix}_selected_types']:
                st.session_state[f'{self.prefix}_selected_types'] = all_types

            selected_types = st.multiselect(
                t("productselector_select_types"),
                options=all_types,
                default=st.session_state[f'{self.prefix}_selected_types'],
                key=f"{self.prefix}_types_select"
            )
            st.session_state[f'{self.prefix}_selected_types'] = selected_types

            # Button to select all types
            if st.button(t("productselector_select_all_types"), key=f"{self.prefix}_select_all_types"):
                st.session_state[f'{self.prefix}_selected_types'] = all_types
                st.rerun()

            # Filter products based on selected types
            filtered_products = self._filter_products_by_types(products, selected_types)

            if not filtered_products:
                st.warning(t("productselector_no_products"))
                return None

            # Initialize selected product if not set
            if not st.session_state[f'{self.prefix}_selected_product'] or \
            st.session_state[f'{self.prefix}_selected_product'] not in filtered_products:
                st.session_state[f'{self.prefix}_selected_product'] = random.choice(filtered_products)

            # Create a list of product titles for the selectbox
            product_titles = [product['title'] for product in filtered_products]
            selected_product_title = st.session_state[f'{self.prefix}_selected_product']['title']
            default_index = product_titles.index(selected_product_title) if selected_product_title in product_titles else 0

            # Selectbox to choose a product
            selected_title = st.selectbox(
                t("productselector_select_product"),
                options=product_titles,
                index=default_index,
                key=f"{self.prefix}_product_select"
            )

            # Button to choose a new random product
            if st.button(t("productselector_random_product"), key=f"{self.prefix}_random_product"):
                st.session_state[f'{self.prefix}_selected_product'] = random.choice(filtered_products)
                st.rerun()

            # Update selected product based on user choice
            selected_product = next((p for p in filtered_products if p['title'] == selected_title), filtered_products[0])
            if selected_product['title'] != st.session_state[f'{self.prefix}_selected_product']['title']:
                st.session_state[f'{self.prefix}_selected_product'] = selected_product

            return selected_product
