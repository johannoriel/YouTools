from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
from lib.products_db import ProductsDB
import os
from widgets.product_db import ProductGridWidget
import pandas as pd


# Translations
translations["en"].update({
    "products_tab_list": "Products List",
    "products_tab_add": "Add Product",
    "products_tab_import": "Import from Videos",
    "products_tab_import_markdown": "Import from Markdown",
    "products_tab_prompts": "Prompt Configuration",
    "products_header_list": "Product Management",
    "products_header_add": "Add New Product",
    "products_header_import": "Import Products from Videos",
    "products_header_import_markdown": "Import Products from Markdown Files",
    "products_header_prompts": "Configure Prompts",
    "products_title_label": "Product Title",
    "products_url_label": "Product URL",
    "products_keywords_label": "Keywords (comma-separated)",
    "products_type_label": "Product Type",
    "products_source_label": "Product Source",
    "products_goal_label": "Product Goal",
    "products_related_label": "Related Products",
    "products_description_label": "Description (Markdown)",
    "products_content_label": "Full Content",
    "products_add_button": "Add Product",
    "products_update_button": "Update Product",
    "products_delete_button": "Delete Selected",
    "products_generate_keywords_button": "Generate Keywords",
    "products_generate_description_button": "Generate Description",
    "products_processing": "Processing...",
    "products_success": "Operation completed successfully!",
    "products_error": "An error occurred: {error}",
    "products_keywords_prompt_label": "Keywords Generation Prompt",
    "products_description_prompt_label": "Description Generation Prompt",
    "products_save_prompts_button": "Save Prompts",
    "products_keywords_prompt": "Generate a comma-separated list of relevant keywords for a product titled '{title}' with description: {description}",
    "products_description_prompt": "Generate a markdown-formatted description for a product titled '{title}' with keywords: {keywords}",
    "products_db_path_label": "Products Database Path",
    "markdown_root_label": "Markdown Root Directory",
    "excluded_directorys_label": "Excluded Directories (comma-separated)",
    "products_tab_edit_grid": "Edit Products in Grid",
    "products_header_edit_grid": "Edit Products Directly in Grid",
    "products_save_grid_button": "Save Grid Changes",
    "products_type_filter_label": "Filter by Type Keywords",
})

translations["fr"].update({
    "products_tab_list": "Liste des produits",
    "products_tab_add": "Ajouter un produit",
    "products_tab_import": "Importer depuis des vidéos",
    "products_tab_import_markdown": "Importer depuis Markdown",
    "products_tab_prompts": "Configuration des prompts",
    "products_header_list": "Gestion des produits",
    "products_header_add": "Ajouter un nouveau produit",
    "products_header_import": "Importer des produits depuis des vidéos",
    "products_header_import_markdown": "Importer des produits depuis des fichiers Markdown",
    "products_header_prompts": "Configurer les prompts",
    "products_title_label": "Titre du produit",
    "products_url_label": "URL du produit",
    "products_keywords_label": "Mots-clés (séparés par des virgules)",
    "products_type_label": "Type de produit",
    "products_source_label": "Source du produit",
    "products_goal_label": "Objectif du produit",
    "products_related_label": "Produits liés",
    "products_description_label": "Description (Markdown)",
    "products_content_label": "Contenu complet",
    "products_add_button": "Ajouter le produit",
    "products_update_button": "Mettre à jour le produit",
    "products_delete_button": "Supprimer la sélection",
    "products_generate_keywords_button": "Générer des mots-clés",
    "products_generate_description_button": "Générer une description",
    "products_processing": "Traitement en cours...",
    "products_success": "Opération terminée avec succès !",
    "products_error": "Une erreur s'est produite : {error}",
    "products_keywords_prompt_label": "Prompt de génération des mots-clés",
    "products_description_prompt_label": "Prompt de génération de la description",
    "products_save_prompts_button": "Enregistrer les prompts",
    "products_keywords_prompt": "Générer une liste de mots-clés pertinents séparés par des virgules pour un produit intitulé '{title}' avec la description : {description}",
    "products_description_prompt": "Générer une description au format markdown pour un produit intitulé '{title}' avec les mots-clés : {keywords}",
    "products_db_path_label": "Chemin de la base de données des produits",
    "markdown_root_label": "Répertoire racine Markdown",
    "excluded_directorys_label": "Répertoires exclus (séparés par des virgules)",
    "products_tab_edit_grid": "Éditer les produits dans la grille",
    "products_header_edit_grid": "Éditer les produits directement dans la grille",
    "products_save_grid_button": "Enregistrer les modifications de la grille",
    "products_type_filter_label": "Filtrer par mots-clés de type",
})

class ProductsPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.db = ProductsDB()

    def get_config_fields(self):
        return {
            "products_db_path": {
                "type": "text",
                "label": t("products_db_path_label"),
                "default": "products.db"
            },
            "markdown_root": {
                "type": "text",
                "label": t("markdown_root_label"),
                "default": "~/markdown"
            },
            "excluded_directorys": {
                "type": "text",
                "label": t("excluded_directorys_label"),
                "default": ".git, node_modules, venv"
            },
            "keywords_prompt": {
                "type": "textarea",
                "label": t("products_keywords_prompt_label"),
                "default": t("products_keywords_prompt")
            },
            "description_prompt": {
                "type": "textarea",
                "label": t("products_description_prompt_label"),
                "default": t("products_description_prompt")
            }
        }

    def get_tabs(self):
        return [
            {"name": t("products_tab_list"), "plugin": "productsplugin"},
            {"name": t("products_tab_add"), "plugin": "productsplugin"},
            {"name": t("products_tab_import"), "plugin": "productsplugin"},
            {"name": t("products_tab_import_markdown"), "plugin": "productsplugin"},
            {"name": t("products_tab_prompts"), "plugin": "productsplugin"},
            {"name": t("products_tab_edit_grid"), "plugin": "productsplugin"}
        ]

    def _get_type_keywords(self):
        products = self.db.get_all_products()
        type_keywords = set()
        for product in products:
            if product.get('type'):
                keywords = [kw.strip() for kw in product['type'].split(',')]
                type_keywords.update(keywords)
        return sorted(list(type_keywords))

    def _run_edit_grid_tab(self, config):
        from widgets.product_db import ProductGridWidget
        st.header(t("products_header_edit_grid"))

        # Type filter multiselect
        type_keywords = self._get_type_keywords()
        selected_types = st.multiselect(
            t("products_type_filter_label"),
            type_keywords,
            key=f"{self.name}_edit_type_filter"
        )

        grid_widget = ProductGridWidget("productgrid", "edit_grid", self.plugin_manager)
        grid_data = grid_widget.display_editable(selected_types)

        if st.button(t("products_save_grid_button")):
            with st.spinner(t("products_processing")):
                try:
                    for _, row in grid_data.iterrows():
                        self.db.update_or_add_product(
                            int(row['id']) if pd.notnull(row['id']) else None,
                            row['title'],
                            row['url'],
                            row['keywords'],
                            row['type'],
                            row['source'],
                            row['goal'],
                            row['related'],
                            row['description'],
                            row['content']
                        )
                    st.success(t("products_success"))
                    st.rerun()
                except Exception as e:
                    st.error(t("products_error").format(error=str(e)))

    def _run_list_tab(self, config):
        from widgets.product_editor import ProductEditorWidget
        st.header(t("products_header_list"))

        # Type filter multiselect
        type_keywords = self._get_type_keywords()
        selected_types = st.multiselect(
            t("products_type_filter_label"),
            type_keywords,
            key=f"{self.name}_list_type_filter"
        )

        coll, colr = st.columns([2,1])

        with coll:
            grid_widget = ProductGridWidget("productgrid", "grid", self.plugin_manager)
            selected_rows = grid_widget.display(selected_types)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(t("products_delete_button")) and selected_rows is not None and not selected_rows.empty:
                    with st.spinner(t("products_processing")):
                        try:
                            for _, row in selected_rows.iterrows():
                                self.db.delete_product(int(row['id']))
                            st.success(t("products_success"))
                            st.rerun()
                        except Exception as e:
                            st.error(t("products_error").format(error=str(e)))

            with col2:
                if st.button(t("products_generate_keywords_button")) and selected_rows is not None and not selected_rows.empty:
                    with st.spinner(t("products_processing")):
                        try:
                            for _, row in selected_rows.iterrows():
                                prompt = config.get("products", {}).get("keywords_prompt", t("products_keywords_prompt")).format(
                                    title=row['title'],
                                )
                                content_context = f"content: {row['content'] or ''}"
                                description_context = f"description: {row['description'] or ''}"
                                llm_response = self.process_with_llm(
                                    [content_context, description_context, prompt],
                                )
                                if llm_response.startswith("```markdown\n"):
                                    llm_response = llm_response[len("```markdown\n"):-len("\n```")]
                                self.db.update_product_field(
                                    int(row['id']),
                                    'keywords',
                                    llm_response
                                )
                            st.success(t("products_success"))
                            st.rerun()
                        except Exception as e:
                            st.error(t("products_error").format(error=str(e)))

            with col3:
                if st.button(t("products_generate_description_button")) and selected_rows is not None and not selected_rows.empty:
                    with st.spinner(t("products_processing")):
                        try:
                            for _, row in selected_rows.iterrows():
                                prompt = config.get("products", {}).get("description_prompt", t("products_description_prompt")).format(
                                    title=row['title']
                                )
                                keywords_context = f"keywords: {row['keywords'] or ''}"
                                content_context = f"content: {row['content'] or ''}"
                                llm_response = self.process_with_llm(
                                    [keywords_context, content_context, prompt],
                                )
                                if llm_response.startswith("```markdown\n"):
                                    llm_response = llm_response[len("```markdown\n"):-len("\n```")]
                                self.db.update_product_field(
                                    int(row['id']),
                                    'description',
                                    llm_response
                                )
                            st.success(t("products_success"))
                            st.rerun()
                        except Exception as e:
                            st.error(t("products_error").format(error=str(e)))

        if selected_rows is not None and not selected_rows.empty and len(selected_rows) == 1:
            with colr.expander("Edit Product", expanded=True):
                product = selected_rows.iloc[0]
                editor = ProductEditorWidget("producteditor", f"edit_{product['id']}", self.plugin_manager)
                form = editor.display(product, t("products_update_button"))
                self._handle_form_submission(form, product['id'])

    def _run_add_tab(self, config):
        from widgets.product_editor import ProductEditorWidget
        st.header(t("products_header_add"))

        editor = ProductEditorWidget("producteditor", "add", self.plugin_manager)
        form = editor.display(button_label=t("products_add_button"))
        self._handle_form_submission(form)

    def _run_import_tab(self, config):
        from widgets.product_importer import ProductImporterWidget
        st.header(t("products_header_import"))

        importer = ProductImporterWidget("productimporter", "import", self.plugin_manager)
        importer.import_videodb_display()

    def _run_import_markdown_tab(self, config):
        from widgets.product_importer import ProductImporterWidget
        st.header(t("products_header_import_markdown"))

        importer = ProductImporterWidget("productimporter", "import_markdown", self.plugin_manager)
        markdown_root = os.path.expanduser(config.get("products", {}).get("markdown_root", "~/markdown"))
        excluded_dirs = self.plugin_manager.config.get("products", {}).get("excluded_directorys", "").split(",")
        importer.import_markdown_display(markdown_root, excluded_dirs)

    def _run_prompts_tab(self, config):
        st.header(t("products_header_prompts"))

        keywords_prompt = st.text_area(
            t("products_keywords_prompt_label"),
            value=config.get("products", {}).get("keywords_prompt", t("products_keywords_prompt")),
            key=f"{self.name}_keywords_prompt"
        )

        description_prompt = st.text_area(
            t("products_description_prompt_label"),
            value=config.get("products", {}).get("description_prompt", t("products_description_prompt")),
            key=f"{self.name}_description_prompt"
        )

        if st.button(t("products_save_prompts_button")):
            with st.spinner(t("products_processing")):
                try:
                    config.setdefault("products", {})
                    config["products"]["keywords_prompt"] = keywords_prompt
                    config["products"]["description_prompt"] = description_prompt
                    self.plugin_manager.save_config(config)
                    st.success(t("products_success"))
                    st.rerun()
                except Exception as e:
                    st.error(t("products_error").format(error=str(e)))

    def _handle_form_submission(self, form, product_id=None):
        if form["button"]:
            with st.spinner(t("products_processing")):
                try:
                    self.db.update_or_add_product(
                        product_id,
                        form["title"],
                        form["url"],
                        form["keywords"],
                        form["type"],
                        form["source"],
                        form["goal"],
                        form["related"],
                        form["description"],
                        form["content"]
                    )
                    st.success(t("products_success"))
                    st.rerun()
                except Exception as e:
                    st.error(t("products_error").format(error=str(e)))

    def run(self, config):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            t("products_tab_list"),
            t("products_tab_edit_grid"),
            t("products_tab_add"),
            t("products_tab_import"),
            t("products_tab_import_markdown"),
            t("products_tab_prompts"),
        ])
        with tab1:
            self._run_list_tab(config)
        with tab2:
            self._run_edit_grid_tab(config)

        with tab3:
            self._run_add_tab(config)
        with tab4:
            self._run_import_tab(config)
        with tab5:
            self._run_import_markdown_tab(config)
        with tab6:
            self._run_prompts_tab(config)
