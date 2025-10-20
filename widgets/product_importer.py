from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from lib.products_db import ProductsDB
from lib.youtube_db import get_video_transcript
from widgets.yt_videos import VideoDatabaseWidget
import pandas as pd
import os, re

translations["en"].update({
    "product_importer_title": "Import Products from Videos",
    "product_importer_markdown_title": "Import Products from Markdown Files",
    "import_button": "Import Selected Videos as Products",
    "import_markdown_button": "Import Selected Markdown Files as Products",
    "import_success": "Successfully imported {count} products!",
    "import_error": "Error during import: {error}",
    "select_directory_label": "Select Directory",
    "no_directories_found": "No valid directories found in the specified root path.",
    "no_markdown_files": "No Markdown files found in the selected directory.",
})

translations["fr"].update({
    "product_importer_title": "Importer des produits depuis des vidéos",
    "product_importer_markdown_title": "Importer des produits depuis des fichiers Markdown",
    "import_button": "Importer les vidéos sélectionnées comme produits",
    "import_markdown_button": "Importer les fichiers Markdown sélectionnés comme produits",
    "import_success": "{count} produits importés avec succès !",
    "import_error": "Erreur lors de l'importation : {error}",
    "select_directory_label": "Sélectionner un répertoire",
    "no_directories_found": "Aucun répertoire valide trouvé dans le chemin racine spécifié.",
    "no_markdown_files": "Aucun fichier Markdown trouvé dans le répertoire sélectionné.",
})

class ProductImporterWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.db = ProductsDB()
        self.video_db_widget = VideoDatabaseWidget("videodatabase", f"{prefix}_videodatabase", plugin_manager)

    def import_videodb_display(self):
        st.title(t("product_importer_title"))

        # Afficher la liste des vidéos et récupérer les lignes sélectionnées
        selected_rows = self.video_db_widget.display_video_database()

        if selected_rows is None or selected_rows.empty:
            st.warning("No videos selected for import.")
            return

        # Bouton pour importer les vidéos sélectionnées
        if st.button(t("import_button"), key=f"{self.prefix}_import_button"):
            with st.spinner(t("products_processing")):
                try:
                    import_count = 0
                    for _, video in selected_rows.iterrows():
                        # Récupérer le transcript depuis la base de données
                        transcript = get_video_transcript(video['video_id']) or ""

                        # Préparer les données du produit
                        product_data = {
                            "title": video['title'],
                            "url": video['url'],
                            "keywords": video['keywords'] if video['keywords'] != "--" else "",
                            "type": "youtube",
                            "source": video['video_id'],
                            "goal": "",
                            "related": "",
                            "description": "",
                            "content": transcript
                        }

                        # Ajouter ou mettre à jour le produit dans la base
                        self.db.update_or_add_product(
                            None,  # Aucun ID pour un nouveau produit
                            product_data["title"],
                            product_data["url"],
                            product_data["keywords"],
                            product_data["type"],
                            product_data["source"],
                            product_data["goal"],
                            product_data["related"],
                            product_data["description"],
                            product_data["content"]
                        )
                        import_count += 1

                    st.success(t("import_success").format(count=import_count))
                    st.rerun()
                except Exception as e:
                    st.error(t("import_error").format(error=str(e)))

    def import_markdown_display(self, root_path, excluded_dirs):
        st.title(t("product_importer_markdown_title"))

        # Récupérer la liste des répertoires et sous-répertoires dans root_path, en excluant ceux spécifiés
        excluded_dirs = [d.strip() for d in excluded_dirs if d.strip()]
        directories = []
        try:
            for root, dirs, _ in os.walk(root_path):
                # Exclure les répertoires spécifiés et les répertoires cachés
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in excluded_dirs and not any(os.path.abspath(os.path.join(root, d)).startswith(os.path.abspath(excl)) for excl in excluded_dirs)]
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    # Calculer le chemin relatif depuis root_path
                    rel_path = os.path.relpath(dir_path, root_path)
                    directories.append(rel_path)
        except Exception as e:
            st.error(f"Error accessing directories: {str(e)}")
            return

        if not directories:
            st.error(t("no_directories_found"))
            return

        # Sélection du répertoire
        selected_directory = st.selectbox(
            t("select_directory_label"),
            options=sorted(directories),
            key=f"{self.prefix}_directory_select"
        )

        # Récupérer les fichiers .md dans le répertoire sélectionné
        selected_dir_path = os.path.join(root_path, selected_directory)
        markdown_files = [
            f for f in os.listdir(selected_dir_path)
            if os.path.isfile(os.path.join(selected_dir_path, f)) and f.endswith(".md")
        ]

        if not markdown_files:
            st.warning(t("no_markdown_files"))
            return

        # Créer un DataFrame pour les fichiers Markdown
        df = pd.DataFrame({
            "filename": markdown_files,
            "path": [os.path.join(selected_dir_path, f) for f in markdown_files],
            "content": [""] * len(markdown_files)  # Initialiser avec des chaînes vides
        })

        # Charger le contenu des fichiers
        for idx, row in df.iterrows():
            try:
                with open(row["path"], "r", encoding="utf-8") as f:
                    df.at[idx, "content"] = f.read()
            except Exception as e:
                st.error(f"Error reading file {row['filename']}: {str(e)}")
                df.at[idx, "content"] = ""

        # Ajouter le champ de recherche et les cases à cocher
        search_term = st.text_input("Rechercher (insensible à la casse)")

        # Cases à cocher pour choisir où chercher
        col1, col2 = st.columns(2)
        with col1:
            search_title = st.checkbox("Chercher dans les titres", value=True)
        with col2:
            search_content = st.checkbox("Chercher dans les contenus", value=True)

        # Filtrer le DataFrame en fonction du terme de recherche
        if search_term:
            search_lower = search_term.lower()
            mask = pd.Series([False] * len(df))

            if search_title:
                mask = mask | df["filename"].str.lower().str.contains(search_lower)
            if search_content:
                mask = mask | df["content"].str.lower().str.contains(search_lower)

            filtered_df = df[mask]
        else:
            filtered_df = df.copy()

        # Afficher le nombre de fichiers trouvés
        st.write(f"Fichiers trouvés : {len(filtered_df)} / {len(df)}")

        # Configurer les colonnes pour l'affichage
        column_config = {
            "filename": st.column_config.TextColumn("File Name", width="medium"),
            "path": st.column_config.TextColumn("Full Path", width="large")
        }

        # Afficher le DataFrame avec sélection multi-lignes
        selected_rows = st.dataframe(
            filtered_df[["filename", "path"]],  # Ne montrer que le nom et le chemin
            column_config=column_config,
            width='stretch',
            height=400,
            selection_mode="multi-row",
            on_select="rerun",
            key=f"{self.prefix}_markdown_dataframe"
        )

        # Bouton pour importer les fichiers sélectionnés
        if st.button(t("import_markdown_button"), key=f"{self.prefix}_import_markdown_button"):
            selected_indices = selected_rows.get('selection', {}).get('rows', [])
            if not selected_indices:
                st.warning("No Markdown files selected for import.")
                return

            with st.spinner(t("products_processing")):
                try:
                    import_count = 0
                    for idx in selected_indices:
                        file_info = filtered_df.iloc[idx]
                        file_path = file_info["path"]
                        filename = file_info["filename"]
                        content = file_info["content"]

                        # Préparer les données du produit
                        product_data = {
                            "title": os.path.splitext(filename)[0],  # Nom du fichier sans .md
                            "url": "",
                            "keywords": "",
                            "type": "markdown",
                            "source": file_path,
                            "goal": "",
                            "related": "",
                            "description": "",
                            "content": content
                        }

                        # Ajouter ou mettre à jour le produit dans la base
                        self.db.update_or_add_product(
                            None,  # Aucun ID pour un nouveau produit
                            product_data["title"],
                            product_data["url"],
                            product_data["keywords"],
                            product_data["type"],
                            product_data["source"],
                            product_data["goal"],
                            product_data["related"],
                            product_data["description"],
                            product_data["content"]
                        )
                        import_count += 1

                    st.success(t("import_success").format(count=import_count))
                    st.rerun()
                except Exception as e:
                    st.error(t("import_error").format(error=str(e)))
