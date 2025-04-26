from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from lib.products_db import ProductsDB
from lib.youtube_db import get_video_transcript
from widgets.yt_videos import VideoDatabaseWidget
import pandas as pd

translations["en"].update({
    "product_importer_title": "Import Products from Videos",
    "import_button": "Import Selected Videos as Products",
    "import_success": "Successfully imported {count} products!",
    "import_error": "Error during import: {error}",
})

translations["fr"].update({
    "product_importer_title": "Importer des produits depuis des vidéos",
    "import_button": "Importer les vidéos sélectionnées comme produits",
    "import_success": "{count} produits importés avec succès !",
    "import_error": "Erreur lors de l'importation : {error}",
})

class ProductImporterWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.db = ProductsDB()
        self.video_db_widget = VideoDatabaseWidget("videodatabase", f"{prefix}_videodatabase", plugin_manager)

    def display(self):
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
