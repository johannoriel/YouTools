# File: video_product_match.py
from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
import os
from lib.products_db import ProductsDB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import unicodedata
import re

translations["en"].update({
    "extract_keywords": "Extract key topics and keywords as a comma-separated list (max 10)",
    "match_title": "Video-Product Relevance Scores",
    "no_videos_error": "No valid video data found",
    "no_products_error": "No products found in database",
    "progress_text": "Calculating relevance scores...",
    "video_details": "Video Details",
    "product_details": "Product Details",
    "calculate_button": "Calculate Relevance Scores",
    "similarity_method": "Select Similarity Method",
    "tfidf_cosine": "TF-IDF Cosine Similarity",
    "sentence_transformer": "Sentence Transformer",
    "fuzzywuzzy": "FuzzyWuzzy",
    "manual_count": "Manual Keyword Count",
    "llm_score": "LLM Similarity Score",
    "comparison_type": "Select Comparison Type",
    "keywords_only": "Keywords Only",
    "full_text": "Full Text",
    "video_title_column": "Video Title",
    "video_keywords": "Video Keywords",
    "product_keywords": "Product Keywords",
    "llm_prompt": "Evaluate the semantic similarity between the following video content and product content. Provide a score between 0 (no similarity) and 1 (perfect similarity). Video: {video_content} Product: {product_content}",
})

translations["fr"].update({
    "extract_keywords": "Extraire des mots-clés et des sujets clés sous forme de liste séparée par des virgules (max 10)",
    "match_title": "Scores de pertinence vidéo-produit",
    "no_videos_error": "Aucune donnée vidéo valide trouvée",
    "no_products_error": "Aucun produit trouvé dans la base de données",
    "progress_text": "Calcul des scores de pertinence...",
    "video_details": "Détails de la vidéo",
    "product_details": "Détails du produit",
    "calculate_button": "Calculer les scores de pertinence",
    "similarity_method": "Sélectionner la méthode de similarité",
    "tfidf_cosine": "Similarité Cosinus TF-IDF",
    "sentence_transformer": "Transformeur de Phrases",
    "fuzzywuzzy": "FuzzyWuzzy",
    "manual_count": "Comptage Manuel des Mots-clés",
    "llm_score": "Score de Similarité LLM",
    "comparison_type": "Sélectionner le type de comparaison",
    "keywords_only": "Mots-clés uniquement",
    "full_text": "Texte complet",
    "video_title_column": "Titre de la vidéo",
    "video_keywords": "Mots-clés de la vidéo",
    "product_keywords": "Mots-clés du produit",
    "llm_prompt": "Évaluez la similarité sémantique entre le contenu vidéo suivant et le contenu du produit. Fournissez un score entre 0 (aucune similarité) et 1 (similarité parfaite). Vidéo : {video_content} Produit : {product_content}",
})

class VideoProductMatchWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.db = ProductsDB()
        self.work_directory = self.plugin_manager.config["common"]["work_directory"]
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def normalize_keyword(self, keyword):
        keyword = keyword.lower()
        keyword = ''.join(c for c in unicodedata.normalize('NFD', keyword)
                         if unicodedata.category(c) != 'Mn')
        keyword = re.sub(r'[_-]+|[^\w\s]', ' ', keyword)
        keyword = ' '.join(keyword.split())
        return keyword

    def load_videos(self):
        csv_files = [f for f in os.listdir(self.work_directory) if f.startswith('video_list') and f.endswith('.csv')]
        if not csv_files:
            return None

        required_columns = {'keyword', 'url', 'video_id', 'title', 'description'}
        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(self.work_directory, csv_file)
            try:
                df = pd.read_csv(file_path)
                if required_columns.issubset(set(df.columns)):
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return None

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df.drop_duplicates(subset='video_id', keep='first')

    def extract_video_keywords(self, video_row):
        keywords = str(video_row['keyword']).split(',') if pd.notnull(video_row['keyword']) else []

        prompts = [
            f"Title: {video_row['title']}",
            f"Description: {video_row['description'] if pd.notnull(video_row['description']) else ''}",
            t("extract_keywords")
        ]
        llm_response = self.process_with_llm(prompts)
        llm_keywords = llm_response.split(',') if llm_response else []

        all_keywords = list(set(keywords + llm_keywords))
        normalized_keywords = [self.normalize_keyword(kw) for kw in all_keywords if kw.strip()]
        return normalized_keywords, ' '.join(normalized_keywords)

    def get_comparison_text(self, video_row, product, comparison_type):
        if comparison_type == "keywords_only":
            _, video_text = self.extract_video_keywords(video_row)
            product_keywords = str(product['keywords']).split(',') if product['keywords'] else []
            product_text = ' '.join([self.normalize_keyword(kw) for kw in product_keywords])
        else:  # full_text
            video_text = f"{video_row['title']} {video_row['description'] if pd.notnull(video_row['description']) else ''}".lower()
            product_text = f"{product['title']} {product['description'] if product['description'] else ''}".lower()
        return video_text, product_text

    def calculate_tfidf_cosine(self, video_texts, product_texts):
        vectorizer = TfidfVectorizer()
        all_texts = video_texts + product_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        video_vectors = tfidf_matrix[:len(video_texts)]
        product_vectors = tfidf_matrix[len(video_texts):]
        return cosine_similarity(video_vectors, product_vectors)

    def calculate_sentence_transformer(self, video_texts, product_texts):
        video_embeddings = self.sentence_model.encode(video_texts, convert_to_tensor=True)
        product_embeddings = self.sentence_model.encode(product_texts, convert_to_tensor=True)
        return util.cos_sim(video_embeddings, product_embeddings).cpu().numpy()

    def calculate_fuzzywuzzy(self, video_texts, product_texts):
        scores = np.zeros((len(video_texts), len(product_texts)))
        for i, v_text in enumerate(video_texts):
            for j, p_text in enumerate(product_texts):
                scores[i, j] = fuzz.token_sort_ratio(v_text, p_text) / 100.0
        return scores

    def calculate_manual_count(self, video_keywords_list, product_keywords_list):
        scores = np.zeros((len(video_keywords_list), len(product_keywords_list)))
        for i, video_keywords in enumerate(video_keywords_list):
            for j, product_keywords in enumerate(product_keywords_list):
                common_keywords = len(set(video_keywords) & set(product_keywords))
                scores[i, j] = common_keywords
        max_score = scores.max() if scores.max() > 0 else 1
        scores = scores / max_score
        return scores

    def calculate_llm_score(self, video_texts, product_texts):
        scores = np.zeros((len(video_texts), len(product_texts)))
        for i, video_text in enumerate(video_texts):
            for j, product_text in enumerate(product_texts):
                prompt = [
                    t("llm_prompt").format(video_content=video_text, product_content=product_text),
                    "Return only a number between 0 and 1."
                ]
                response = self.process_with_llm(prompt)
                try:
                    score = float(response)
                    scores[i, j] = max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
                except (ValueError, TypeError):
                    scores[i, j] = 0.0
        return scores

    def calculate_relevance_scores(self, videos_df, products, similarity_method, comparison_type):
        video_texts = []
        video_ids = []
        video_keywords_list = []
        progress_bar = st.progress(0)
        total_steps = len(videos_df)

        for idx, row in videos_df.iterrows():
            keywords_list, keywords_text = self.extract_video_keywords(row)
            if keywords_text:
                video_texts.append(keywords_text if comparison_type == "keywords_only" else f"{row['title']} {row['description'] if pd.notnull(row['description']) else ''}".lower())
                video_ids.append(row['video_id'])
                video_keywords_list.append(keywords_list)
            progress_bar.progress((idx + 1) / total_steps)

        product_texts = []
        product_ids = []
        product_keywords_list = []
        for product in products:
            text = str(product['keywords']).lower() if comparison_type == "keywords_only" and product['keywords'] else f"{product['title']} {product['description'] if product['description'] else ''}".lower()
            product_texts.append(text)
            product_ids.append(str(product['id']))
            product_keywords = str(product['keywords']).split(',') if product['keywords'] else []
            product_keywords_list.append([self.normalize_keyword(kw.strip()) for kw in product_keywords])

        if not video_texts or not product_texts:
            return None, None, None, None

        if similarity_method == "tfidf_cosine":
            similarity_matrix = self.calculate_tfidf_cosine(video_texts, product_texts)
        elif similarity_method == "sentence_transformer":
            similarity_matrix = self.calculate_sentence_transformer(video_texts, product_texts)
        elif similarity_method == "fuzzywuzzy":
            similarity_matrix = self.calculate_fuzzywuzzy(video_texts, product_texts)
        elif similarity_method == "manual_count":
            similarity_matrix = self.calculate_manual_count(video_keywords_list, product_keywords_list)
        else:  # llm_score
            similarity_matrix = self.calculate_llm_score(video_texts, product_texts)

        scores_df = pd.DataFrame(
            similarity_matrix,
            index=video_ids,
            columns=product_ids
        )
        progress_bar.empty()
        return scores_df, videos_df, products, video_keywords_list

    def display(self):
        st.title(t("match_title"))

        videos_df = self.load_videos()
        if videos_df is None:
            st.error(t("no_videos_error"))
            return

        products = self.db.get_all_products()
        if not products:
            st.error(t("no_products_error"))
            return

        # Similarity method selection
        similarity_method = st.selectbox(
            t("similarity_method"),
            ["tfidf_cosine", "sentence_transformer", "fuzzywuzzy", "manual_count", "llm_score"],
            format_func=lambda x: t(x),
            key=f"{self.prefix}_similarity_method"
        )

        # Comparison type selection
        comparison_type = st.selectbox(
            t("comparison_type"),
            ["keywords_only", "full_text"],
            format_func=lambda x: t(x),
            key=f"{self.prefix}_comparison_type"
        )

        if st.button(t("calculate_button"), key=f"{self.prefix}_calculate_button"):
            st.write(t("progress_text"))
            scores_df, videos_df, products, video_keywords_list = self.calculate_relevance_scores(videos_df, products, similarity_method, comparison_type)
            if scores_df is None:
                st.error(t("no_videos_error"))
                return
            # Add video titles to scores_df
            video_titles = videos_df.set_index('video_id')['title'].to_dict()
            scores_df.insert(0, 'video_title', [video_titles.get(vid, '') for vid in scores_df.index])
            st.session_state[f"{self.prefix}_scores_data"] = (scores_df, videos_df, products, video_keywords_list)

        # Check if scores data exists in session state
        if f"{self.prefix}_scores_data" in st.session_state:
            scores_df, videos_df, products, video_keywords_list = st.session_state[f"{self.prefix}_scores_data"]

            # Configure cell style for gradient background (black to red)
            cellsytle_jscode = JsCode("""
            function(params) {
                if (params.value != null) {
                    var value = parseFloat(params.value);
                    var red = Math.round(255 * value);
                    var green = 0;
                    var blue = 0;
                    return {
                        'color': 'white',
                        'backgroundColor': 'rgb(' + red + ',' + green + ',' + blue + ')'
                    }
                }
                return {
                    'color': 'white',
                    'backgroundColor': 'black'
                }
            }
            """)

            # Configure AgGrid
            gb = GridOptionsBuilder.from_dataframe(scores_df)
            gb.configure_default_column(editable=False)
            gb.configure_column('video_title', headerName=t("video_title_column"), width=300, pinned='left')
            for product in products:
                col_id = str(product['id'])
                gb.configure_column(
                    col_id,
                    headerName=col_id,
                    width=100,
                    type=["numericColumn"],
                    valueFormatter="Number(x).toFixed(3)",
                    headerTooltip=product['title'],
                    cellStyle=cellsytle_jscode
                )
            gb.configure_selection(selection_mode="single")
            grid_options = gb.build()

            # Display AgGrid
            grid_response = AgGrid(
                scores_df,
                gridOptions=grid_options,
                height=400,
                fit_columns_on_grid_load=True,
                key=f"{self.prefix}_scores_grid",
                allow_unsafe_jscode=True
            )

            # Check for focused cell
            if "grid_response" in grid_response and "gridState" in grid_response["grid_response"]:
                focused_cell = grid_response["grid_response"]["gridState"].get("focusedCell", {})
                if focused_cell:
                    row_index = focused_cell.get("rowIndex")
                    col_id = focused_cell.get("colId")

                    if row_index is not None and col_id is not None and col_id != 'video_title':
                        video_id = scores_df.index[row_index]
                        product_id = col_id

                        video_row = videos_df[videos_df['video_id'] == video_id].iloc[0]
                        product = next(p for p in products if str(p['id']) == product_id)

                        st.subheader(t("video_details"))
                        st.write(f"**Title**: {video_row['title']}")
                        st.write(f"**URL**: [{video_row['url']}]({video_row['url']})")
                        st.write(f"**{t('video_keywords')}**: {', '.join(video_keywords_list[row_index])}")

                        st.subheader(t("product_details"))
                        st.write(f"**Title**: {product['title']}")
                        st.write(f"**URL**: [{product['url']}]({product['url']})")
                        product_keywords = str(product['keywords']).split(',') if product['keywords'] else []
                        product_keywords = [self.normalize_keyword(kw.strip()) for kw in product_keywords]
                        st.write(f"**{t('product_keywords')}**: {', '.join(product_keywords)}")
