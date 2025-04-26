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

translations["en"].update({
    "match_title": "Video-Product Relevance Scores",
    "no_videos_error": "No valid video data found",
    "no_products_error": "No products found in database",
})

translations["fr"].update({
    "match_title": "Scores de pertinence vidéo-produit",
    "no_videos_error": "Aucune donnée vidéo valide trouvée",
    "no_products_error": "Aucun produit trouvé dans la base de données",
})

class VideoProductMatchWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.db = ProductsDB()
        self.work_directory = self.plugin_manager.config["common"]["work_directory"]

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
        # Keywords from CSV
        keywords = str(video_row['keyword']).split(',') if pd.notnull(video_row['keyword']) else []

        # LLM-based keyword extraction from title and description
        prompt = f"""
        Extract key topics and keywords from the following video title and description.
        Return a comma-separated list of keywords (max 10).

        Title: {video_row['title']}
        Description: {video_row['description'] if pd.notnull(video_row['description']) else ''}
        """
        llm_response = self.process_with_llm(prompt)
        llm_keywords = llm_response.split(',') if llm_response else []

        # Combine and clean keywords
        all_keywords = list(set(keywords + llm_keywords))
        return ' '.join([kw.strip().lower() for kw in all_keywords if kw.strip()])

    def calculate_relevance_scores(self, videos_df, products):
        # Prepare text data
        video_texts = []
        video_ids = []
        for _, row in videos_df.iterrows():
            keywords = self.extract_video_keywords(row)
            if keywords:
                video_texts.append(keywords)
                video_ids.append(row['video_id'])

        product_texts = []
        product_ids = []
        for product in products:
            keywords = str(product['keywords']).lower() if product['keywords'] else ''
            product_texts.append(keywords)
            product_ids.append(product['id'])

        if not video_texts or not product_texts:
            return None

        # Calculate TF-IDF vectors and cosine similarity
        vectorizer = TfidfVectorizer()
        all_texts = video_texts + product_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        video_vectors = tfidf_matrix[:len(video_texts)]
        product_vectors = tfidf_matrix[len(video_texts):]

        similarity_matrix = cosine_similarity(video_vectors, product_vectors)

        # Create score DataFrame
        scores_df = pd.DataFrame(
            similarity_matrix,
            index=video_ids,
            columns=[f"Product_{pid}" for pid in product_ids]
        )
        return scores_df

    def display(self):
        st.title(t("match_title"))

        # Load videos
        videos_df = self.load_videos()
        if videos_df is None:
            st.error(t("no_videos_error"))
            return

        # Load products
        products = self.db.get_all_products()
        if not products:
            st.error(t("no_products_error"))
            return

        # Calculate scores
        scores_df = self.calculate_relevance_scores(videos_df, products)
        if scores_df is None:
            st.error(t("no_videos_error"))
            return

        # Display scores table
        st.dataframe(
            scores_df,
            use_container_width=True,
            height=400,
            column_config={
                col: st.column_config.NumberColumn(
                    col,
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0
                ) for col in scores_df.columns
            }
        )
