from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
from sentence_transformers import SentenceTransformer
import pickle
from fuzzywuzzy import fuzz
import pandas as pd

# Traductions
translations["en"].update({
    "title": "Keyword Clustering Tool",
    "upload_json": "Upload JSON file with keywords (optional)",
    "keywords_file": "Keywords file (keywords.json)",
    "predefined_themes": "Themes (format: theme: keyword1, keyword2, ...)",
    "save_themes": "Save Themes",
    "themes_saved": "Themes saved to themes.json",
    "suggest_themes": "Suggest New Themes",
    "select_themes": "Select themes to add",
    "confirm_themes": "Confirm Theme Selection",
    "suggest_keywords": "Suggest Keywords for Theme",
    "select_theme": "Select a theme",
    "current_keywords": "Current keywords",
    "suggested_keywords": "Suggested keywords",
    "add_keywords": "Add Selected Keywords",
    "similarity_threshold": "Similarity threshold for suggestions",
    "fuzzy_ratio": "Fuzzy ratio for merging similar keywords",
    "no_file": "No keywords.json found in work directory and no file uploaded.",
    "no_keywords_left": "No keywords left to process.",
    "results_title": "Current Themes",
    "remaining_keywords": "Remaining Keywords",
    "thematized_keywords": "Thematized Keywords",
    "remaining_count": "Remaining keywords count",
    "remaining_weight": "Remaining keywords total weight",
    "suggest_category": "Suggest Category for Keyword",
    "suggested_categories": "Suggested Categories",
    "confirm_categories": "Confirm Category Selection",
})

translations["fr"].update({
    "title": "Outil de regroupement de mots-clés",
    "upload_json": "Télécharger un fichier JSON avec les mots-clés (optionnel)",
    "keywords_file": "Fichier de mots-clés (keywords.json)",
    "predefined_themes": "Thématiques (format : thématique : motclé1, motclé2, ...)",
    "save_themes": "Sauvegarder les thématiques",
    "themes_saved": "Thématiques sauvegardées dans themes.json",
    "suggest_themes": "Suggérer de nouvelles thématiques",
    "select_themes": "Sélectionner les thématiques à ajouter",
    "confirm_themes": "Confirmer la sélection des thématiques",
    "suggest_keywords": "Suggérer des mots-clés pour une thématique",
    "select_theme": "Sélectionner une thématique",
    "current_keywords": "Mots-clés actuels",
    "suggested_keywords": "Mots-clés suggérés",
    "add_keywords": "Ajouter les mots-clés sélectionnés",
    "similarity_threshold": "Seuil de similarité pour les suggestions",
    "fuzzy_ratio": "Ratio de fusion pour les mots-clés similaires",
    "no_file": "Aucun keywords.json trouvé dans le répertoire de travail et aucun fichier téléchargé.",
    "no_keywords_left": "Aucun mot-clé restant à traiter.",
    "results_title": "Thématiques actuelles",
    "remaining_keywords": "Mots-clés restants",
    "thematized_keywords": "Mots-clés thématisés",
    "remaining_count": "Nombre de mots-clés restants",
    "remaining_weight": "Poids total des mots-clés restants",
    "suggest_category": "Suggérer une catégorie pour un mot-clé",
    "suggested_categories": "Catégories suggérées",
    "confirm_categories": "Confirmer la sélection des catégories",
})

@st.cache_data
def load_keywords(work_directory, fuzzy_ratio=90, json_file=None):
    """Charger et prétraiter les mots-clés depuis keywords.json ou un fichier uploadé."""
    keywords = None
    if json_file:
        try:
            keywords = json.load(json_file)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier uploadé : {e}")
            return None
    else:
        keywords_file = os.path.join(work_directory, "keywords.json")
        if os.path.exists(keywords_file):
            try:
                with open(keywords_file, "r", encoding="utf-8") as f:
                    keywords = json.load(f)
            except Exception as e:
                st.error(f"Erreur lors du chargement de keywords.json : {e}")
                return None

    if not keywords:
        return None

    keywords = {k.lower(): v for k, v in keywords.items() if isinstance(v, (int, float))}
    # Fusionner les mots-clés similaires
    merged_keywords = {}
    used = set()
    for k1 in keywords:
        if k1 not in used:
            merged_keywords[k1] = keywords[k1]
            for k2 in keywords:
                if k2 != k1 and k2 not in used:
                    if fuzz.ratio(k1, k2) > fuzzy_ratio or k1 in k2 or k2 in k1:
                        merged_keywords[k1] += keywords[k2]
                        used.add(k2)
            used.add(k1)
    return merged_keywords

class KeywordClusteringWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.model = None
        self.work_directory = self.plugin_manager.config["common"]["work_directory"]
        # Thématiques par défaut
        self.default_themes = {
            "intelligence artificielle": [],
            "politique": [],
            "démocratie": [],
            "post-nationalisme": [],
            "santé": [],
            "méditation": [],
            "agentivité": [],
            "écologie": [],
            "technologie": [],
            "philosophie": [],
            "éthique": [],
            "science": [],
            "société": [],
            "économie": [],
            "conscience": []
        }
        self.themes = self.load_themes()
        # Initialiser l'état de session
        if f"{self.prefix}_themes_text" not in st.session_state:
            st.session_state[f"{self.prefix}_themes_text"] = self.themes_to_text(self.themes)

    def load_themes(self):
        """Charger les thématiques depuis themes.json ou utiliser la liste par défaut."""
        themes_file = os.path.join(self.work_directory, "themes.json")
        if os.path.exists(themes_file):
            try:
                with open(themes_file, "r", encoding="utf-8") as f:
                    themes = json.load(f)
                return {k.lower(): v for k, v in themes.items()}
            except Exception as e:
                st.warning(f"Erreur lors du chargement de themes.json : {e}. Utilisation de la liste par défaut.")
        return self.default_themes

    def save_themes(self, themes):
        """Sauvegarder les thématiques dans themes.json."""
        themes_file = os.path.join(self.work_directory, "themes.json")
        try:
            with open(themes_file, "w", encoding="utf-8") as f:
                json.dump(themes, f, ensure_ascii=False, indent=2)
            st.success(t("themes_saved"))
        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde de themes.json : {e}")

    def themes_to_text(self, themes):
        """Convertir les thématiques en texte pour le textarea."""
        return "\n".join([f"{th}: {', '.join(mots)}" if mots else f"{th}:" for th, mots in themes.items()])

    def text_to_themes(self, text):
        """Convertir le texte du textarea en dictionnaire de thématiques."""
        themes = {}
        for line in text.split("\n"):
            if ":" in line:
                theme, keywords = line.split(":", 1)
                theme = theme.strip().lower()
                keywords = [k.strip().lower() for k in keywords.split(",") if k.strip()]
                themes[theme] = keywords
        return themes

    def get_remaining_keywords(self, keywords, themes):
        """Retourner les mots-clés non assignés, leur nombre et leur poids total."""
        assigned_keywords = set()
        for mots in themes.values():
            assigned_keywords.update(mots)
        remaining = {k: v for k, v in keywords.items() if k not in assigned_keywords}
        count = len(remaining)
        weight = sum(remaining.values())
        return remaining, count, weight

    def get_thematized_keywords(self, keywords, themes):
        """Retourner les mots-clés thématisés avec leurs poids."""
        thematized = {}
        for mots in themes.values():
            for mot in mots:
                if mot in keywords:
                    thematized[mot] = keywords[mot]
        return thematized

    def load_model(self):
        """Charger le modèle SentenceTransformer."""
        if self.model is None:
            with st.spinner(t("results_title")):
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return self.model

    def suggest_themes(self, keywords, n_clusters=5):
        """Suggérer de nouvelles thématiques via clustering."""
        if not keywords:
            return []
        model = self.load_model()
        embeddings = {mot: model.encode(mot) for mot in keywords}
        X = np.array(list(embeddings.values()))

        # Clustering avec K-Means
        kmeans = KMeans(n_clusters=min(n_clusters, len(keywords)), random_state=42)
        labels = kmeans.fit_predict(X)

        # Regrouper les mots par cluster
        clusters = {}
        for mot, label in zip(keywords, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(mot)

        # Nommer les thématiques par mot-clé le plus fréquent
        themes = []
        for mots in clusters.values():
            if mots:
                theme = max(mots, key=lambda mot: keywords[mot])
                themes.append(theme)
        return themes

    def suggest_keywords(self, theme, keywords, seuil=0.8):
        """Suggérer des mots-clés proches sémantiquement d'une thématique."""
        if not keywords:
            return []
        model = self.load_model()
        emb_theme = model.encode(theme)
        embeddings = {mot: model.encode(mot) for mot in keywords}
        suggestions = []
        for mot in keywords:
            sim = cosine_similarity([emb_theme], [embeddings[mot]])[0][0]
            if sim > seuil:
                suggestions.append((mot, sim))
        # Trier par similarité décroissante et limiter à 20 suggestions
        suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)[:20]
        return [mot for mot, _ in suggestions]

    def suggest_category_for_keyword(self, keyword, themes, seuil=0.8):
        """Suggérer la catégorie la plus proche pour un mot-clé donné."""
        if not themes:
            return []
        model = self.load_model()
        emb_keyword = model.encode(keyword)
        suggestions = []
        for theme in themes:
            emb_theme = model.encode(theme)
            sim = cosine_similarity([emb_keyword], [emb_theme])[0][0]
            if sim > seuil:
                suggestions.append((theme, sim))
        # Trier par similarité décroissante
        suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
        return suggestions

    def display(self):
        """Afficher l'interface Streamlit."""
        st.title(t("title"))

        # Gestion des thématiques
        st.subheader(t("predefined_themes"))
        themes_text = st.text_area(
            t("predefined_themes"),
            value=st.session_state[f"{self.prefix}_themes_text"],
            height=200,
            key=f"{self.prefix}_themes_text_input"
        )
        self.themes = self.text_to_themes(themes_text)
        if st.button(t("save_themes")):
            self.save_themes(self.themes)
            st.session_state[f"{self.prefix}_themes_text"] = themes_text

        # Chargement des mots-clés
        st.subheader(t("keywords_file"))
        json_file = st.file_uploader(t("upload_json"), type=["json"])
        keywords = load_keywords(self.work_directory, st.session_state.get(f"{self.prefix}_fuzzy_ratio", 90), json_file)
        if not keywords:
            st.warning(t("no_file"))
            return

        # Afficher les mots-clés restants et thématisés
        st.subheader(t("remaining_keywords"))
        remaining_keywords, remaining_count, remaining_weight = self.get_remaining_keywords(keywords, self.themes)
        thematized_keywords = self.get_thematized_keywords(keywords, self.themes)

        col1, col2, col3, col4 = st.columns(4)

        col1.write(f"**{t('remaining_count')}**: {remaining_count}")
        col2.write(f"**{t('remaining_weight')}**: {remaining_weight}")
        remaining_text = "\n".join([f"{k}: {v}" for k, v in remaining_keywords.items()])
        thematized_text = "\n".join([f"{k}: {v}" for k, v in thematized_keywords.items()])
        col3.text_area(
            t("remaining_keywords"),
            value=remaining_text if remaining_text else "Aucun",
            height=100,
            disabled=True,
            key=f"{self.prefix}_remaining_keywords"
        )
        col4.text_area(
            t("thematized_keywords"),
            value=thematized_text if thematized_text else "Aucun",
            height=100,
            disabled=True,
            key=f"{self.prefix}_thematized_keywords"
        )

        # Paramètres
        col1, col2 = st.columns(2)
        with col1:
            seuil = st.slider(
                t("similarity_threshold"),
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.01,
                key=f"{self.prefix}_seuil"
            )
        with col2:
            fuzzy_ratio = st.slider(
                t("fuzzy_ratio"),
                min_value=70,
                max_value=100,
                value=90,
                step=5,
                key=f"{self.prefix}_fuzzy_ratio"
            )

        # Étape 1 : Suggérer de nouvelles thématiques
        st.subheader(t("suggest_themes"))
        if not remaining_keywords:
            st.warning(t("no_keywords_left"))
        else:
            n_clusters = st.slider(
                "Nombre de thématiques à suggérer",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key=f"{self.prefix}_n_clusters_themes"
            )
            if st.button(t("suggest_themes")):
                with st.spinner(t("results_title")):
                    suggested_themes = self.suggest_themes(remaining_keywords, n_clusters)
                    st.session_state[f"{self.prefix}_suggested_themes"] = suggested_themes
            if f"{self.prefix}_suggested_themes" in st.session_state and st.session_state[f"{self.prefix}_suggested_themes"]:
                selected_themes = st.multiselect(
                    t("select_themes"),
                    options=st.session_state[f"{self.prefix}_suggested_themes"],
                    default=[],
                    key=f"{self.prefix}_select_themes"
                )
                if selected_themes and st.button(t("confirm_themes")):
                    for theme in selected_themes:
                        if theme not in self.themes:
                            self.themes[theme] = []
                    st.session_state[f"{self.prefix}_themes_text"] = self.themes_to_text(self.themes)
                    st.rerun()

        # Étape 2 : Suggérer des mots-clés pour une thématique
        st.subheader(t("suggest_keywords"))
        if self.themes:
            selected_theme = st.selectbox(
                t("select_theme"),
                options=list(self.themes.keys()),
                key=f"{self.prefix}_select_theme"
            )
            st.write(f"**{t('current_keywords')}**: {', '.join(self.themes[selected_theme]) if self.themes[selected_theme] else 'Aucun'}")
            if st.button(t("suggest_keywords")):
                with st.spinner(t("results_title")):
                    suggested_keywords = self.suggest_keywords(selected_theme, remaining_keywords, seuil)
                    st.session_state[f"{self.prefix}_suggested_keywords"] = suggested_keywords
            if f"{self.prefix}_suggested_keywords" in st.session_state and st.session_state[f"{self.prefix}_suggested_keywords"]:
                selected_keywords = st.multiselect(
                    t("suggested_keywords"),
                    options=st.session_state[f"{self.prefix}_suggested_keywords"],
                    default=[],
                    key=f"{self.prefix}_select_keywords"
                )
                if selected_keywords and st.button(t("add_keywords")):
                    self.themes[selected_theme].extend([k for k in selected_keywords if k not in self.themes[selected_theme]])
                    st.session_state[f"{self.prefix}_themes_text"] = self.themes_to_text(self.themes)
                    st.rerun()

        # Étape 3 : Suggérer une catégorie pour un mot-clé
        st.subheader(t("suggest_category"))
        if not remaining_keywords:
            st.warning(t("no_keywords_left"))
        else:
            selected_keyword = st.selectbox(
                t("remaining_keywords"),
                options=list(remaining_keywords.keys()),
                key=f"{self.prefix}_select_keyword"
            )
            if st.button(t("suggest_category")):
                with st.spinner(t("results_title")):
                    suggested_categories = self.suggest_category_for_keyword(selected_keyword, self.themes.keys(), seuil)
                    if suggested_categories:
                        df = pd.DataFrame(suggested_categories, columns=["Category", "Similarity"])
                        st.session_state[f"{self.prefix}_suggested_categories"] = df
                    else:
                        st.warning("Aucune catégorie suggérée.")
            if f"{self.prefix}_suggested_categories" in st.session_state and not st.session_state[f"{self.prefix}_suggested_categories"].empty:
                selected_categories = st.multiselect(
                    t("suggested_categories"),
                    options=st.session_state[f"{self.prefix}_suggested_categories"]["Category"].tolist(),
                    default=[],
                    key=f"{self.prefix}_select_categories"
                )
                if selected_categories and st.button(t("confirm_categories")):
                    for category in selected_categories:
                        if category in self.themes:
                            if selected_keyword not in self.themes[category]:
                                self.themes[category].append(selected_keyword)
                    st.session_state[f"{self.prefix}_themes_text"] = self.themes_to_text(self.themes)
                    st.rerun()
