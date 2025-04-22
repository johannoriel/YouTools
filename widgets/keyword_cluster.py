from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import os
from sentence_transformers import SentenceTransformer
import pickle
from fuzzywuzzy import fuzz

# Traductions
translations["en"].update({
    "title": "Keyword Clustering Tool",
    "upload_json": "Upload JSON file with keywords (optional)",
    "keywords_file": "Keywords file (keywords.json)",
    "select_clusters": "Select number of clusters",
    "auto_clusters": "Automatic (optimize clusters)",
    "min_clusters": "Minimum number of clusters (auto mode)",
    "similarity_threshold": "Similarity threshold for cross-thematic keywords",
    "fuzzy_ratio": "Fuzzy ratio for merging similar keywords",
    "use_predefined_themes": "Use predefined themes",
    "predefined_themes": "Predefined themes (one per line)",
    "cluster_button": "Cluster Keywords",
    "no_file": "No keywords.json found in work directory and no file uploaded.",
    "results_title": "Thematic Clusters",
    "loading_model": "Loading embedding model...",
    "clustering": "Clustering keywords...",
    "export_results": "Export Results",
    "export_success": "Results exported to thematiques.json",
    "save_themes": "Save Themes",
    "themes_saved": "Themes saved to themes.json",
})

translations["fr"].update({
    "title": "Outil de regroupement de mots-clés",
    "upload_json": "Télécharger un fichier JSON avec les mots-clés (optionnel)",
    "keywords_file": "Fichier de mots-clés (keywords.json)",
    "select_clusters": "Sélectionner le nombre de clusters",
    "auto_clusters": "Automatique (optimisation des clusters)",
    "min_clusters": "Nombre minimum de clusters (mode auto)",
    "similarity_threshold": "Seuil de similarité pour les mots-clés transversaux",
    "fuzzy_ratio": "Ratio de fusion pour les mots-clés similaires",
    "use_predefined_themes": "Utiliser les thématiques prédéfinies",
    "predefined_themes": "Thématiques prédéfinies (une par ligne)",
    "cluster_button": "Regrouper les mots-clés",
    "no_file": "Aucun keywords.json trouvé dans le répertoire de travail et aucun fichier téléchargé.",
    "results_title": "Regroupements thématiques",
    "loading_model": "Chargement du modèle d'embeddings...",
    "clustering": "Regroupement des mots-clés...",
    "export_results": "Exporter les résultats",
    "export_success": "Résultats exportés vers thematiques.json",
    "save_themes": "Sauvegarder les thématiques",
    "themes_saved": "Thématiques sauvegardées dans themes.json",
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
        # Liste par défaut des thématiques
        self.default_themes = [
            "intelligence artificielle", "politique", "démocratie", "post-nationalisme",
            "santé", "méditation", "agentivité", "écologie", "technologie", "philosophie",
            "éthique", "science", "société", "économie", "conscience"
        ]
        self.themes = self.load_themes()

    def load_themes(self):
        """Charger les thématiques depuis themes.json ou utiliser la liste par défaut."""
        themes_file = os.path.join(self.work_directory, "themes.json")
        if os.path.exists(themes_file):
            try:
                with open(themes_file, "r", encoding="utf-8") as f:
                    themes = json.load(f)
                return [t.strip().lower() for t in themes if t.strip()]
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

    def load_model(self):
        """Charger le modèle SentenceTransformer."""
        if self.model is None:
            with st.spinner(t("loading_model")):
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return self.model

    def optimize_clusters(self, X, min_clusters=5, max_clusters=30):
        """Trouver le nombre optimal de clusters avec le score de silhouette."""
        best_n = min_clusters
        best_score = -1
        for n in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_n = n
        return best_n

    def cluster_keywords(self, keywords, n_clusters, auto_clusters=False, min_clusters=5, seuil=0.8, use_predefined=True):
        """Regrouper les mots-clés en clusters et assigner des thématiques."""
        model = self.load_model()
        embedding_file = os.path.join(self.work_directory, "keyword_embeddings.pkl")

        # Charger ou calculer les embeddings
        if os.path.exists(embedding_file):
            with open(embedding_file, "rb") as f:
                embeddings = pickle.load(f)
        else:
            embeddings = {mot: model.encode(mot) for mot in keywords}
            with open(embedding_file, "wb") as f:
                pickle.dump(embeddings, f)

        X = np.array(list(embeddings.values()))
        if auto_clusters:
            n_clusters = self.optimize_clusters(X, min_clusters, max_clusters=30)

        # Clustering avec K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        # Regrouper les mots par cluster
        clusters = {}
        for mot, label in zip(keywords, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(mot)

        # Assigner les thématiques
        resultats = {}
        if use_predefined:
            # Approche hybride : utiliser thématiques prédéfinies si similarité > 0.7
            embeddings_thematiques = {th: model.encode(th) for th in self.themes}
            for label, mots in clusters.items():
                emb_cluster = np.mean([embeddings[mot] for mot in mots], axis=0)
                similarites = {
                    th: cosine_similarity([emb_cluster], [embeddings_thematiques[th]])[0][0]
                    for th in self.themes
                }
                max_sim = max(similarites.values())
                if max_sim > 0.7:  # Seuil pour utiliser une thématique prédéfinie
                    th_choisie = max(similarites, key=similarites.get)
                else:
                    th_choisie = max(mots, key=lambda mot: keywords[mot])  # Mot-clé le plus fréquent
                resultats[th_choisie] = mots
        else:
            # Nommage par mot-clé le plus fréquent
            for label, mots in clusters.items():
                th_choisie = max(mots, key=lambda mot: keywords[mot])
                resultats[th_choisie] = mots

        # Gérer la transversalité
        resultats_transversaux = {th: [] for th in resultats}
        for mot in keywords:
            emb_mot = embeddings[mot]
            for th in resultats:
                emb_th = embeddings[th]
                sim = cosine_similarity([emb_mot], [emb_th])[0][0]
                if sim > seuil:
                    resultats_transversaux[th].append(mot)

        # Calculer les poids
        poids_thematiques = {
            th: sum(keywords.get(mot, 0) for mot in mots)
            for th, mots in resultats_transversaux.items()
        }

        return resultats_transversaux, poids_thematiques

    def display(self):
        """Afficher l'interface Streamlit."""
        st.title(t("title"))

        # Gestion des thématiques prédéfinies
        st.subheader(t("predefined_themes"))
        themes_text = "\n".join(self.themes)
        new_themes = st.text_area(
            t("predefined_themes"),
            value=themes_text,
            height=200
        )
        if st.button(t("save_themes")):
            themes = [t.strip().lower() for t in new_themes.split("\n") if t.strip()]
            self.themes = themes
            self.save_themes(themes)

        # Checkbox pour thématiques prédéfinies
        use_predefined = st.checkbox(t("use_predefined_themes"), value=True)

        # Chargement des mots-clés
        st.subheader(t("keywords_file"))
        json_file = st.file_uploader(t("upload_json"), type=["json"])
        fuzzy_ratio = st.session_state.get(f"{self.prefix}_fuzzy_ratio", 90)
        keywords = load_keywords(self.work_directory, fuzzy_ratio=fuzzy_ratio, json_file=json_file)
        if not keywords:
            st.warning(t("no_file"))
            return

        # Paramètres de clustering
        cluster_option = st.radio(
            t("select_clusters"),
            options=[t("auto_clusters"), "Manuel"],
            index=0
        )
        n_clusters = 10
        min_clusters = 5
        seuil = 0.8
        fuzzy_ratio = 90

        # Sliders sur une même ligne
        col1, col2, col3 = st.columns(3)
        with col1:
            if cluster_option == "Manuel":
                n_clusters = st.slider(
                    "Nombre de clusters",
                    min_value=2,
                    max_value=50,
                    value=10,
                    step=1,
                    key=f"{self.prefix}_n_clusters"
                )
            else:
                min_clusters = st.slider(
                    t("min_clusters"),
                    min_value=5,
                    max_value=20,
                    value=5,
                    step=1,
                    key=f"{self.prefix}_min_clusters"
                )
        with col2:
            seuil = st.slider(
                t("similarity_threshold"),
                min_value=0.6,
                max_value=0.9,
                value=0.8,
                step=0.05,
                key=f"{self.prefix}_seuil"
            )
        with col3:
            fuzzy_ratio = st.slider(
                t("fuzzy_ratio"),
                min_value=70,
                max_value=100,
                value=90,
                step=5,
                key=f"{self.prefix}_fuzzy_ratio"
            )

        # Bouton pour lancer le clustering
        if st.button(t("cluster_button")):
            with st.spinner(t("clustering")):
                auto_clusters = (cluster_option == t("auto_clusters"))
                resultats, poids = self.cluster_keywords(
                    keywords, n_clusters, auto_clusters, min_clusters, seuil, use_predefined
                )

                # Afficher les résultats
                st.subheader(t("results_title"))
                thematiques_triees = sorted(
                    [(th, poids[th]) for th in resultats if poids[th] > 0],
                    key=lambda x: x[1],
                    reverse=True
                )
                data = []
                for th, poids_th in thematiques_triees:
                    mots = resultats[th]
                    mots_str = ", ".join(mots)
                    data.append({"Thématique": th, "Poids": poids_th, "Mots-clés": mots_str})
                st.table(data)

                # Bouton d'exportation
                if st.button(t("export_results")):
                    with open(os.path.join(self.work_directory, "thematiques.json"), "w", encoding="utf-8") as f:
                        json.dump(
                            {"thematiques": {th: {"mots": resultats[th], "poids": poids[th]} for th in resultats}},
                            f,
                            ensure_ascii=False,
                            indent=2
                        )
                    st.success(t("export_success"))
