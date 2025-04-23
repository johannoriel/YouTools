from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import json
import os
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
    "suggest_keywords": "Suggest Keywords for Theme",
    "select_theme": "Select a theme",
    "current_keywords": "Current keywords",
    "suggested_keywords": "Suggested keywords",
    "add_keywords": "Add Selected Keywords",
    "fuzzy_ratio": "Fuzzy ratio for merging similar keywords",
    "no_file": "No keywords.json found in work directory and no file uploaded.",
    "no_keywords_left": "No keywords left to process.",
    "results_title": "Current Themes",
    "remaining_keywords": "Remaining Keywords",
    "thematized_keywords": "Thematized Keywords",
    "remaining_count": "Remaining keywords count",
    "remaining_weight": "Remaining keywords total weight",
    "suggest_category": "Suggest Category for Keyword",
    "suggest_mass_categories": "Suggest Categories for All Keywords",
    "suggested_categories": "Suggested Categories",
    "confirm_categories": "Confirm Category Selection",
    "notfound": "No theme found",
    "assign_keywords": "Assign Keywords to a Theme",
    "filter_by_theme": "Filter by another theme",
    "no_filter": "No filter",
    "show_all_keywords": "Show all keywords",
    "add_selected_keywords": "Add Selected Keywords",
    "search_keywords": "Search Keywords for a Theme"
})

translations["fr"].update({
    "title": "Outil de regroupement de mots-clés",
    "upload_json": "Télécharger un fichier JSON avec les mots-clés (optionnel)",
    "keywords_file": "Fichier de mots-clés (keywords.json)",
    "predefined_themes": "Thématiques (format : thématique : motclé1, motclé2, ...)",
    "save_themes": "Sauvegarder les thématiques",
    "themes_saved": "Thématiques sauvegardées dans themes.json",
    "suggest_keywords": "Suggérer des mots-clés pour une thématique",
    "select_theme": "Sélectionner une thématique",
    "current_keywords": "Mots-clés actuels",
    "suggested_keywords": "Mots-clés suggérés",
    "add_keywords": "Ajouter les mots-clés sélectionnés",
    "fuzzy_ratio": "Ratio de fusion pour les mots-clés similaires",
    "no_file": "Aucun keywords.json trouvé dans le répertoire de travail et aucun fichier téléchargé.",
    "no_keywords_left": "Aucun mot-clé restant à traiter.",
    "results_title": "Thématiques actuelles",
    "remaining_keywords": "Mots-clés restants",
    "thematized_keywords": "Mots-clés thématisés",
    "remaining_count": "Nombre de mots-clés restants",
    "remaining_weight": "Poids total des mots-clés restants",
    "suggest_category": "Suggérer une catégorie pour un mot-clé",
    "suggest_mass_categories": "Suggérer des catégories pour tous les mots-clés",
    "suggested_categories": "Catégories suggérées",
    "confirm_categories": "Confirmer la sélection des catégories",
    "notfound": "Aucune thématique trouvée",
    "assign_keywords": "Assigner des mots-clés à une thématique",
    "filter_by_theme": "Filtrer par une autre thématique",
    "no_filter": "Aucun filtre",
    "show_all_keywords": "Afficher tous les mots-clés",
    "add_selected_keywords": "Ajouter les mots-clés sélectionnés",
    "search_keywords": "Chercher les mots-clés pour une thématique"
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

    keywords = {k.lower(): v for k, v in keywords.items()
                if isinstance(v, (int, float))}
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
            st.session_state[f"{self.prefix}_themes_text"] = self.themes_to_text(
                self.themes)

    def load_themes(self):
        """Charger les thématiques depuis themes.json ou utiliser la liste par défaut."""
        themes_file = os.path.join(self.work_directory, "themes.json")
        if os.path.exists(themes_file):
            try:
                with open(themes_file, "r", encoding="utf-8") as f:
                    themes = json.load(f)
                return {k.lower(): v for k, v in themes.items()}
            except Exception as e:
                st.warning(
                    f"Erreur lors du chargement de themes.json : {e}. Utilisation de la liste par défaut.")
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
        """Convertir les thématiques en texte pour le textarea avec une ligne vierge entre chaque thématique."""
        lines = []
        for th, mots in themes.items():
            line = f"{th}: {', '.join(mots)}" if mots else f"{th}:"
            lines.append(line)
        return "\n\n".join(lines)

    def text_to_themes(self, text):
        """Convertir le texte du textarea en dictionnaire de thématiques, en ignorant les lignes vides."""
        themes = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line:  # Ignorer les lignes vides
                continue
            if ":" in line:
                theme, keywords = line.split(":", 1)
                theme = theme.strip().lower()
                keywords = [k.strip().lower()
                            for k in keywords.split(",") if k.strip()]
                themes[theme] = keywords
        return themes

    def get_remaining_keywords(self, keywords, themes):
        """Retourner les mots-clés non assignés, leur nombre et leur poids total."""
        assigned_keywords = set()
        for mots in themes.values():
            assigned_keywords.update(mots)
        remaining = {k: v for k, v in keywords.items()
                     if k not in assigned_keywords}
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

    def suggest_with_llm(self, items, targets, mode="keyword_to_theme", theme=None):
        """Factorisation de la suggestion avec LLM pour associer des mots-clés à des thématiques ou vérifier l'appartenance."""
        suggestions = []
        total = len(items)
        progress_bar = st.progress(0)
        targets_list = ", ".join(targets)

        for i, item in enumerate(items):
            if mode == "keyword_to_theme":
                prompt = (
                    f"Étant donné le mot-clé \"{item}\" et la liste suivante de thématiques : {targets_list}, "
                    "sélectionnez la thématique la plus appropriée à laquelle le mot-clé appartient. "
                    "Si aucune thématique n'est pertinente, retournez \"notfound\". "
                    "Fournissez uniquement le nom de la thématique ou \"notfound\" comme réponse."
                )
            elif mode == "keyword_to_theme_binary":
                prompt = (
                    f"Étant donné le mot-clé \"{item}\" et la thématique \"{theme}\", "
                    "ce mot-clé appartient-il à cette thématique ? "
                    "Répondez uniquement par \"oui\" ou \"non\"."
                )

            try:
                response = self.plugin_manager.get_plugin(
                    self.name).process_with_llm(prompt)
                response = response.strip().lower()
                if mode == "keyword_to_theme":
                    if response in targets or response == "notfound":
                        suggestions.append(
                            (item, response, 1.0 if response != "notfound" else 0.0))
                elif mode == "keyword_to_theme_binary":
                    if response == "oui":
                        suggestions.append((item, theme, 1.0))
            except Exception as e:
                st.error(f"Erreur lors de l'appel au LLM pour {item} : {e}")
            progress_bar.progress((i + 1) / total)

        return suggestions

    def suggest_keywords(self, theme, keywords):
        """Suggérer des mots-clés pour une thématique en vérifiant chaque mot-clé avec le LLM."""
        if not keywords:
            return []
        suggestions = self.suggest_with_llm(
            keywords, [theme], mode="keyword_to_theme_binary", theme=theme)
        return [kw for kw, _, _ in suggestions]

    def suggest_category_for_keyword(self, keyword, themes):
        """Suggérer la catégorie la plus proche pour un mot-clé donné en utilisant le LLM."""
        if not themes:
            return []
        suggestions = self.suggest_with_llm(
            [keyword], themes, mode="keyword_to_theme")
        return [(theme, conf) for _, theme, conf in suggestions if theme != "notfound"]

    def suggest_mass_categories(self, keywords, themes):
        """Suggérer des catégories pour tous les mots-clés non assignés en utilisant le LLM."""
        if not keywords or not themes:
            return []
        return self.suggest_with_llm(keywords, themes, mode="keyword_to_theme")

    def display_themes_management(self):
        """Afficher la section de gestion des thématiques."""
        st.subheader(t("predefined_themes"))
        themes_text = st.text_area(
            t("predefined_themes"),
            value=st.session_state[f"{self.prefix}_themes_text"],
            height=200,
            key=f"{self.prefix}_themes_text_input"
        )
        self.themes = self.text_to_themes(themes_text)
        if st.button(t("save_themes"), key=f"{self.prefix}_save_themes_button"):
            self.save_themes(self.themes)
            st.session_state[f"{self.prefix}_themes_text"] = themes_text

    def display_keywords_loading(self):
        """Afficher la section de chargement des mots-clés."""
        st.subheader(t("keywords_file"))
        json_file = st.file_uploader(
            t("upload_json"),
            type=["json"],
            key=f"{self.prefix}_json_uploader"
        )
        keywords = load_keywords(self.work_directory, st.session_state.get(
            f"{self.prefix}_fuzzy_ratio", 90), json_file)
        if not keywords:
            st.warning(t("no_file"))
            return None
        return keywords

    def display_keywords_stats(self, keywords, remaining_keywords, thematized_keywords):
        """Afficher les statistiques des mots-clés restants et thématisés."""
        st.subheader(t("remaining_keywords"))
        remaining_count = len(remaining_keywords)
        remaining_weight = sum(remaining_keywords.values())

        col1, col2, col3, col4 = st.columns(4)
        col1.write(f"**{t('remaining_count')}**: {remaining_count}")
        col2.write(f"**{t('remaining_weight')}**: {remaining_weight}")
        remaining_text = "\n".join(
            [f"{k}: {v}" for k, v in remaining_keywords.items()])
        thematized_text = "\n".join(
            [f"{k}: {v}" for k, v in thematized_keywords.items()])
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

    def display_parameters(self):
        """Afficher les paramètres."""
        st.slider(
            t("fuzzy_ratio"),
            min_value=70,
            max_value=100,
            value=90,
            step=5,
            key=f"{self.prefix}_fuzzy_ratio"
        )

    def display_suggest_keywords(self, keywords, remaining_keywords):
        """Afficher la section pour suggérer des mots-clés pour une thématique."""
        st.subheader(t("suggest_keywords"))
        if self.themes:
            selected_theme = st.selectbox(
                t("select_theme"),
                options=list(self.themes.keys()),
                key=f"{self.prefix}_select_theme"
            )
            st.write(
                f"**{t('current_keywords')}**: {', '.join(self.themes[selected_theme]) if self.themes[selected_theme] else 'Aucun'}")
            show_all_keywords_suggest = st.checkbox(
                t("show_all_keywords"),
                value=False,
                key=f"{self.prefix}_show_all_keywords_suggest"
            )
            keywords_to_suggest = list(keywords.keys(
            )) if show_all_keywords_suggest else list(remaining_keywords.keys())
            if st.button(t("suggest_keywords"), key=f"{self.prefix}_suggest_keywords_button"):
                with st.spinner(t("results_title")):
                    suggested_keywords = self.suggest_keywords(
                        selected_theme, keywords_to_suggest)
                    st.session_state[f"{self.prefix}_suggested_keywords"] = suggested_keywords
            if f"{self.prefix}_suggested_keywords" in st.session_state and st.session_state[f"{self.prefix}_suggested_keywords"]:
                selected_keywords = st.multiselect(
                    t("suggested_keywords"),
                    options=st.session_state[f"{self.prefix}_suggested_keywords"],
                    default=[],
                    key=f"{self.prefix}_select_keywords"
                )
                if selected_keywords and st.button(t("add_keywords"), key=f"{self.prefix}_add_keywords_button"):
                    self.themes[selected_theme].extend(
                        [k for k in selected_keywords if k not in self.themes[selected_theme]])
                    st.session_state[f"{self.prefix}_themes_text"] = self.themes_to_text(
                        self.themes)
                    st.rerun()

    def display_suggest_category(self, remaining_keywords):
        """Afficher la section pour suggérer une catégorie pour un mot-clé."""
        st.subheader(t("suggest_category"))
        if not remaining_keywords:
            st.warning(t("no_keywords_left"))
        else:
            selected_keyword = st.selectbox(
                t("remaining_keywords"),
                options=list(remaining_keywords.keys()),
                key=f"{self.prefix}_select_keyword"
            )
            if st.button(t("suggest_category"), key=f"{self.prefix}_suggest_category_button"):
                with st.spinner(t("results_title")):
                    suggested_categories = self.suggest_category_for_keyword(
                        selected_keyword, self.themes.keys())
                    if suggested_categories:
                        df = pd.DataFrame(suggested_categories, columns=[
                                          "Category", "Confidence"])
                        st.session_state[f"{self.prefix}_suggested_categories"] = df
                    else:
                        st.warning("Aucune catégorie suggérée.")
            if f"{self.prefix}_suggested_categories" in st.session_state and not st.session_state[f"{self.prefix}_suggested_categories"].empty:
                selected_categories = st.multiselect(
                    t("suggested_categories"),
                    options=st.session_state[f"{self.prefix}_suggested_categories"]["Category"].tolist(
                    ),
                    default=[],
                    key=f"{self.prefix}_select_categories"
                )
                if selected_categories and st.button(t("confirm_categories"), key=f"{self.prefix}_confirm_categories_button"):
                    for category in selected_categories:
                        if category in self.themes:
                            if selected_keyword not in self.themes[category]:
                                self.themes[category].append(selected_keyword)
                    st.session_state[f"{self.prefix}_themes_text"] = self.themes_to_text(
                        self.themes)
                    st.rerun()

    def display_suggest_mass_categories(self, remaining_keywords):
        """Afficher la section pour suggérer des catégories pour tous les mots-clés non assignés."""
        st.subheader(t("suggest_mass_categories"))
        if not remaining_keywords:
            st.warning(t("no_keywords_left"))
        else:
            if st.button(t("suggest_mass_categories"), key=f"{self.prefix}_suggest_mass_categories_button"):
                with st.spinner(t("results_title")):
                    suggested_mass_categories = self.suggest_mass_categories(
                        remaining_keywords, self.themes.keys())
                    if suggested_mass_categories:
                        df = pd.DataFrame(suggested_mass_categories, columns=[
                                          "Keyword", "Category", "Confidence"])
                        df = df.sort_values(by="Category")
                        st.session_state[f"{self.prefix}_suggested_mass_categories"] = df
                    else:
                        st.warning("Aucune catégorie suggérée.")
            if f"{self.prefix}_suggested_mass_categories" in st.session_state and not st.session_state[f"{self.prefix}_suggested_mass_categories"].empty:
                selected = st.dataframe(
                    st.session_state[f"{self.prefix}_suggested_mass_categories"],
                    selection_mode="multi-row",
                    on_select="rerun",
                    key=f"{self.prefix}_mass_categories_dataframe"
                )
                if st.button(t("confirm_categories"), key=f"{self.prefix}_confirm_mass_categories_button"):
                    selected_rows = selected["selection"]["rows"]
                    if selected_rows:
                        for row in selected_rows:
                            keyword = st.session_state[f"{self.prefix}_suggested_mass_categories"].iloc[row]["Keyword"]
                            category = st.session_state[f"{self.prefix}_suggested_mass_categories"].iloc[row]["Category"]
                            if category in self.themes and category != "notfound":
                                if keyword not in self.themes[category]:
                                    self.themes[category].append(keyword)
                        st.session_state[f"{self.prefix}_themes_text"] = self.themes_to_text(
                            self.themes)
                        st.rerun()

    def display_assign_keywords(self, keywords, remaining_keywords):
        """Afficher la section pour assigner manuellement des mots-clés à une thématique."""
        st.subheader(t("assign_keywords"))
        if not self.themes:
            st.warning("Aucune thématique disponible.")
        else:
            selected_theme = st.selectbox(
                t("select_theme"),
                options=list(self.themes.keys()),
                key=f"{self.prefix}_assign_theme"
            )
            st.write(
                f"**{t('current_keywords')}**: {', '.join(self.themes[selected_theme]) if self.themes[selected_theme] else 'Aucun'}")
            filter_theme = st.selectbox(
                t("filter_by_theme"),
                options=["Aucun filtre"] + list(self.themes.keys()),
                key=f"{self.prefix}_filter_theme"
            )
            show_all_keywords = st.checkbox(
                t("show_all_keywords"),
                value=False,
                key=f"{self.prefix}_show_all_keywords"
            )
            keywords_to_show = list(keywords.keys()) if show_all_keywords else list(
                remaining_keywords.keys())
            if filter_theme != "Aucun filtre":
                keywords_to_show = [
                    kw for kw in keywords_to_show if kw in self.themes[filter_theme]]
            if keywords_to_show:
                df_keywords = pd.DataFrame({"Keyword": keywords_to_show})
                selected_keywords = st.dataframe(
                    df_keywords,
                    selection_mode="multi-row",
                    on_select="rerun",
                    key=f"{self.prefix}_keywords_dataframe"
                )
                if st.button(t("add_selected_keywords"), key=f"{self.prefix}_add_selected_keywords_button"):
                    selected_rows = selected_keywords["selection"]["rows"]
                    if selected_rows:
                        for row in selected_rows:
                            keyword = df_keywords.loc[row, "Keyword"]
                            if keyword not in self.themes[selected_theme]:
                                self.themes[selected_theme].append(keyword)
                        st.session_state[f"{self.prefix}_themes_text"] = self.themes_to_text(
                            self.themes)
                        st.rerun()
            else:
                st.warning(
                    "Aucun mot-clé à afficher avec les filtres actuels.")

    def display_search_keywords(self, keywords, remaining_keywords):
        """Afficher la section pour chercher des mots-clés pour une thématique avec un DataFrame."""
        st.subheader(t("search_keywords"))
        if not self.themes:
            st.warning("Aucune thématique disponible.")
        else:
            selected_theme = st.selectbox(
                t("select_theme"),
                options=list(self.themes.keys()),
                key=f"{self.prefix}_search_theme"
            )
            st.write(
                f"**{t('current_keywords')}**: {', '.join(self.themes[selected_theme]) if self.themes[selected_theme] else 'Aucun'}")
            show_all_keywords_search = st.checkbox(
                t("show_all_keywords"),
                value=False,
                key=f"{self.prefix}_show_all_keywords_search"
            )
            keywords_to_search = list(keywords.keys()) if show_all_keywords_search else list(
                remaining_keywords.keys())
            if st.button(t("suggest_keywords"), key=f"{self.prefix}_search_keywords_button"):
                with st.spinner(t("results_title")):
                    suggested_keywords = self.suggest_keywords(
                        selected_theme, keywords_to_search)
                    if suggested_keywords:
                        df = pd.DataFrame({"Keyword": suggested_keywords})
                        st.session_state[f"{self.prefix}_searched_keywords"] = df
                    else:
                        st.warning("Aucun mot-clé suggéré.")
            if f"{self.prefix}_searched_keywords" in st.session_state and not st.session_state[f"{self.prefix}_searched_keywords"].empty:
                selected = st.dataframe(
                    st.session_state[f"{self.prefix}_searched_keywords"],
                    selection_mode="multi-row",
                    on_select="rerun",
                    key=f"{self.prefix}_searched_keywords_dataframe"
                )
                if st.button(t("add_selected_keywords"), key=f"{self.prefix}_add_searched_keywords_button"):
                    selected_rows = selected["selection"]["rows"]
                    if selected_rows:
                        for row in selected_rows:
                            keyword = st.session_state[f"{self.prefix}_searched_keywords"].loc[row, "Keyword"]
                            if keyword not in self.themes[selected_theme]:
                                self.themes[selected_theme].append(keyword)
                        st.session_state[f"{self.prefix}_themes_text"] = self.themes_to_text(
                            self.themes)
                        st.rerun()

    def display(self):
        """Afficher l'interface Streamlit en utilisant des sous-fonctions."""
        st.title(t("title"))

        # Gestion des thématiques
        self.display_themes_management()

        # Chargement des mots-clés
        keywords = self.display_keywords_loading()
        if not keywords:
            return

        # Calcul des statistiques
        remaining_keywords, _, _ = self.get_remaining_keywords(
            keywords, self.themes)
        thematized_keywords = self.get_thematized_keywords(
            keywords, self.themes)

        # Afficher les statistiques
        self.display_keywords_stats(
            keywords, remaining_keywords, thematized_keywords)

        # Paramètres
        self.display_parameters()

        # Étape 1 : Suggérer des mots-clés pour une thématique
        self.display_suggest_keywords(keywords, remaining_keywords)

        # Étape 2 : Suggérer une catégorie pour un mot-clé
        self.display_suggest_category(remaining_keywords)

        # Étape 3 : Suggérer des catégories pour tous les mots-clés non assignés
        self.display_suggest_mass_categories(remaining_keywords)

        # Étape 4 : Assigner manuellement des mots-clés à une thématique
        self.display_assign_keywords(keywords, remaining_keywords)

        # Étape 5 : Chercher les mots-clés pour une thématique
        self.display_search_keywords(keywords, remaining_keywords)
