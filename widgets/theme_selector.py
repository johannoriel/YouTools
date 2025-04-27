# Nouveau fichier: widgets/theme_selector.py
from lib.global_vars import translations, t
from app import Widget
import streamlit as st

translations["en"].update({
    "themeselector_select_themes": "Select Themes to Search",
    "themeselector_select_all": "Select All Themes",
})

translations["fr"].update({
    "themeselector_select_themes": "Sélectionner les thèmes à rechercher",
    "themeselector_select_all": "Sélectionner tous les thèmes",
})

class ThemeSelectorWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def get_keywords_for_theme(self, theme):
        """Return the list of main keywords and synonyms for a given theme."""
        keywords_config = self.plugin_manager.config.get(self.name, {}).get(
            "trendwatcher_keywords", t("trendwatcher_keywords_default"))

        keywords = []
        for line in keywords_config.split("\n"):
            line = line.strip()
            if line:
                parts = line.split(":", 2)
                current_theme = parts[0].strip() if len(parts) >= 2 else "Default"
                if current_theme == theme:
                    main_keyword = parts[1].strip() if len(parts) >= 2 else ""
                    synonyms = [s.strip() for s in parts[2].split(",") if s.strip()] if len(parts) > 2 else []
                    if main_keyword:
                        keywords.append({"main": main_keyword, "synonyms": synonyms})

        return keywords

    def display(self):
        # Extraire les thèmes depuis la configuration
        keywords_config = self.plugin_manager.config.get(self.name, {}).get(
            "trendwatcher_keywords", t("trendwatcher_keywords_default"))

        # Extraire les thèmes uniques
        themes = []
        for line in keywords_config.split("\n"):
            line = line.strip()
            if line:
                parts = line.split(":", 2)
                theme = parts[0].strip() if len(parts) >= 2 else "Default"
                if theme not in themes:
                    themes.append(theme)

        # Initialiser l'état de la session
        if f"{self.prefix}_selected_themes" not in st.session_state:
            st.session_state[f"{self.prefix}_selected_themes"] = themes

        # Interface de sélection des thèmes
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_themes = st.multiselect(
                t("themeselector_select_themes"),
                themes,
                default=st.session_state[f"{self.prefix}_selected_themes"],
                key=f"{self.prefix}_theme_filter"
            )
        with col2:
            if st.button(t("themeselector_select_all"), key=f"{self.prefix}_select_all_themes"):
                selected_themes = themes
                st.session_state[f"{self.prefix}_selected_themes"] = themes
                st.rerun()

        # Mettre à jour l'état de la session
        st.session_state[f"{self.prefix}_selected_themes"] = selected_themes

        return selected_themes
