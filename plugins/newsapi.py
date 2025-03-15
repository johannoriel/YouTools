from global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
# Optionnel pour enrichir les résultats avec un LLM
from plugins.ragllm import RagllmPlugin
import os
import requests
from datetime import datetime, timedelta

# Ajout des traductions spécifiques au plugin NewsAPI
translations["en"].update({
    "newsapi_tab": "NewsAPI Trends",
    "newsapi_header": "Search News Trends with NewsAPI",
    "newsapi_api_key_label": "NewsAPI Key",
    "newsapi_api_key_default": "Enter your NewsAPI key",
    "newsapi_query_label": "Search Query",
    "newsapi_query_placeholder": "e.g., AI trends",
    "newsapi_endpoint_label": "Endpoint",
    "newsapi_language_label": "Language",
    "newsapi_country_label": "Country",
    "newsapi_sort_by_label": "Sort By",
    "newsapi_source_label": "Source",
    "newsapi_category_label": "Category",
    "newsapi_summarize_label": "Summarize Trends with LLM",
    "newsapi_search_button": "Search Trends",
    "newsapi_summarize_button": "Summarize Trends",
    "newsapi_processing": "Fetching news trends...",
    "newsapi_success": "Trends fetched successfully! Found {count} articles.",
    "newsapi_error": "An error occurred: {error}",
    "newsapi_no_results": "No results found for this query.",
    "newsapi_request_debug": "API Request Details",
})

translations["fr"].update({
    "newsapi_tab": "Tendances NewsAPI",
    "newsapi_header": "Rechercher les tendances avec NewsAPI",
    "newsapi_api_key_label": "Clé API NewsAPI",
    "newsapi_api_key_default": "Entrez votre clé NewsAPI",
    "newsapi_query_label": "Requête de recherche",
    "newsapi_query_placeholder": "ex. : tendances IA",
    "newsapi_endpoint_label": "Point de terminaison",
    "newsapi_language_label": "Langue",
    "newsapi_country_label": "Pays",
    "newsapi_sort_by_label": "Trier par",
    "newsapi_source_label": "Source",
    "newsapi_category_label": "Catégorie",
    "newsapi_summarize_label": "Résumer les tendances avec LLM",
    "newsapi_search_button": "Rechercher les tendances",
    "newsapi_summarize_button": "Résumer les tendances",
    "newsapi_processing": "Récupération des tendances en cours...",
    "newsapi_success": "Tendances récupérées avec succès ! {count} articles trouvés.",
    "newsapi_error": "Une erreur s'est produite : {error}",
    "newsapi_no_results": "Aucun résultat trouvé pour cette requête.",
    "newsapi_request_debug": "Détails de la requête API",
})


class NewsapiPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.ragllm_plugin = self.plugin_manager.get_plugin('ragllm')
        # Initialisation des sources dans session_state si elles n'existent pas encore
        if "newsapi_french_sources" not in st.session_state:
            st.session_state["newsapi_french_sources"] = self._fetch_french_sources(
            )

    def _fetch_french_sources(self):
        """Récupère dynamiquement les sources françaises via l'endpoint /sources."""
        api_key = self.plugin_manager.config.get(
            self.name, {}).get("newsapi_api_key", "")
        if not api_key or api_key == t("newsapi_api_key_default"):
            return [""]  # Retourne une liste vide par défaut si pas de clé

        url = "https://newsapi.org/v2/top-headlines/sources"
        params = {
            "apiKey": api_key,
            "country": "fr"  # Limite aux sources françaises
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data["status"] == "ok":
                return [""] + [source["id"] for source in data["sources"]]
        except Exception as e:
            st.error(f"Failed to fetch sources: {str(e)}")
        return [""]  # Retourne une liste vide en cas d'erreur

    def get_config_fields(self):
        """Définit les champs de configuration, dont la clé API."""
        return {
            "newsapi_api_key": {
                "type": "text",
                "label": t("newsapi_api_key_label"),
                "default": t("newsapi_api_key_default")
            },
            "newsapi_default_language": {
                "type": "select",
                "label": t("newsapi_language_label"),
                "options": [("en", "English"), ("fr", "Français"), ("es", "Español"), ("de", "Deutsch")],
                "default": "fr"
            }
        }

    def get_tabs(self):
        """Définit l'onglet pour NewsAPI dans l'interface."""
        return [{"name": t("newsapi_tab"), "plugin": "newsapiplugin"}]

    def run(self, config):
        """Logique principale du plugin NewsAPI."""
        st.header(t("newsapi_header"))

        # Récupération de la clé API depuis la configuration
        api_key = config.get(self.name, {}).get("newsapi_api_key", "")
        if not api_key or api_key == t("newsapi_api_key_default"):
            st.error("Please provide a valid NewsAPI key in the configuration.")
            return

        # Interface utilisateur pour les paramètres de recherche
        endpoint = st.selectbox(
            t("newsapi_endpoint_label"),
            options=["everything", "top-headlines"],
            index=0
        )

        query = st.text_input(
            t("newsapi_query_label"),
            value="",
            placeholder=t("newsapi_query_placeholder")
        )

        language = st.selectbox(
            t("newsapi_language_label"),
            options=["en", "fr", "es", "de"],
            index=["en", "fr", "es", "de"].index(config.get(
                self.name, {}).get("newsapi_default_language", "fr"))
        )

        # Paramètres spécifiques selon l'endpoint
        if endpoint == "everything":
            sort_by = st.selectbox(
                t("newsapi_sort_by_label"),
                options=["relevancy", "popularity", "publishedAt"],
                index=0
            )
            country = None
            source = None
            category = None
        else:  # top-headlines
            sort_by = None
            country = st.selectbox(
                t("newsapi_country_label"),
                options=["", "fr", "us", "gb", "de"],
                index=1,  # France par défaut
                format_func=lambda x: "All Countries" if x == "" else {
                    "fr": "France", "us": "USA", "gb": "UK", "de": "Germany"}.get(x, x)
            )
            source = st.selectbox(
                t("newsapi_source_label"),
                # Sources depuis session_state
                options=st.session_state["newsapi_french_sources"],
                index=0,
                format_func=lambda x: "All Sources" if x == "" else x
            )
            category = st.selectbox(
                t("newsapi_category_label"),
                options=["", "business", "entertainment", "general",
                         "health", "science", "sports", "technology"],
                index=0,
                format_func=lambda x: "All Categories" if x == "" else x
            )

            if source and country:
                st.warning(
                    "Note: Selecting a specific source overrides the country filter in NewsAPI.")

        # Option pour résumer avec LLM
        summarize_with_llm = st.checkbox(
            t("newsapi_summarize_label"), value=False) if self.ragllm_plugin else False

        # Bouton pour lancer la recherche
        if st.button(t("newsapi_search_button")):
            with st.spinner(t("newsapi_processing")):
                try:
                    # Préparation de la requête à l'API NewsAPI
                    url = f"https://newsapi.org/v2/{endpoint}"
                    params = {
                        "apiKey": api_key,
                        "language": language,
                        "pageSize": 10,
                    }

                    if endpoint == "everything":
                        params["q"] = remove_quotes(query) if query else "*"
                        params["sortBy"] = sort_by
                        params["from"] = (
                            datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                    else:  # top-headlines
                        if query:
                            params["q"] = remove_quotes(query)
                        if source:
                            params["sources"] = source
                        elif country:
                            params["country"] = country
                        if category:
                            params["category"] = category

                    # Affichage de la requête dans un expander collapsé
                    with st.expander(t("newsapi_request_debug"), expanded=False):
                        st.write(f"**URL**: {url}")
                        st.write(f"**Params**: {params}")

                    # Appel à l'API
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()

                    # Vérification des résultats
                    if data["status"] == "ok" and data["totalResults"] > 0:
                        articles = data["articles"]
                        st.success(t("newsapi_success").format(
                            count=len(articles)))

                        # Affichage des résultats
                        for article in articles:
                            st.subheader(article["title"])
                            st.write(f"Source: {article['source']['name']}")
                            st.write(f"Published: {article['publishedAt']}")
                            st.write(article["description"])
                            st.markdown(f"[Read more]({article['url']})")
                            st.write("---")

                        # Résumé LLM si activé dès le départ
                        if summarize_with_llm and self.ragllm_plugin:
                            llm_prompt = "Summarize the key trends from these news articles."
                            llm_sys_prompt = config['ragllm']['llm_sys_prompt']
                            article_texts = "\n".join(
                                [a["description"] or "" for a in articles])
                            llm_response = self.ragllm_plugin.process_with_llm(
                                llm_prompt,
                                llm_sys_prompt,
                                article_texts
                            )
                            st.write("**LLM Trend Summary:**")
                            st.write(llm_response)
                        elif self.ragllm_plugin:
                            if st.button(t("newsapi_summarize_button")):
                                llm_prompt = "Summarize the key trends from these news articles."
                                llm_sys_prompt = config['ragllm']['llm_sys_prompt']
                                article_texts = "\n".join(
                                    [a["description"] or "" for a in articles])
                                llm_response = self.ragllm_plugin.process_with_llm(
                                    llm_prompt,
                                    llm_sys_prompt,
                                    article_texts
                                )
                                st.write("**LLM Trend Summary:**")
                                st.write(llm_response)

                    else:
                        st.warning(t("newsapi_no_results"))

                except Exception as e:
                    st.error(t("newsapi_error").format(error=str(e)))


if __name__ == "__main__":
    st.write("NewsAPI Plugin standalone test")
