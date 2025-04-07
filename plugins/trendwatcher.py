from global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
from duckduckgo_search import DDGS
import random
import time
from datetime import datetime, timedelta
from langdetect import detect
import pandas as pd

# Translations
translations["en"].update({
    "trendwatcher_tab": "Trend Watcher",
    "trendwatcher_header": "Trend Monitoring Dashboard",
    "trendwatcher_keywords_label": "Keywords to Monitor (one per line)",
    "trendwatcher_keywords_default": "AI\nblockchain\nclimate change",
    "trendwatcher_search_button": "Start Monitoring",
    "trendwatcher_save_keywords_button": "Save Keywords to Config",
    "trendwatcher_processing": "Searching for recent content...",
    "trendwatcher_useragents_label": "User Agents (one per line)",
    "trendwatcher_useragents_default": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36\nMozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "trendwatcher_delay_min": "Minimum Delay (seconds)",
    "trendwatcher_delay_max": "Maximum Delay (seconds)",
    "trendwatcher_results": "Results for '{keyword}':",
    "trendwatcher_no_results": "No recent results found within 7 days",
    "trendwatcher_error": "Error during search: {error}",
    "trendwatcher_debug": "Debug Mode",
    "trendwatcher_debug_query": "Querying DDG with: {query}",
    "trendwatcher_debug_results": "Found {count} results ({vids} videos, {texts} texts)",
    "trendwatcher_debug_date": "Parsing date: {date_str} -> {result}",
    "trendwatcher_language_filter": "Filter by Language",
    "trendwatcher_keyword_filter": "Filter by Keyword",
    "trendwatcher_table_title": "Title (Link)",
    "trendwatcher_table_views": "Views",
    "trendwatcher_table_days": "Days Old",
    "trendwatcher_table_type": "Type",
    "trendwatcher_table_language": "Language",
    "trendwatcher_table_keyword": "Keyword",
    "trendwatcher_videos_table": "Videos",
    "trendwatcher_texts_table": "Text Articles"
})

translations["fr"].update({
    "trendwatcher_tab": "Observateur de Tendances",
    "trendwatcher_header": "Tableau de Bord de Suivi des Tendances",
    "trendwatcher_keywords_label": "Mots-clés à surveiller (un par ligne)",
    "trendwatcher_keywords_default": "IA\nblockchain\nchangement climatique",
    "trendwatcher_search_button": "Lancer la surveillance",
    "trendwatcher_save_keywords_button": "Sauvegarder les mots-clés dans la config",
    "trendwatcher_processing": "Recherche de contenu récent...",
    "trendwatcher_useragents_label": "Agents Utilisateurs (un par ligne)",
    "trendwatcher_useragents_default": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36\nMozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "trendwatcher_delay_min": "Délai Minimum (secondes)",
    "trendwatcher_delay_max": "Délai Maximum (secondes)",
    "trendwatcher_results": "Résultats pour '{keyword}':",
    "trendwatcher_no_results": "Aucun résultat récent trouvé dans les 7 derniers jours",
    "trendwatcher_error": "Erreur pendant la recherche : {error}",
    "trendwatcher_debug": "Mode Débogage",
    "trendwatcher_debug_query": "Requête DDG avec : {query}",
    "trendwatcher_debug_results": "Trouvé {count} résultats ({vids} vidéos, {texts} textes)",
    "trendwatcher_debug_date": "Analyse de la date : {date_str} -> {result}",
    "trendwatcher_language_filter": "Filtrer par langue",
    "trendwatcher_keyword_filter": "Filtrer par mot-clé",
    "trendwatcher_table_title": "Titre (Lien)",
    "trendwatcher_table_views": "Vues",
    "trendwatcher_table_days": "Jours d'ancienneté",
    "trendwatcher_table_type": "Type",
    "trendwatcher_table_language": "Langue",
    "trendwatcher_table_keyword": "Mot-clé",
    "trendwatcher_videos_table": "Vidéos",
    "trendwatcher_texts_table": "Articles Textes"
})

class TrendwatcherPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)

    def get_config_fields(self):
        """Define configuration fields"""
        return {
            "trendwatcher_keywords": {
                "type": "textarea",
                "label": t("trendwatcher_keywords_label"),
                "default": t("trendwatcher_keywords_default")
            },
            "trendwatcher_useragents": {
                "type": "textarea",
                "label": t("trendwatcher_useragents_label"),
                "default": t("trendwatcher_useragents_default")
            },
            "trendwatcher_delay_min": {
                "type": "number",
                "label": t("trendwatcher_delay_min"),
                "default": 1
            },
            "trendwatcher_delay_max": {
                "type": "number",
                "label": t("trendwatcher_delay_max"),
                "default": 3
            }
        }

    def get_tabs(self):
        """Define plugin tabs"""
        return [{"name": t("trendwatcher_tab"), "plugin": "trendwatcher"}]

    def parse_date(self, date_str, debug=False):
        """Try parsing date in multiple formats"""
        date_formats = [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.0000000"
        ]

        if "+" in date_str or "-" in date_str[-6:]:
            date_str = date_str.rsplit("+", 1)[0].rsplit("-", 1)[0]

        for fmt in date_formats:
            try:
                result = datetime.strptime(date_str, fmt)
                #if debug:
                #    st.write(t("trendwatcher_debug_date").format(date_str=date_str, result=result))
                return result
            except ValueError:
                continue

        if debug:
            st.write(t("trendwatcher_debug_date").format(date_str=date_str, result="Failed to parse"))
        return None

    def search_trends(self, keyword, useragents, debug=False):
        """Search for recent videos and web content"""
        query = f"{keyword} site:youtube.com OR -inurl:(signup login)"
        if debug:
            st.write(t("trendwatcher_debug_query").format(query=query))

        # Create new DDGS instance with random User-Agent
        headers = {"User-Agent": random.choice(useragents)}
        ddgs = DDGS(headers=headers)

        try:
            video_results = ddgs.videos(
                keywords=keyword,
                region="fr-fr",
                timelimit="w",
                max_results=5
            )

            text_results = ddgs.text(
                keywords=keyword,
                region="fr-fr",
                timelimit="w",
                max_results=5
            )

            results = []
            cutoff_date = datetime.now() - timedelta(days=7)

            # Process video results
            for video in video_results:
                published_date = self.parse_date(video["published"], debug=debug)
                if published_date and published_date > cutoff_date:
                    days_old = (datetime.now() - published_date).days
                    title_link = f"[{video['title'].replace('|','')}]({video['content']})"
                    language = detect(video["title"]) if video["title"] else "unknown"
                    results.append({
                        "title_link": title_link,
                        "views": video["statistics"].get("viewCount", "N/A"),
                        "days_old": days_old,
                        "type": "video",
                        "language": language,
                        "keyword": keyword
                    })

            # Process text results
            for text in text_results:
                title_link = f"[{text['title'].replace('|','')}]({text['href']})"
                language = detect(text["title"]) if text["title"] else "unknown"
                if debug:
                    st.write(f"Text article: {title_link}")
                results.append({
                    "title_link": title_link,
                    "views": "N/A",
                    "days_old": "N/A",
                    "type": "web",
                    "language": language,
                    "keyword": keyword
                })

            if debug:
                st.write(t("trendwatcher_debug_results").format(
                    count=len(results),
                    vids=sum(1 for r in results if r["type"] == "video"),
                    texts=sum(1 for r in results if r["type"] == "web")
                ))

            return results  # Return all results

        except Exception as e:
            return str(e)

    def run(self, config):
        """Main plugin logic"""
        st.header(t("trendwatcher_header"))

        keywords_config = config.get(self.name, {}).get("trendwatcher_keywords", t("trendwatcher_keywords_default"))
        useragents = config.get(self.name, {}).get("trendwatcher_useragents", t("trendwatcher_useragents_default")).split("\n")
        delay_min = float(config.get(self.name, {}).get("trendwatcher_delay_min", 1))
        delay_max = float(config.get(self.name, {}).get("trendwatcher_delay_max", 3))

        keywords_input = st.text_area(
            t("trendwatcher_keywords_label"),
            value=keywords_config,
            height=150
        )

        if st.button(t("trendwatcher_save_keywords_button")):
            if self.plugin_manager:
                self.plugin_manager.config[self.name]["trendwatcher_keywords"] = keywords_input
                self.plugin_manager.save_config(config)
                st.success("Keywords saved to configuration!")
            else:
                st.warning("Plugin manager not available, cannot save config.")

        debug_mode = st.checkbox(t("trendwatcher_debug"), value=False)

        # Initialize session state
        if "trendwatcher_results" not in st.session_state:
            st.session_state.trendwatcher_results = []

        if st.button(t("trendwatcher_search_button")):
            with st.spinner(t("trendwatcher_processing")):
                keywords = [k.strip() for k in keywords_input.split("\n") if k.strip()]
                all_results = []
                for keyword in keywords:
                    results = self.search_trends(keyword, useragents, debug=debug_mode)
                    if isinstance(results, list):
                        all_results.extend(results)
                    else:
                        st.error(t("trendwatcher_error").format(error=results))
                    time.sleep(random.uniform(delay_min, delay_max))

                st.session_state.trendwatcher_results = all_results

        if st.session_state.trendwatcher_results:
            # Split into videos and texts
            video_results = [r for r in st.session_state.trendwatcher_results if r["type"] == "video"]
            text_results = [r for r in st.session_state.trendwatcher_results if r["type"] == "web"]

            # Filters
            all_keywords = list(set(r["keyword"] for r in st.session_state.trendwatcher_results))
            selected_keywords = st.multiselect(
                t("trendwatcher_keyword_filter"),
                all_keywords,
                default=all_keywords
            )

            all_languages = list(set(r["language"] for r in st.session_state.trendwatcher_results))
            selected_language = st.selectbox(
                t("trendwatcher_language_filter"),
                ["All"] + all_languages,
                index=0
            )

            # Filter and create DataFrames
            def filter_df(results):
                df = pd.DataFrame(results)
                if selected_keywords:
                    df = df[df["keyword"].isin(selected_keywords)]
                if selected_language != "All":
                    df = df[df["language"] == selected_language]
                return df.rename(columns={
                    "title_link": t("trendwatcher_table_title"),
                    "views": t("trendwatcher_table_views"),
                    "days_old": t("trendwatcher_table_days"),
                    "type": t("trendwatcher_table_type"),
                    "language": t("trendwatcher_table_language"),
                    "keyword": t("trendwatcher_table_keyword")
                })

            # Videos table
            if video_results:
                st.subheader(t("trendwatcher_videos_table"))
                video_df = filter_df(video_results)
                if not video_df.empty:
                    st.markdown(video_df.to_markdown(index=False), unsafe_allow_html=True)
                else:
                    st.info("No videos match the filters.")

            # Texts table
            if text_results:
                st.subheader(t("trendwatcher_texts_table"))
                text_df = filter_df(text_results)
                if not text_df.empty:
                    st.markdown(text_df.to_markdown(index=False), unsafe_allow_html=True)
                else:
                    st.info("No text articles match the filters.")

            if not video_results and not text_results:
                st.info(t("trendwatcher_no_results"))
        else:
            st.info(t("trendwatcher_no_results"))

if __name__ == "__main__":
    st.write("Trendwatcher Plugin standalone test")
