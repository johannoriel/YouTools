from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
import random
import time
from datetime import datetime, timedelta
from langdetect import detect
import pandas as pd
import os
import re
import csv
from lib.youtube_api import YoutubeAPI
import yt_dlp
from lib.search_engines import (
    search_videos_duckduckgo,
    search_texts_duckduckgo,
    search_videos_google,
    search_texts_google,
    search_videos_ytdlp,
    search_texts_ytdlp,
    search_videos_searxng,
    search_texts_searxng,
    search_videos_bing,
    search_texts_bing,
    parse_date,
    extract_youtube_id
)

# Translations
translations["en"].update({
    "trendwatcher_tab": "Trend Watcher",
    "trendwatcher_header": "Trend Monitoring Dashboard (searched_video_list.csv)",
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
    "trendwatcher_texts_table": "Text Articles",
    "trendwatcher_working_dir": "Working Directory",
    "trendwatcher_working_dir_default": "~/Videos",
    "trendwatcher_save_success": "Results saved to {dir}",
    "trendwatcher_save_button": "Save Results",
    "trendwatcher_table_view_count": "View Count",
    "trendwatcher_table_relevance_score": "Relevance Score",
    "trendwatcher_select_all_videos": "Select/Deselect All Videos",
    "trendwatcher_select_all_texts": "Select/Deselect All Articles",
    "trendwatcher_search_mode_label": "Search Mode",
    "trendwatcher_search_mode_or": "OR (Combine keywords with OR)",
    "trendwatcher_search_mode_subsearches": "Subsearches (Separate search for each keyword)",
    "trendwatcher_search_engine_label": "Search Engine",
    "trendwatcher_search_engine_duckduckgo": "DuckDuckGo",
    "trendwatcher_search_engine_brave": "Brave Search",
    "trendwatcher_brave_api_key_label": "Brave Search API Key",
    "trendwatcher_brave_api_key_help": "Enter your Brave Search API key (get it from api.brave.com)",
    "trendwatcher_search_engine_google": "Google Custom Search",
    "trendwatcher_search_engine_ytdlp": "yt-dlp",
    "trendwatcher_google_api_key_label": "Google API Key",
    "trendwatcher_google_api_key_help": "Enter your Google Custom Search API key (get it from console.cloud.google.com)",
    "trendwatcher_google_cx_id_label": "Google Custom Search Engine ID",
    "trendwatcher_google_cx_id_help": "Enter your Google Custom Search Engine ID (CX ID)",
    "trendwatcher_search_engine_searxng": "SearxNG",
    "trendwatcher_searxng_server_url_label": "SearxNG Server URL",
    "trendwatcher_searxng_server_url_help": "Enter the URL of your SearxNG instance (e.g., https://search.example.com)",
    "trendwatcher_search_engine_bing": "Bing Web Search",
    "trendwatcher_bing_api_key_label": "Bing API Key",
    "trendwatcher_bing_api_key_help": "Enter your Bing Web Search API key (get it from portal.azure.com)",
})

translations["fr"].update({
    "trendwatcher_tab": "Observateur de Tendances",
    "trendwatcher_header": "Tableau de Bord de Suivi des Tendances (searched_video_list.csv)",
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
    "trendwatcher_texts_table": "Articles Textes",
    "trendwatcher_working_dir": "Répertoire de travail",
    "trendwatcher_working_dir_default": "~/Vidéos",
    "trendwatcher_save_success": "Résultats sauvegardés dans {dir}",
    "trendwatcher_save_button": "Sauvegarder les résultats",
    "trendwatcher_table_view_count": "Nombre de vues",
    "trendwatcher_table_relevance_score": "Score de pertinence",
    "trendwatcher_select_all_videos": "Tout sélectionner/désélectionner les vidéos",
    "trendwatcher_select_all_texts": "Tout sélectionner/désélectionner les articles",
    "trendwatcher_search_mode_label": "Mode de recherche",
    "trendwatcher_search_mode_or": "OU (Combiner les mots-clés avec OU)",
    "trendwatcher_search_mode_subsearches": "Sous-recherches (Recherche séparée pour chaque mot-clé)",
    "trendwatcher_search_engine_label": "Moteur de recherche",
    "trendwatcher_search_engine_duckduckgo": "DuckDuckGo",
    "trendwatcher_search_engine_brave": "Brave Search",
    "trendwatcher_brave_api_key_label": "Clé API Brave Search",
    "trendwatcher_brave_api_key_help": "Entrez votre clé API Brave Search (obtenez-la sur api.brave.com)",
    "trendwatcher_search_engine_google": "Google Custom Search",
    "trendwatcher_search_engine_ytdlp": "yt-dlp",
    "trendwatcher_google_api_key_label": "Clé API Google",
    "trendwatcher_google_api_key_help": "Entrez votre clé API Google Custom Search (obtenez-la sur console.cloud.google.com)",
    "trendwatcher_google_cx_id_label": "ID du moteur de recherche personnalisé Google",
    "trendwatcher_google_cx_id_help": "Entrez votre ID de moteur de recherche personnalisé Google (CX ID)",
    "trendwatcher_search_engine_searxng": "SearxNG",
    "trendwatcher_searxng_server_url_label": "URL du Serveur SearxNG",
    "trendwatcher_searxng_server_url_help": "Entrez l'URL de votre instance SearxNG (par exemple, https://search.example.com)",
    "trendwatcher_search_engine_bing": "Recherche Web Bing",
    "trendwatcher_bing_api_key_label": "Clé API Bing",
    "trendwatcher_bing_api_key_help": "Entrez votre clé API Bing Web Search (obtenez-la sur portal.azure.com)",
})

# Official CSV headers
VIDEO_CSV_HEADERS = [
    "keyword", "url", "video_id", "title", "view_count", "language", "published_at",
    "channel_id", "channel_title", "subscriber_count", "comment_count", "relevance_score"
]


class TrendwatcherPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        # Define search engines and their methods
        self.SEARCH_ENGINES = {
            "duckduckgo": {
                "name": t("trendwatcher_search_engine_duckduckgo"),
                "search_videos": search_videos_duckduckgo,
                "search_texts": search_texts_duckduckgo,
                "config": {}
            },
            "google": {
                "name": t("trendwatcher_search_engine_google"),
                "search_videos": lambda *args, **kwargs: search_videos_google(
                    *args,
                    api_key=self.plugin_manager.config.get(
                        self.name, {}).get("trendwatcher_google_api_key", ""),
                    cx_id=self.plugin_manager.config.get(
                        self.name, {}).get("trendwatcher_google_cx_id", ""),
                    **kwargs
                ),
                "search_texts": lambda *args, **kwargs: search_texts_google(
                    *args,
                    api_key=self.plugin_manager.config.get(
                        self.name, {}).get("trendwatcher_google_api_key", ""),
                    cx_id=self.plugin_manager.config.get(
                        self.name, {}).get("trendwatcher_google_cx_id", ""),
                    **kwargs
                ),
                "config": {
                    "api_key": lambda: self.plugin_manager.config.get(self.name, {}).get("trendwatcher_google_api_key", ""),
                    "cx_id": lambda: self.plugin_manager.config.get(self.name, {}).get("trendwatcher_google_cx_id", "")
                }
            },
            "ytdlp": {
                "name": t("trendwatcher_search_engine_ytdlp"),
                "search_videos": search_videos_ytdlp,
                "search_texts": search_texts_ytdlp,
                "config": {}
            },
            "searxng": {
                "name": t("trendwatcher_search_engine_searxng"),
                "search_videos": lambda *args, **kwargs: search_videos_searxng(
                    *args,
                    server_url=self.plugin_manager.config.get(self.name, {}).get(
                        "trendwatcher_searxng_server_url", ""),
                    **kwargs
                ),
                "search_texts": lambda *args, **kwargs: search_texts_searxng(
                    *args,
                    server_url=self.plugin_manager.config.get(self.name, {}).get(
                        "trendwatcher_searxng_server_url", ""),
                    **kwargs
                ),
                "config": {
                    "server_url": lambda: self.plugin_manager.config.get(self.name, {}).get("trendwatcher_searxng_server_url", "")
                }
            },
            "bing": {
                "name": t("trendwatcher_search_engine_bing"),
                "search_videos": lambda *args, **kwargs: search_videos_bing(
                    *args,
                    api_key=self.plugin_manager.config.get(
                        self.name, {}).get("trendwatcher_bing_api_key", ""),
                    **kwargs
                ),
                "search_texts": lambda *args, **kwargs: search_texts_bing(
                    *args,
                    api_key=self.plugin_manager.config.get(
                        self.name, {}).get("trendwatcher_bing_api_key", ""),
                    **kwargs
                ),
                "config": {
                    "api_key": lambda: self.plugin_manager.config.get(self.name, {}).get("trendwatcher_bing_api_key", "")
                }
            }
        }

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
            },
            "trendwatcher_working_dir": {
                "type": "text",
                "label": t("trendwatcher_working_dir"),
                "default": t("trendwatcher_working_dir_default")
            },
            "trendwatcher_brave_api_key": {
                "type": "text",
                "label": t("trendwatcher_brave_api_key_label"),
                "default": "",
                "help": t("trendwatcher_brave_api_key_help")
            },
            "trendwatcher_google_api_key": {
                "type": "text",
                "label": t("trendwatcher_google_api_key_label"),
                "default": "",
                "help": t("trendwatcher_google_api_key_help")
            },
            "trendwatcher_google_cx_id": {
                "type": "text",
                "label": t("trendwatcher_google_cx_id_label"),
                "default": "",
                "help": t("trendwatcher_google_cx_id_help")
            },
            "trendwatcher_searxng_server_url": {
                "type": "text",
                "label": t("trendwatcher_searxng_server_url_label"),
                "default": "",
                "help": t("trendwatcher_searxng_server_url_help")
            },
            "trendwatcher_bing_api_key": {
                "type": "text",
                "label": t("trendwatcher_bing_api_key_label"),
                "default": "",
                "help": t("trendwatcher_bing_api_key_help")
            }
        }

    def get_tabs(self):
        """Define plugin tabs"""
        return [
            {"name": t("trendwatcher_tab"), "plugin": "trendwatcher"},
            {"name": "Process Videos", "plugin": "trendwatcher"},
            {"name": "Keyword cluster", "plugin": "trendwatcher"},
            {"name": "Video List", "plugin": "trendwatcher"},
        ]

    def extract_channel_info_yt_dlp(self, video_url, debug=False):
        """Extract channel ID, title, and subscriber count using yt-dlp"""
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "force_generic_extractor": False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

                channel_id = info.get("channel_id", "N/A")
                channel_title = info.get("channel", "N/A")
                subscriber_count = info.get("channel_follower_count", "N/A")

                if isinstance(subscriber_count, int):
                    subscriber_count = f"{subscriber_count:,}".replace(
                        ",", " ") + " subscribers"

                if debug:
                    st.write(
                        f"Extracted from {video_url}: channel_id={channel_id}, channel_title={channel_title}, subscriber_count={subscriber_count}")

                return {
                    "channel_id": channel_id,
                    "channel_title": channel_title,
                    "subscriber_count": subscriber_count
                }

        except Exception as e:
            if debug:
                st.write(
                    f"Error extracting channel info with yt-dlp for {video_url}: {str(e)}")
            return {
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "N/A"
            }

    def extract_video_metadata_yt_dlp(self, video_url, debug=False):
        """Extract video metadata including channel info, view count, comment count, and published date using yt-dlp"""
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "force_generic_extractor": False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

                channel_id = info.get("channel_id", "N/A")
                channel_title = info.get("channel", "N/A")
                subscriber_count = info.get("channel_follower_count", "N/A")
                view_count = info.get("view_count", "N/A")
                comment_count = info.get("comment_count", "N/A")
                published_at = info.get("upload_date", "N/A")

                if isinstance(subscriber_count, int):
                    subscriber_count = str(subscriber_count)
                else:
                    subscriber_count = ""

                if isinstance(view_count, int):
                    view_count = str(view_count)
                else:
                    view_count = ""

                if isinstance(comment_count, int):
                    comment_count = str(comment_count)
                else:
                    comment_count = ""

                if published_at != "N/A":
                    try:
                        published_at = f"{published_at[:4]}-{published_at[4:6]}-{published_at[6:8]}"
                    except:
                        published_at = ""

                if debug:
                    st.write(f"Extracted from {video_url}: channel_id={channel_id}, channel_title={channel_title}, "
                             f"subscriber_count={subscriber_count}, view_count={view_count}, "
                             f"comment_count={comment_count}, published_at={published_at}")

                return {
                    "channel_id": channel_id,
                    "channel_title": channel_title,
                    "subscriber_count": subscriber_count,
                    "view_count": view_count,
                    "comment_count": comment_count,
                    "published_at": published_at
                }

        except Exception as e:
            if debug:
                st.write(
                    f"Error extracting video metadata with yt-dlp for {video_url}: {str(e)}")
            return {
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "view_count": "",
                "comment_count": "",
                "published_at": ""
            }

    # Modifier la méthode save_videos_to_csv
    def save_videos_to_csv(self, working_dir, selected_keywords, selected_urls, overwrite=True):
        """Save selected video results to CSV file"""
        working_dir = os.path.expanduser(working_dir)
        os.makedirs(working_dir, exist_ok=True)

        base_filename = "searched_video_list.csv"
        video_file = os.path.join(working_dir, base_filename)

        if not overwrite and os.path.exists(video_file):
            i = 1
            while True:
                new_filename = f"searched_video_list_{i:03d}.csv"
                new_file = os.path.join(working_dir, new_filename)
                if not os.path.exists(new_file):
                    video_file = new_file
                    break
                i += 1

        headers = ["keyword", "url", "video_id", "title", "language"]

        filtered_results = [
            r for r in st.session_state.trendwatcher_results
            if r["keyword"] in selected_keywords and r["url"] in selected_urls and r["type"] == "video"
        ]

        with open(video_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for video in filtered_results:
                writer.writerow([
                    video["keyword"],
                    video["url"],
                    video["video_id"],
                    video["title"],
                    video["language"]
                ])

        return video_file

    def save_articles_to_csv(self, working_dir, selected_keywords, selected_urls):
        """Save selected article results to CSV file"""
        working_dir = os.path.expanduser(working_dir)
        os.makedirs(working_dir, exist_ok=True)

        article_file = os.path.join(working_dir, "article_list.csv")
        headers = ["keyword", "url", "title", "views", "language", "date"]

        filtered_results = [
            r for r in st.session_state.trendwatcher_results
            if r["keyword"] in selected_keywords and r["url"] in selected_urls and r["type"] == "web"
        ]

        with open(article_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for article in filtered_results:
                writer.writerow([
                    article["keyword"],
                    article["url"],
                    article["title"],
                    article["view_count"],
                    article["language"],
                    article["published_at"]
                ])

        return working_dir

    def search_trends(self, main_keyword, synonyms, useragents, search_mode="or", search_engine="duckduckgo", debug=False):
        """Search for recent videos and web content"""
        engine = self.SEARCH_ENGINES.get(
            search_engine, self.SEARCH_ENGINES["duckduckgo"])
        search_videos = engine["search_videos"]
        search_texts = engine["search_texts"]

        if search_mode == "or":
            query_terms = [main_keyword] + synonyms
            query = " OR ".join(f'"{term}"' for term in query_terms if term)
            if search_engine == "duckduckgo":
                query += " site:youtube.com OR -inurl:(signup login)"
            elif search_engine == "google":
                query += " site:youtube.com"
            elif search_engine == "searxng":
                query += " site:youtube.com"
            elif search_engine == "bing":
                query += " site:youtube.com"
            if debug:
                st.write(t("trendwatcher_debug_query").format(query=query))
            queries = [(query, main_keyword)]
        else:
            queries = [(f'"{main_keyword}"', main_keyword)]
            queries.extend((f'"{syn}"', main_keyword) for syn in synonyms)
            if search_engine == "duckduckgo":
                queries = [(f"{q} site:youtube.com OR -inurl:(signup login)", k)
                           for q, k in queries]
            elif search_engine == "google":
                queries = [(f"{q} site:youtube.com", k) for q, k in queries]
            elif search_engine == "searxng":
                queries = [(f"{q} site:youtube.com", k) for q, k in queries]
            elif search_engine == "bing":
                queries = [(f"{q} site:youtube.com", k) for q, k in queries]
            if debug:
                st.write(t("trendwatcher_debug_query").format(
                    query=", ".join(q for q, _ in queries)))

        valid_video_domains = [
            "youtube.com",
            "youtu.be",
            "dailymotion.com",
            "vimeo.com",
            "bilibili.com",
            "twitch.tv",
            "tiktok.com"
        ]

        try:
            all_results = []
            for query, keyword in queries:
                video_results = search_videos(
                    query, keyword, useragents, valid_video_domains, debug=debug)
                text_results = search_texts(
                    query, keyword, useragents, debug=debug)
                all_results.extend(video_results + text_results)

            if debug:
                video_count = sum(
                    1 for r in all_results if r.get("type") == "video")
                text_count = sum(
                    1 for r in all_results if r.get("type") == "web")
                st.write(t("trendwatcher_debug_results").format(
                    count=len(all_results),
                    vids=video_count,
                    texts=text_count
                ))

            return all_results
        except Exception as e:
            return str(e)

    def trend_watcher(self, config):
        st.header(t("trendwatcher_header"))

        keywords_config = config.get(self.name, {}).get(
            "trendwatcher_keywords", t("trendwatcher_keywords_default"))
        useragents = config.get(self.name, {}).get(
            "trendwatcher_useragents", t("trendwatcher_useragents_default")).split("\n")
        delay_min = float(config.get(self.name, {}).get(
            "trendwatcher_delay_min", 1))
        delay_max = float(config.get(self.name, {}).get(
            "trendwatcher_delay_max", 3))
        working_dir = config.get(self.name, {}).get(
            "working_dir", t("trendwatcher_working_dir_default"))

        keywords_input = st.text_area(
            t("trendwatcher_keywords_label"),
            value=keywords_config,
            height=150,
            help="Format: main_keyword:synonym1,synonym2,... (one per line)"
        )

        if st.button(t("trendwatcher_save_keywords_button"), key="save_keywords"):
            if self.plugin_manager:
                self.plugin_manager.config[self.name]["trendwatcher_keywords"] = keywords_input
                self.plugin_manager.save_config(config)
                st.success("Keywords saved to configuration!")
            else:
                st.warning("Plugin manager not available, cannot save config.")

        debug_mode = st.checkbox(t("trendwatcher_debug"), value=False)

        max_keywords_debug = 3
        if debug_mode:
            max_keywords_debug = st.number_input(
                "Maximum Keywords to Search in Debug Mode",
                min_value=1,
                max_value=100,
                value=3,
                step=1
            )

        search_engine_options = {
            engine_id: engine["name"] for engine_id, engine in self.SEARCH_ENGINES.items()}
        search_engine_name = st.selectbox(
            t("trendwatcher_search_engine_label"),
            list(search_engine_options.values()),
            index=0,
            key="search_engine"
        )
        search_engine = next(
            k for k, v in search_engine_options.items() if v == search_engine_name)

        search_mode = st.selectbox(
            t("trendwatcher_search_mode_label"),
            [t("trendwatcher_search_mode_or"), t(
                "trendwatcher_search_mode_subsearches")],
            index=0,
            key="search_mode"
        )
        search_mode_value = "or" if search_mode == t(
            "trendwatcher_search_mode_or") else "subsearches"

        overwrite = st.checkbox(t("overwrite_checkbox"), value=True)

        if "trendwatcher_results" not in st.session_state:
            st.session_state.trendwatcher_results = []
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = debug_mode
        if "select_all_videos" not in st.session_state:
            st.session_state.select_all_videos = False

        if st.button(t("trendwatcher_search_button"), key="search_trends"):
            with st.spinner(t("trendwatcher_processing")):
                keyword_configs = []
                for line in keywords_input.split("\n"):
                    line = line.strip()
                    if line:
                        if ":" in line:
                            main_keyword, synonyms = line.split(":", 1)
                            main_keyword = main_keyword.strip()
                            synonyms = [s.strip()
                                        for s in synonyms.split(",") if s.strip()]
                        else:
                            main_keyword = line
                            synonyms = []
                        keyword_configs.append(
                            {"main": main_keyword, "synonyms": synonyms})

                if debug_mode and len(keyword_configs) > max_keywords_debug:
                    st.warning(
                        f"Debug mode: Limiting to first {max_keywords_debug} keywords: {', '.join(k['main'] for k in keyword_configs[:max_keywords_debug])}"
                    )
                    keyword_configs = keyword_configs[:max_keywords_debug]

                all_results = []
                for config in keyword_configs:
                    results = self.search_trends(
                        config["main"],
                        config["synonyms"],
                        useragents,
                        search_mode_value,
                        search_engine,
                        debug=debug_mode
                    )
                    if isinstance(results, list):
                        all_results.extend(results)
                    else:
                        st.error(t("trendwatcher_error").format(error=results))
                    time.sleep(random.uniform(delay_min, delay_max))

                st.session_state.trendwatcher_results = all_results
                st.session_state.debug_mode = debug_mode

        if st.session_state.trendwatcher_results:
            video_results = [
                r for r in st.session_state.trendwatcher_results if r["type"] == "video"]

            all_keywords = list(
                set(r["keyword"] for r in st.session_state.trendwatcher_results))
            selected_keywords = st.multiselect(
                t("trendwatcher_keyword_filter"),
                all_keywords,
                default=all_keywords,
                key="keyword_filter"
            )

            all_languages = list(
                set(r["language"] for r in st.session_state.trendwatcher_results))
            selected_language = st.selectbox(
                t("trendwatcher_language_filter"),
                ["All"] + all_languages,
                index=0,
                key="language_filter"
            )

            def filter_df(results):
                df = pd.DataFrame(results)
                if selected_keywords:
                    df = df[df["keyword"].isin(selected_keywords)]
                if selected_language != "All":
                    df = df[df["language"] == selected_language]
                df["Select"] = False
                cols = ["Select", "title", "url", "keyword", "language"]
                df = df[cols]
                return df

            selected_video_urls = []
            if video_results:
                st.subheader(t("trendwatcher_videos_table"))
                video_df = filter_df(video_results)
                if not video_df.empty:
                    st.session_state.select_all_videos = st.checkbox(
                        t("trendwatcher_select_all_videos"),
                        value=st.session_state.select_all_videos,
                        key="select_all_videos_checkbox"
                    )
                    video_df["Select"] = st.session_state.select_all_videos
                    edited_video_df = st.data_editor(
                        video_df[["Select", "title",
                                  "url", "keyword", "language"]],
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "Select for Export",
                                help="Check to include this video in the export",
                                default=False
                            ),
                            "url": st.column_config.LinkColumn(
                                "URL",
                                help="Click to visit the video",
                                display_text="Visit"
                            ),
                            "title": st.column_config.TextColumn(
                                "Title",
                                help="Video title"
                            ),
                            "keyword": st.column_config.TextColumn(
                                "Keyword",
                                help="Associated keyword"
                            ),
                            "language": st.column_config.TextColumn(
                                "Language",
                                help="Detected language"
                            )
                        },
                        disabled=["title", "url", "keyword", "language"],
                        hide_index=True,
                        key="video_selector"
                    )
                    video_df.update(edited_video_df[["Select"]])
                    selected_video_urls = video_df[video_df["Select"]]["url"].tolist(
                    )
                    if debug_mode:
                        st.write("Selected video URLs:", selected_video_urls)
                else:
                    st.info("No videos match the filters.")

            if not video_results:
                st.info(t("trendwatcher_no_results"))

            # Dans la méthode trend_watcher, avant le bouton de sauvegarde (juste avant if st.button(t("trendwatcher_save_button"), key="save_results"))
            overwrite = st.checkbox(
                t("overwrite_checkbox"), value=True, key="trendwatcher_overwrite")

            # Modifier l'appel à save_videos_to_csv dans le même bloc
            if st.button(t("trendwatcher_save_button"), key="save_results"):
                if selected_video_urls:
                    with st.spinner("Saving results..."):
                        try:
                            video_file = self.save_videos_to_csv(
                                working_dir, selected_keywords, selected_video_urls, overwrite)
                            st.success(t("trendwatcher_save_success").format(
                                dir=video_file))
                        except Exception as e:
                            st.error(
                                t("trendwatcher_error").format(error=str(e)))
                else:
                    st.warning(
                        "Please select at least one video to export.")
        else:
            st.info(t("trendwatcher_no_results"))

    def keyword_cluster(self, config):
        from widgets.keyword_cluster import KeywordClusteringWidget
        KeywordClusteringWidget("trendwatcher", "kwc",
                                plugin_manager=self.plugin_manager).display()

    def video_list(self, config):
        from widgets.video_list import VideoListWidget
        VideoListWidget("trendwatcher", "vlc",
                        plugin_manager=self.plugin_manager).display()

    def process_videos(self, config):
        from widgets.video_metadata import ProcessVideosWidget
        ProcessVideosWidget("trendwatcher", "pvw",
                            plugin_manager=self.plugin_manager).display()

    def run(self, config):
        """Main plugin logic"""
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Surveiller", "Process Videos", "Keyword cluster", "Video List"])
        with tab1:
            self.trend_watcher(config)
        with tab2:
            self.process_videos(config)
        with tab3:
            self.keyword_cluster(config)
        with tab4:
            self.video_list(config)


if __name__ == "__main__":
    st.write("Trendwatcher Plugin standalone test")
