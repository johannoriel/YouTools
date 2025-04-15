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
import os
import re
import csv
from youtube_api import YoutubeAPI
from datetime import datetime
import pytz
#from brave import Brave
import requests
import yt_dlp

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
    "trendwatcher_texts_table": "Text Articles",
    "trendwatcher_working_dir": "Working Directory",
    "trendwatcher_working_dir_default": "~/Videos",
    "trendwatcher_results": "Results for '{keyword}':",
    "trendwatcher_no_results": "No recent results found within 7 days",
    "trendwatcher_error": "Error during search: {error}",
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
    "trendwatcher_texts_table": "Articles Textes",
    "trendwatcher_working_dir": "Répertoire de travail",
    "trendwatcher_working_dir_default": "~/Vidéos",
    "trendwatcher_results": "Résultats pour '{keyword}':",
    "trendwatcher_no_results": "Aucun résultat récent trouvé dans les 7 derniers jours",
    "trendwatcher_error": "Erreur pendant la recherche : {error}",
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
                "search_videos": self.search_videos_duckduckgo,
                "search_texts": self.search_texts_duckduckgo,
                "config": {}  # Pas de config supplémentaire pour DDG
            },
            #"brave": {
            #    "name": t("trendwatcher_search_engine_brave"),
            #    "search_videos": self.search_videos_brave,
            #    "search_texts": self.search_texts_brave,
            #    "config": {"api_key": lambda: self.plugin_manager.config.get(self.name, {}).get("trendwatcher_brave_api_key", "")}
            #},
            "google": {
                "name": t("trendwatcher_search_engine_google"),
                "search_videos": self.search_videos_google,
                "search_texts": self.search_texts_google,
                "config": {
                    "api_key": lambda: self.plugin_manager.config.get(self.name, {}).get("trendwatcher_google_api_key", ""),
                    "cx_id": lambda: self.plugin_manager.config.get(self.name, {}).get("trendwatcher_google_cx_id", "")
                }
            },
            "ytdlp": {
                "name": t("trendwatcher_search_engine_ytdlp"),
                "search_videos": self.search_videos_ytdlp,
                "search_texts": self.search_texts_ytdlp,  # Ne renvoie rien pour textes
                "config": {}
            },
            "searxng": {
                "name": t("trendwatcher_search_engine_searxng"),
                "search_videos": self.search_videos_searxng,
                "search_texts": self.search_texts_searxng,
                "config": {
                    "server_url": lambda: self.plugin_manager.config.get(self.name, {}).get("trendwatcher_searxng_server_url", "")
                }
            },
            "bing": {
                "name": t("trendwatcher_search_engine_bing"),
                "search_videos": self.search_videos_bing,
                "search_texts": self.search_texts_bing,
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

    def search_videos_duckduckgo(self, query, keyword, useragents, valid_video_domains, debug=False):
        """Search for recent videos with URL validation using DuckDuckGo"""
        from urllib.parse import urlparse

        # Create new DDGS instance with random User-Agent
        headers = {"User-Agent": random.choice(useragents)}
        ddgs = DDGS(headers=headers)

        video_results = ddgs.videos(
            keywords=query,
            region="fr-fr",
            timelimit="w",
            max_results=5
        )

        results = []
        cutoff_date = datetime.now() - timedelta(days=7)

        for video in video_results:
            try:
                parsed_url = urlparse(video["content"])
                domain = parsed_url.netloc.lower().replace("www.", "")
                if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                    st.warning(
                        f"Non-video URL detected: {video['content']} "
                        f"for keyword '{keyword}'. Skipping."
                    )
                    continue
            except Exception as e:
                st.warning(
                    f"Invalid URL: {video['content']} "
                    f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
                )
                continue

            published_date = self.parse_date(video["published"], debug=debug)
            if published_date and published_date > cutoff_date:
                title = video["title"].replace("|", "")
                language = detect(video["title"]) if video["title"] else "unknown"
                is_youtube = domain in ["youtube.com", "youtu.be"]

                results.append({
                    "keyword": keyword,
                    "url": video["content"],
                    "video_id": self.extract_youtube_id(video["content"]) if is_youtube else "N/A",
                    "title": title,
                    "view_count": "",
                    "language": language,
                    "published_at": "",
                    "channel_id": "N/A",
                    "channel_title": "N/A",
                    "subscriber_count": "",
                    "comment_count": "",
                    "relevance_score": 0,
                    "type": "video"
                })

        return results

    def search_texts_duckduckgo(self, query, keyword, useragents, debug=False):
        """Search for recent text articles using DuckDuckGo"""
        headers = {"User-Agent": random.choice(useragents)}
        ddgs = DDGS(headers=headers)

        text_results = ddgs.text(
            keywords=query,
            region="fr-fr",
            timelimit="w",
            max_results=5
        )

        results = []
        for text in text_results:
            title = text["title"].replace("|", "")
            language = detect(text["title"]) if text["title"] else "unknown"
            if debug:
                st.write(f"Text article: [{title}]({text['href']})")
            results.append({
                "keyword": keyword,
                "url": text["href"],
                "title": title,
                "view_count": "",
                "language": language,
                "published_at": "",
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "comment_count": "",
                "relevance_score": 0,
                "type": "web"
            })

        return results


    def search_videos_brave(self, query, keyword, useragents, valid_video_domains, debug=False):
        """Search for recent videos with URL validation using Brave Search"""
        from urllib.parse import urlparse

        api_key = self.plugin_manager.config.get(self.name, {}).get("trendwatcher_brave_api_key", "")
        if not api_key:
            st.error(t("trendwatcher_error").format(error="Brave Search API key is required."))
            return []

        brave = Brave(api_key)
        try:
            # Utiliser raw=True pour éviter la validation Pydantic
            search_results = brave.search(
                q=query,
                count=5,
                result_filter="videos",
                freshness="pw",
                raw=True
            )
            if debug:
                st.write(f"Brave Search response for videos: {search_results}")
            # Vérifier si la réponse contient des résultats vidéos
            video_results = search_results.get("videos", {}).get("results", []) if search_results.get("videos") else []
            if not video_results:
                if debug:
                    st.warning(f"No video results found for query: {query}")
                return []

            results = []
            cutoff_date = datetime.now() - timedelta(days=7)

            for video in video_results[:5]:
                url = video.get("url", "")
                title = video.get("title", "")
                if not url or not title:
                    if debug:
                        st.warning(f"Skipping video with missing url or title for query: {query}")
                    continue

                try:
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc.lower().replace("www.", "")
                    if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                        if debug:
                            st.warning(
                                f"Non-video URL detected: {url} "
                                f"for keyword '{keyword}'. Skipping."
                            )
                        continue
                except Exception as e:
                    if debug:
                        st.warning(
                            f"Invalid URL: {url} "
                            f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
                        )
                    continue

                # Extraire la date de publication
                published_date_str = video.get("meta", {}).get("published_date", "") or video.get("published_date", "")
                published_date = self.parse_date(published_date_str, debug=debug)
                if published_date and published_date > cutoff_date:
                    title = title.replace("|", "")
                    language = detect(title) if title else "unknown"
                    is_youtube = domain in ["youtube.com", "youtu.be"]

                    results.append({
                        "keyword": keyword,
                        "url": url,
                        "video_id": self.extract_youtube_id(url) if is_youtube else "N/A",
                        "title": title,
                        "view_count": "",
                        "language": language,
                        "published_at": "",
                        "channel_id": "N/A",
                        "channel_title": "N/A",
                        "subscriber_count": "",
                        "comment_count": "",
                        "relevance_score": 0,
                        "type": "video"
                    })

            return results
        except Exception as e:
            if debug:
                st.error(t("trendwatcher_error").format(error=f"Brave Search error: {str(e)}"))
            return []

    def search_texts_brave(self, query, keyword, useragents, debug=False):
        """Search for recent text articles using Brave Search"""
        api_key = self.plugin_manager.config.get(self.name, {}).get("trendwatcher_brave_api_key", "")
        if not api_key:
            st.error(t("trendwatcher_error").format(error="Brave Search API key is required."))
            return []

        brave = Brave(api_key)
        try:
            # Utiliser raw=True pour éviter la validation Pydantic
            search_results = brave.search(
                q=query,
                count=5,
                result_filter="web",
                freshness="pw",
                raw=True
            )
            if debug:
                st.write(f"Brave Search response for web: {search_results}")
            # Vérifier si la réponse contient des résultats web
            web_results = search_results.get("web", {}).get("results", []) if search_results.get("web") else []
            if not web_results:
                if debug:
                    st.warning(f"No web results found for query: {query}")
                return []

            results = []
            for text in web_results[:5]:
                url = text.get("url", "")
                title = text.get("title", "")
                if not url or not title:
                    if debug:
                        st.warning(f"Skipping article with missing url or title for query: {query}")
                    continue

                title = title.replace("|", "")
                language = detect(title) if title else "unknown"
                if debug:
                    st.write(f"Text article: [{title}]({url})")
                results.append({
                    "keyword": keyword,
                    "url": url,
                    "title": title,
                    "view_count": "",
                    "language": language,
                    "published_at": "",
                    "channel_id": "N/A",
                    "channel_title": "N/A",
                    "subscriber_count": "",
                    "comment_count": "",
                    "relevance_score": 0,
                    "type": "web"
                })

            return results
        except Exception as e:
            if debug:
                st.error(t("trendwatcher_error").format(error=f"Brave Search error: {str(e)}"))
            return []


    def search_videos_google(self, query, keyword, useragents, valid_video_domains, debug=False):
        """Search for recent videos using Google Custom Search API"""
        from urllib.parse import urlparse

        api_key = self.plugin_manager.config.get(self.name, {}).get("trendwatcher_google_api_key", "")
        cx_id = self.plugin_manager.config.get(self.name, {}).get("trendwatcher_google_cx_id", "")
        if not api_key or not cx_id:
            st.error(t("trendwatcher_error").format(error="Google API key and CX ID are required."))
            return []

        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": f"{query} site:youtube.com",
            "key": api_key,
            "cx": cx_id,
            "num": 5,
            "dateRestrict": "w1"  # Limiter à 1 semaine
        }

        try:
            headers = {"User-Agent": random.choice(useragents)}
            response = requests.get(base_url, params=params, headers=headers)
            if response.status_code != 200:
                if debug:
                    st.error(f"Google API error: {response.text}")
                return []
            data = response.json()

            results = []
            cutoff_date = datetime.now() - timedelta(days=7)

            for item in data.get("items", [])[:5]:
                url = item.get("link", "")
                title = item.get("title", "").replace("|", "")
                if not url or not title:
                    continue

                try:
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc.lower().replace("www.", "")
                    if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                        if debug:
                            st.warning(
                                f"Non-video URL detected: {url} "
                                f"for keyword '{keyword}'. Skipping."
                            )
                        continue
                except Exception as e:
                    if debug:
                        st.warning(
                            f"Invalid URL: {url} "
                            f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
                        )
                    continue

                # Google ne fournit pas toujours la date exacte, on suppose récent
                language = detect(title) if title else "unknown"
                is_youtube = domain in ["youtube.com", "youtu.be"]

                results.append({
                    "keyword": keyword,
                    "url": url,
                    "video_id": self.extract_youtube_id(url) if is_youtube else "N/A",
                    "title": title,
                    "view_count": "",
                    "language": language,
                    "published_at": "",
                    "channel_id": "N/A",
                    "channel_title": "N/A",
                    "subscriber_count": "",
                    "comment_count": "",
                    "relevance_score": 0,
                    "type": "video"
                })

            return results
        except Exception as e:
            if debug:
                st.error(t("trendwatcher_error").format(error=f"Google Search error: {str(e)}"))
            return []

    def search_texts_google(self, query, keyword, useragents, debug=False):
        """Search for recent text articles using Google Custom Search API"""
        api_key = self.plugin_manager.config.get(self.name, {}).get("trendwatcher_google_api_key", "")
        cx_id = self.plugin_manager.config.get(self.name, {}).get("trendwatcher_google_cx_id", "")
        if not api_key or not cx_id:
            st.error(t("trendwatcher_error").format(error="Google API key and CX ID are required."))
            return []

        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": api_key,
            "cx": cx_id,
            "num": 5,
            "dateRestrict": "w1"  # Limiter à 1 semaine
        }

        try:
            headers = {"User-Agent": random.choice(useragents)}
            response = requests.get(base_url, params=params, headers=headers)
            if response.status_code != 200:
                if debug:
                    st.error(f"Google API error: {response.text}")
                return []
            data = response.json()

            results = []
            for item in data.get("items", [])[:5]:
                url = item.get("link", "")
                title = item.get("title", "").replace("|", "")
                if not url or not title:
                    continue

                language = detect(title) if title else "unknown"
                if debug:
                    st.write(f"Text article: [{title}]({url})")
                results.append({
                    "keyword": keyword,
                    "url": url,
                    "title": title,
                    "view_count": "",
                    "language": language,
                    "published_at": "",
                    "channel_id": "N/A",
                    "channel_title": "N/A",
                    "subscriber_count": "",
                    "comment_count": "",
                    "relevance_score": 0,
                    "type": "web"
                })

            return results
        except Exception as e:
            if debug:
                st.error(t("trendwatcher_error").format(error=f"Google Search error: {str(e)}"))
            return []

    def search_videos_ytdlp(self, query, keyword, useragents, valid_video_domains, debug=False):
        """Search for recent videos using yt-dlp"""
        from urllib.parse import urlparse

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,  # Ne télécharge pas, juste les métadonnées
            "max_downloads": 5,
            "dateafter": (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Rechercher sur YouTube
                search_query = f"ytsearch5:{query}"  # Limite à 5 résultats
                info = ydl.extract_info(search_query, download=False)

            results = []
            cutoff_date = datetime.now() - timedelta(days=7)

            for entry in info.get("entries", [])[:5]:
                url = entry.get("url", "") or entry.get("webpage_url", "")
                title = entry.get("title", "").replace("|", "")
                if not url or not title:
                    continue

                try:
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc.lower().replace("www.", "")
                    if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                        if debug:
                            st.warning(
                                f"Non-video URL detected: {url} "
                                f"for keyword '{keyword}'. Skipping."
                            )
                        continue
                except Exception as e:
                    if debug:
                        st.warning(
                            f"Invalid URL: {url} "
                            f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
                        )
                    continue

                # Vérifier la date
                upload_date = entry.get("upload_date", "")
                if upload_date:
                    try:
                        published_date = datetime.strptime(upload_date, "%Y%m%d")
                        if published_date <= cutoff_date:
                            continue
                    except ValueError:
                        if debug:
                            st.warning(f"Invalid date format for {url}: {upload_date}")
                        continue
                else:
                    published_date = datetime.now()  # Suppose récent si pas de date

                language = detect(title) if title else "unknown"
                is_youtube = domain in ["youtube.com", "youtu.be"]

                results.append({
                    "keyword": keyword,
                    "url": url,
                    "video_id": entry.get("id", "N/A") if is_youtube else "N/A",
                    "title": title,
                    "view_count": str(entry.get("view_count", "")),
                    "language": language,
                    "published_at": upload_date,
                    "channel_id": entry.get("channel_id", "N/A"),
                    "channel_title": entry.get("uploader", "N/A"),
                    "subscriber_count": "",
                    "comment_count": "",
                    "relevance_score": 0,
                    "type": "video"
                })

            return results
        except Exception as e:
            if debug:
                st.error(t("trendwatcher_error").format(error=f"yt-dlp error: {str(e)}"))
            return []

    def search_texts_ytdlp(self, query, keyword, useragents, debug=False):
        """Search for text articles using yt-dlp (not supported, returns empty)"""
        if debug:
            st.warning("yt-dlp does not support text article search.")
        return []

    def search_videos_searxng(self, query, keyword, useragents, valid_video_domains, debug=False):
        """Search for recent videos using SearxNG"""
        from urllib.parse import urlparse
        import requests

        server_url = self.plugin_manager.config.get(self.name, {}).get("trendwatcher_searxng_server_url", "")
        if not server_url:
            st.error(t("trendwatcher_error").format(error="SearxNG server URL is required."))
            return []

        # Configure query for videos (restrict to YouTube)
        search_url = f"{server_url.rstrip('/')}/search"
        params = {
            "q": f"{query} site:youtube.com",
            "categories": "general,videos",
            "time_range": "week",  # Limit to past week
            "format": "json",
            "safesearch": 0,
            "language": "all"
        }

        try:
            headers = {"User-Agent": random.choice(useragents)}
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                if debug:
                    st.error(f"SearxNG API error: {response.text}")
                return []
            data = response.json()

            results = []
            cutoff_date = datetime.now() - timedelta(days=7)

            for item in data.get("results", [])[:5]:
                url = item.get("url", "")
                title = item.get("title", "").replace("|", "")
                if not url or not title:
                    continue

                try:
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc.lower().replace("www.", "")
                    if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                        if debug:
                            st.warning(
                                f"Non-video URL detected: {url} "
                                f"for keyword '{keyword}'. Skipping."
                            )
                        continue
                except Exception as e:
                    if debug:
                        st.warning(
                            f"Invalid URL: {url} "
                            f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
                        )
                    continue

                # SearxNG may not provide exact dates, assume recent
                language = detect(title) if title else "unknown"
                is_youtube = domain in ["youtube.com", "youtu.be"]

                results.append({
                    "keyword": keyword,
                    "url": url,
                    "video_id": self.extract_youtube_id(url) if is_youtube else "N/A",
                    "title": title,
                    "view_count": "",
                    "language": language,
                    "published_at": "",
                    "channel_id": "N/A",
                    "channel_title": "N/A",
                    "subscriber_count": "",
                    "comment_count": "",
                    "relevance_score": 0,
                    "type": "video"
                })

            return results
        except Exception as e:
            if debug:
                st.error(t("trendwatcher_error").format(error=f"SearxNG Search error: {str(e)}"))
            return []

    def search_texts_searxng(self, query, keyword, useragents, debug=False):
        """Search for recent text articles using SearxNG"""
        import requests

        server_url = self.plugin_manager.config.get(self.name, {}).get("trendwatcher_searxng_server_url", "")
        if not server_url:
            st.error(t("trendwatcher_error").format(error="SearxNG server URL is required."))
            return []

        search_url = f"{server_url.rstrip('/')}/search"
        params = {
            "q": query,
            "categories": "general",
            "time_range": "week",
            "format": "json",
            "safesearch": 0,
            "language": "all"
        }

        try:
            headers = {"User-Agent": random.choice(useragents)}
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                if debug:
                    st.error(f"SearxNG API error: {response.text}")
                return []
            data = response.json()

            results = []
            for item in data.get("results", [])[:5]:
                url = item.get("url", "")
                title = item.get("title", "").replace("|", "")
                if not url or not title:
                    continue

                language = detect(title) if title else "unknown"
                if debug:
                    st.write(f"Text article: [{title}]({url})")
                results.append({
                    "keyword": keyword,
                    "url": url,
                    "title": title,
                    "view_count": "",
                    "language": language,
                    "published_at": "",
                    "channel_id": "N/A",
                    "channel_title": "N/A",
                    "subscriber_count": "",
                    "comment_count": "",
                    "relevance_score": 0,
                    "type": "web"
                })

            return results
        except Exception as e:
            if debug:
                st.error(t("trendwatcher_error").format(error=f"SearxNG Search error: {str(e)}"))
            return []

    def search_videos_bing(self, query, keyword, useragents, valid_video_domains, debug=False):
        """Search for recent videos using Bing Web Search API"""
        from urllib.parse import urlparse
        import requests

        api_key = self.plugin_manager.config.get(self.name, {}).get("trendwatcher_bing_api_key", "")
        if not api_key:
            st.error(t("trendwatcher_error").format(error="Bing API key is required."))
            return []

        search_url = "https://api.bing.microsoft.com/v7.0/search"
        params = {
            "q": f"{query} site:youtube.com",
            "count": 5,
            "freshness": "Week",  # Limit to past week
            "responseFilter": "Videos,Webpages"
        }
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "User-Agent": random.choice(useragents)
        }

        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                if debug:
                    st.error(f"Bing API error: {response.text}")
                return []
            data = response.json()

            results = []
            cutoff_date = datetime.now() - timedelta(days=7)

            # Bing returns videos in 'videos' or 'webPages' depending on query
            items = data.get("videos", {}).get("value", []) or data.get("webPages", {}).get("value", [])

            for item in items[:5]:
                url = item.get("url") or item.get("contentUrl", "")
                title = item.get("name", "").replace("|", "")
                if not url or not title:
                    continue

                try:
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc.lower().replace("www.", "")
                    if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                        if debug:
                            st.warning(
                                f"Non-video URL detected: {url} "
                                f"for keyword '{keyword}'. Skipping."
                            )
                        continue
                except Exception as e:
                    if debug:
                        st.warning(
                            f"Invalid URL: {url} "
                            f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
                        )
                    continue

                # Bing may provide datePublished
                date_str = item.get("datePublished", "")
                published_date = self.parse_date(date_str, debug=debug) if date_str else None
                if published_date and published_date <= cutoff_date:
                    continue

                language = detect(title) if title else "unknown"
                is_youtube = domain in ["youtube.com", "youtu.be"]

                results.append({
                    "keyword": keyword,
                    "url": url,
                    "video_id": self.extract_youtube_id(url) if is_youtube else "N/A",
                    "title": title,
                    "view_count": "",
                    "language": language,
                    "published_at": "",
                    "channel_id": "N/A",
                    "channel_title": "N/A",
                    "subscriber_count": "",
                    "comment_count": "",
                    "relevance_score": 0,
                    "type": "video"
                })

            return results
        except Exception as e:
            if debug:
                st.error(t("trendwatcher_error").format(error=f"Bing Search error: {str(e)}"))
            return []

    def search_texts_bing(self, query, keyword, useragents, debug=False):
        """Search for recent text articles using Bing Web Search API"""
        import requests

        api_key = self.plugin_manager.config.get(self.name, {}).get("trendwatcher_bing_api_key", "")
        if not api_key:
            st.error(t("trendwatcher_error").format(error="Bing API key is required."))
            return []

        search_url = "https://api.bing.microsoft.com/v7.0/search"
        params = {
            "q": query,
            "count": 5,
            "freshness": "Week",
            "responseFilter": "Webpages"
        }
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "User-Agent": random.choice(useragents)
        }

        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                if debug:
                    st.error(f"Bing API error: {response.text}")
                return []
            data = response.json()

            results = []
            for item in data.get("webPages", {}).get("value", [])[:5]:
                url = item.get("url", "")
                title = item.get("name", "").replace("|", "")
                if not url or not title:
                    continue

                language = detect(title) if title else "unknown"
                if debug:
                    st.write(f"Text article: [{title}]({url})")
                results.append({
                    "keyword": keyword,
                    "url": url,
                    "title": title,
                    "view_count": "",
                    "language": language,
                    "published_at": "",
                    "channel_id": "N/A",
                    "channel_title": "N/A",
                    "subscriber_count": "",
                    "comment_count": "",
                    "relevance_score": 0,
                    "type": "web"
                })

            return results
        except Exception as e:
            if debug:
                st.error(t("trendwatcher_error").format(error=f"Bing Search error: {str(e)}"))
            return []


    def search_trends(self, main_keyword, synonyms, useragents, search_mode="or", search_engine="duckduckgo", debug=False):
        """Search for recent videos and web content"""
        # Get search engine methods
        engine = self.SEARCH_ENGINES.get(search_engine, self.SEARCH_ENGINES["duckduckgo"])
        search_videos = engine["search_videos"]
        search_texts = engine["search_texts"]

        # Prepare query based on search mode
        if search_mode == "or":
            query_terms = [main_keyword] + synonyms
            query = " OR ".join(f'"{term}"' for term in query_terms if term)
            if search_engine == "duckduckgo":
                query += " site:youtube.com OR -inurl:(signup login)"
            elif search_engine == "google":
                query += " site:youtube.com"
            elif search_engine == "searxng":
                query += " site:youtube.com"  # SearxNG supports site: operator
            elif search_engine == "bing":
                query += " site:youtube.com"
            # ytdlp and others don't need specific restrictions
            if debug:
                st.write(t("trendwatcher_debug_query").format(query=query))
            queries = [(query, main_keyword)]
        else:
            queries = [(f'"{main_keyword}"', main_keyword)]
            queries.extend((f'"{syn}"', main_keyword) for syn in synonyms)
            if search_engine == "duckduckgo":
                queries = [(f"{q} site:youtube.com OR -inurl:(signup login)", k) for q, k in queries]
            elif search_engine == "google":
                queries = [(f"{q} site:youtube.com", k) for q, k in queries]
            elif search_engine == "searxng":
                queries = [(f"{q} site:youtube.com", k) for q, k in queries]
            elif search_engine == "bing":
                queries = [(f"{q} site:youtube.com", k) for q, k in queries]
            if debug:
                st.write(t("trendwatcher_debug_query").format(query=", ".join(q for q, _ in queries)))

        # List of valid video platform domains
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
                video_results = search_videos(query, keyword, useragents, valid_video_domains, debug=debug)
                text_results = search_texts(query, keyword, useragents, debug=debug)
                all_results.extend(video_results + text_results)

            if debug:
                video_count = sum(1 for r in all_results if r.get("type") == "video")
                text_count = sum(1 for r in all_results if r.get("type") == "web")
                st.write(t("trendwatcher_debug_results").format(
                    count=len(all_results),
                    vids=video_count,
                    texts=text_count
                ))

            return all_results
        except Exception as e:
            return str(e)

    def extract_youtube_id(self, url):
        """Extract YouTube video ID from URL"""
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            r"(?:embed\/)([0-9A-Za-z_-]{11})",
            r"(?:watch\?v=)([0-9A-Za-z_-]{11})"
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return "N/A"

    def extract_channel_info_yt_dlp(self, video_url, debug=False):
        """Extract channel ID, title, and subscriber count using yt-dlp"""
        try:
            import yt_dlp

            # Configure yt-dlp options
            ydl_opts = {
                "quiet": True,  # Suppress console output
                "no_warnings": True,
                "extract_flat": True,  # Don't download, just extract metadata
                "force_generic_extractor": False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

                # Extract required fields
                channel_id = info.get("channel_id", "N/A")
                channel_title = info.get("channel", "N/A")  # 'channel' contains the channel name
                subscriber_count = info.get("channel_follower_count", "N/A")

                # Format subscriber_count
                if isinstance(subscriber_count, int):
                    subscriber_count = f"{subscriber_count:,}".replace(",", " ") + " subscribers"

                if debug:
                    st.write(f"Extracted from {video_url}: channel_id={channel_id}, channel_title={channel_title}, subscriber_count={subscriber_count}")

                return {
                    "channel_id": channel_id,
                    "channel_title": channel_title,
                    "subscriber_count": subscriber_count
                }

        except Exception as e:
            if debug:
                st.write(f"Error extracting channel info with yt-dlp for {video_url}: {str(e)}")
            return {
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "N/A"
            }

    def extract_video_metadata_yt_dlp(self, video_url, debug=False):
        """Extract video metadata including channel info, view count, comment count, and published date using yt-dlp"""
        try:
            import yt_dlp

            # Configure yt-dlp options
            ydl_opts = {
                "quiet": True,  # Suppress console output
                "no_warnings": True,
                "extract_flat": True,  # Don't download, just extract metadata
                "force_generic_extractor": False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

                # Extract required fields
                channel_id = info.get("channel_id", "N/A")
                channel_title = info.get("channel", "N/A")
                subscriber_count = info.get("channel_follower_count", "N/A")
                view_count = info.get("view_count", "N/A")
                comment_count = info.get("comment_count", "N/A")
                published_at = info.get("upload_date", "N/A")

                # Format subscriber_count
                if isinstance(subscriber_count, int):
                    subscriber_count = str(subscriber_count)  # Convert to string without spaces or suffix
                else:
                    subscriber_count = ""  # Replace "N/A" with empty string

                # Format view_count
                if isinstance(view_count, int):
                    view_count = str(view_count)  # Convert to string without spaces
                else:
                    view_count = ""  # Replace "N/A" with empty string

                # Format comment_count
                if isinstance(comment_count, int):
                    comment_count = str(comment_count)  # Convert to string without spaces
                else:
                    comment_count = ""  # Replace "N/A" with empty string

                # Format published_at (comes as YYYYMMDD, convert to YYYY-MM-DD)
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
                st.write(f"Error extracting video metadata with yt-dlp for {video_url}: {str(e)}")
            return {
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "view_count": "",
                "comment_count": "",
                "published_at": ""
            }

    def save_videos_to_csv(self, working_dir, selected_keywords, selected_urls):
        """Save selected video results to CSV file"""
        from urllib.parse import urlparse
        from datetime import datetime

        # Expand ~ in working_dir
        working_dir = os.path.expanduser(working_dir)
        os.makedirs(working_dir, exist_ok=True)

        video_file = os.path.join(working_dir, "video_list.csv")

        # Use official headers
        headers = VIDEO_CSV_HEADERS

        # Filter results based on selected_keywords and selected_urls
        filtered_results = [
            r for r in st.session_state.trendwatcher_results
            if r["keyword"] in selected_keywords and r["url"] in selected_urls and r["type"] == "video"
        ]

        # Get debug mode from session state
        debug_mode = st.session_state.get("debug_mode", False)

        # Initialize YoutubeAPI for relevance score
        youtube_api = YoutubeAPI(self.plugin_manager.config if self.plugin_manager else {})

        # Write videos CSV
        with open(video_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for video in filtered_results:
                # Check if the video is from YouTube
                parsed_url = urlparse(video["url"])
                domain = parsed_url.netloc.lower().replace("www.", "")
                is_youtube = domain in ["youtube.com", "youtu.be"]

                if is_youtube:
                    # Fetch metadata only for selected videos
                    metadata = self.extract_video_metadata_yt_dlp(video["url"], debug=debug_mode)

                    # Calculate relevance score
                    published_at = metadata["published_at"]
                    relevance_score = 0
                    if published_at and metadata["subscriber_count"] and metadata["comment_count"]:
                        try:
                            # Convert YYYY-MM-DD to YYYY-MM-DDTHH:MM:SSZ
                            if published_at:
                                try:
                                    # Parse YYYY-MM-DD and convert to ISO format
                                    date_obj = datetime.strptime(published_at, "%Y-%m-%d")
                                    published_at_iso = date_obj.strftime("%Y-%m-%dT00:00:00Z")
                                except ValueError as e:
                                    if debug_mode:
                                        st.write(f"Error parsing date {published_at} for {video['url']}: {str(e)}")
                                    published_at_iso = None
                            else:
                                published_at_iso = None

                            if published_at_iso:
                                video_data = {
                                    "published_at": published_at_iso,
                                    "subscriber_count": int(metadata["subscriber_count"] or 0),
                                    "comment_count": int(metadata["comment_count"] or 0)
                                }
                                relevance_score = youtube_api.calculate_relevance_score(video_data)
                        except Exception as e:
                            if debug_mode:
                                st.write(f"Error calculating relevance score for {video['url']}: {str(e)}")
                else:
                    metadata = {
                        "view_count": "",
                        "published_at": "",
                        "channel_id": "N/A",
                        "channel_title": "N/A",
                        "subscriber_count": "",
                        "comment_count": ""
                    }
                    relevance_score = 0

                writer.writerow([
                    video["keyword"],
                    video["url"],
                    video["video_id"],
                    video["title"],
                    metadata["view_count"],
                    video["language"],
                    metadata["published_at"],
                    metadata["channel_id"],
                    metadata["channel_title"],
                    metadata["subscriber_count"],
                    metadata["comment_count"],
                    relevance_score
                ])

        return working_dir

    def save_articles_to_csv(self, working_dir, selected_keywords, selected_urls):
        """Save selected article results to CSV file"""
        # Expand ~ in working_dir
        working_dir = os.path.expanduser(working_dir)
        os.makedirs(working_dir, exist_ok=True)

        article_file = os.path.join(working_dir, "article_list.csv")

        # CSV headers
        headers = ["keyword", "url", "title", "views", "language", "date"]

        # Filter results based on selected_keywords and selected_urls
        filtered_results = [
            r for r in st.session_state.trendwatcher_results
            if r["keyword"] in selected_keywords and r["url"] in selected_urls and r["type"] == "web"
        ]

        # Write articles CSV
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

    def run(self, config):
        """Main plugin logic"""
        st.header(t("trendwatcher_header"))

        keywords_config = config.get(self.name, {}).get("trendwatcher_keywords", t("trendwatcher_keywords_default"))
        useragents = config.get(self.name, {}).get("trendwatcher_useragents", t("trendwatcher_useragents_default")).split("\n")
        delay_min = float(config.get(self.name, {}).get("trendwatcher_delay_min", 1))
        delay_max = float(config.get(self.name, {}).get("trendwatcher_delay_max", 3))
        working_dir = config.get(self.name, {}).get("working_dir", t("trendwatcher_working_dir_default"))

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

        # Add number input for max keywords in debug mode
        max_keywords_debug = 3
        if debug_mode:
            max_keywords_debug = st.number_input(
                "Maximum Keywords to Search in Debug Mode",
                min_value=1,
                max_value=100,
                value=3,
                step=1
            )

        # Add search engine selection
        search_engine_options = {engine_id: engine["name"] for engine_id, engine in self.SEARCH_ENGINES.items()}
        search_engine_name = st.selectbox(
            t("trendwatcher_search_engine_label"),
            list(search_engine_options.values()),
            index=0,
            key="search_engine"
        )
        search_engine = next(k for k, v in search_engine_options.items() if v == search_engine_name)

        # Add search mode selection
        search_mode = st.selectbox(
            t("trendwatcher_search_mode_label"),
            [t("trendwatcher_search_mode_or"), t("trendwatcher_search_mode_subsearches")],
            index=0,
            key="search_mode"
        )
        search_mode_value = "or" if search_mode == t("trendwatcher_search_mode_or") else "subsearches"

        # Initialize session state
        if "trendwatcher_results" not in st.session_state:
            st.session_state.trendwatcher_results = []
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = debug_mode
        if "select_all_videos" not in st.session_state:
            st.session_state.select_all_videos = False
        if "select_all_texts" not in st.session_state:
            st.session_state.select_all_texts = False

        if st.button(t("trendwatcher_search_button"), key="search_trends"):
            with st.spinner(t("trendwatcher_processing")):
                # Parse keywords with synonyms
                keyword_configs = []
                for line in keywords_input.split("\n"):
                    line = line.strip()
                    if line:
                        if ":" in line:
                            main_keyword, synonyms = line.split(":", 1)
                            main_keyword = main_keyword.strip()
                            synonyms = [s.strip() for s in synonyms.split(",") if s.strip()]
                        else:
                            main_keyword = line
                            synonyms = []
                        keyword_configs.append({"main": main_keyword, "synonyms": synonyms})

                # Limit keywords in debug mode
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
            # Split into videos and texts
            video_results = [r for r in st.session_state.trendwatcher_results if r["type"] == "video"]
            text_results = [r for r in st.session_state.trendwatcher_results if r["type"] == "web"]

            # Filters
            all_keywords = list(set(r["keyword"] for r in st.session_state.trendwatcher_results))
            selected_keywords = st.multiselect(
                t("trendwatcher_keyword_filter"),
                all_keywords,
                default=all_keywords,
                key="keyword_filter"
            )

            all_languages = list(set(r["language"] for r in st.session_state.trendwatcher_results))
            selected_language = st.selectbox(
                t("trendwatcher_language_filter"),
                ["All"] + all_languages,
                index=0,
                key="language_filter"
            )

            # Filter and create DataFrames
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

            # Videos table
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
                        video_df[["Select", "title", "url", "keyword", "language"]],
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
                    selected_video_urls = video_df[video_df["Select"]]["url"].tolist()
                    if debug_mode:
                        st.write("Selected video URLs:", selected_video_urls)
                else:
                    st.info("No videos match the filters.")

            # Texts table
            selected_text_urls = []
            if text_results:
                st.subheader(t("trendwatcher_texts_table"))
                text_df = filter_df(text_results)
                if not text_df.empty:
                    st.session_state.select_all_texts = st.checkbox(
                        t("trendwatcher_select_all_texts"),
                        value=st.session_state.select_all_texts,
                        key="select_all_texts_checkbox"
                    )
                    text_df["Select"] = st.session_state.select_all_texts
                    edited_text_df = st.data_editor(
                        text_df[["Select", "title", "url", "keyword", "language"]],
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "Select for Export",
                                help="Check to include this article in the export",
                                default=False
                            ),
                            "url": st.column_config.LinkColumn(
                                "URL",
                                help="Click to visit the article",
                                display_text="Visit"
                            ),
                            "title": st.column_config.TextColumn(
                                "Title",
                                help="Article title"
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
                        key="text_selector"
                    )
                    text_df.update(edited_text_df[["Select"]])
                    selected_text_urls = text_df[text_df["Select"]]["url"].tolist()
                    if debug_mode:
                        st.write("Selected text URLs:", selected_text_urls)
                else:
                    st.info("No text articles match the filters.")

            if not video_results and not text_results:
                st.info(t("trendwatcher_no_results"))

            if st.button(t("trendwatcher_save_button"), key="save_results"):
                if selected_video_urls or selected_text_urls:
                    with st.spinner("Saving results..."):
                        try:
                            video_dir = None
                            if selected_video_urls:
                                video_dir = self.save_videos_to_csv(working_dir, selected_keywords, selected_video_urls)

                            article_dir = None
                            if selected_text_urls:
                                article_dir = self.save_articles_to_csv(working_dir, selected_keywords, selected_text_urls)

                            if video_dir or article_dir:
                                st.success(t("trendwatcher_save_success").format(dir=video_dir or article_dir))
                            else:
                                st.warning("No results selected for export.")
                        except Exception as e:
                            st.error(t("trendwatcher_error").format(error=str(e)))
                else:
                    st.warning("Please select at least one video or article to export.")
        else:
            st.info(t("trendwatcher_no_results"))

if __name__ == "__main__":
    st.write("Trendwatcher Plugin standalone test")
