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
})

# Official CSV headers
VIDEO_CSV_HEADERS = [
    "keyword", "url", "video_id", "title", "view_count", "language", "published_at",
    "channel_id", "channel_title", "subscriber_count", "comment_count", "relevance_score"
]

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
            },
            "trendwatcher_working_dir": {
                "type": "text",
                "label": t("trendwatcher_working_dir"),
                "default": t("trendwatcher_working_dir_default")
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

    def search_videos(self, keyword, useragents, valid_video_domains, debug=False):
        """Search for recent videos with URL validation"""
        from urllib.parse import urlparse

        # Create new DDGS instance with random User-Agent
        headers = {"User-Agent": random.choice(useragents)}
        ddgs = DDGS(headers=headers)

        video_results = ddgs.videos(
            keywords=keyword,
            region="fr-fr",
            timelimit="w",
            max_results=5
        )

        results = []
        cutoff_date = datetime.now() - timedelta(days=7)

        # Process video results
        for video in video_results:
            # Validate URL
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

                # Check if the video is from YouTube
                is_youtube = domain in ["youtube.com", "youtu.be"]

                results.append({
                    "keyword": keyword,
                    "url": video["content"],
                    "video_id": self.extract_youtube_id(video["content"]) if is_youtube else "N/A",
                    "title": title,
                    "view_count": "",  # Metadata will be fetched later
                    "language": language,
                    "published_at": "",  # Metadata will be fetched later
                    "channel_id": "N/A",
                    "channel_title": "N/A",
                    "subscriber_count": "",
                    "comment_count": "",
                    "relevance_score": 0,  # Will be calculated during export if needed
                    "type": "video"
                })

        return results


    def search_texts(self, keyword, useragents, debug=False):
        """Search for recent text articles"""
        # Create new DDGS instance with random User-Agent
        headers = {"User-Agent": random.choice(useragents)}
        ddgs = DDGS(headers=headers)

        text_results = ddgs.text(
            keywords=keyword,
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


    def search_trends(self, keyword, useragents, debug=False):
        """Search for recent videos and web content"""
        query = f"{keyword} site:youtube.com OR -inurl:(signup login)"
        if debug:
            st.write(t("trendwatcher_debug_query").format(query=query))

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
            # Search videos
            video_results = self.search_videos(keyword, useragents, valid_video_domains, debug=debug)

            # Search texts
            text_results = self.search_texts(keyword, useragents, debug=debug)

            results = video_results + text_results

            if debug:
                video_count = sum(1 for r in results if r.get("type") == "video")
                text_count = sum(1 for r in results if r.get("type") == "web")
                st.write(t("trendwatcher_debug_results").format(
                    count=len(results),
                    vids=video_count,
                    texts=text_count
                ))

            return results

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
            height=150
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
                keywords = [k.strip() for k in keywords_input.split("\n") if k.strip()]

                if debug_mode and len(keywords) > max_keywords_debug:
                    st.warning(
                        f"Debug mode: Limiting to first {max_keywords_debug} keywords: {', '.join(keywords[:max_keywords_debug])}"
                    )
                    keywords = keywords[:max_keywords_debug]

                all_results = []
                for keyword in keywords:
                    results = self.search_trends(keyword, useragents, debug=debug_mode)
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
                df["title_link"] = df.apply(lambda x: f"[{x['title'].replace('|','')}]({x['url']})", axis=1)
                # Add a selection column
                df["Select"] = False
                # Reorder columns to put Select first
                cols = ["Select", "title_link", "keyword", "language", "url"]  # Include url but don't display it
                df = df[cols]
                return df

            # Videos table
            selected_video_urls = []
            if video_results:
                st.subheader(t("trendwatcher_videos_table"))
                video_df = filter_df(video_results)
                if not video_df.empty:
                    # Add select all checkbox
                    st.session_state.select_all_videos = st.checkbox(
                        t("trendwatcher_select_all_videos"),
                        value=st.session_state.select_all_videos,
                        key="select_all_videos_checkbox"
                    )
                    # Update Select column based on select_all_videos
                    video_df["Select"] = st.session_state.select_all_videos
                    # Use st.data_editor for interactive selection
                    edited_video_df = st.data_editor(
                        video_df[["Select", "title_link", "keyword", "language"]],  # Exclude url from display
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "Select for Export",
                                help="Check to include this video in the export",
                                default=False
                            )
                        },
                        disabled=["title_link", "keyword", "language"],
                        hide_index=True,
                        key="video_selector"
                    )
                    # Merge edited selections back to original DataFrame to retain url
                    video_df.update(edited_video_df[["Select"]])
                    # Collect selected URLs
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
                    # Add select all checkbox
                    st.session_state.select_all_texts = st.checkbox(
                        t("trendwatcher_select_all_texts"),
                        value=st.session_state.select_all_texts,
                        key="select_all_texts_checkbox"
                    )
                    # Update Select column based on select_all_texts
                    text_df["Select"] = st.session_state.select_all_texts
                    # Use st.data_editor for interactive selection
                    edited_text_df = st.data_editor(
                        text_df[["Select", "title_link", "keyword", "language"]],  # Exclude url from display
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "Select for Export",
                                help="Check to include this article in the export",
                                default=False
                            )
                        },
                        disabled=["title_link", "keyword", "language"],
                        hide_index=True,
                        key="text_selector"
                    )
                    # Merge edited selections back to original DataFrame to retain url
                    text_df.update(edited_text_df[["Select"]])
                    # Collect selected URLs
                    selected_text_urls = text_df[text_df["Select"]]["url"].tolist()
                    if debug_mode:
                        st.write("Selected text URLs:", selected_text_urls)
                else:
                    st.info("No text articles match the filters.")

            if not video_results and not text_results:
                st.info(t("trendwatcher_no_results"))

            # Single save button
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
