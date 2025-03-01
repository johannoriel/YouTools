from app import Plugin
import streamlit as st
from typing import List, Dict, Any
from youtube_api import YoutubeAPI
import pandas as pd
from global_vars import translations, t
from plugins.ragllm import RagllmPlugin
import os
from youtube_db import *

# Add new translations for sorting functionality
translations["en"].update({
    "trendsyoutube_tab": "YouTube Trends",
    "trendsyoutube_subscriptions": "Subscription Analysis",
    "trendsyoutube_num_videos": "Number of videos per channel",
    "trendsyoutube_list_subs": "List Subscriptions",
    "trendsyoutube_loading": "Loading subscription data...",
    "trendsyoutube_error": "Error loading data",
    "trendsyoutube_select_all": "Select All",
    "trendsyoutube_deselect_all": "Deselect All",
    "trendsyoutube_selected_count": "Selected videos: {}",
    "trendsyoutube_no_data": "No subscription data available",
    "trendsyoutube_sort_by": "Sort by",
    "trendsyoutube_relevance_score": "Relevance Score",
    "trendsyoutube_source": "Video Source",
    "trendsyoutube_source_subscriptions": "Subscriptions",
    "trendsyoutube_source_trending": "Trending",
    "trendsyoutube_source_search": "Search by Keywords",
    "trendsyoutube_search_keywords": "Keywords",
    "trendsyoutube_search_order": "Search Order",
    "trendsyoutube_search_button": "Search Videos",
    "trendsyoutube_source_keywords_compare": "Keywords Comparison",
    "trendsyoutube_keywords_list": "Keywords List (comma separated)",
    "trendsyoutube_suggest_keywords": "Suggest Keywords",
    "trendsyoutube_theme_for_suggestions": "Theme for keyword suggestions",
    "trendsyoutube_add_to_targets": "Add to Targets",
})

translations["fr"].update({
    "trendsyoutube_tab": "Tendances YouTube",
    "trendsyoutube_subscriptions": "Analyse des Abonnements",
    "trendsyoutube_num_videos": "Nombre de vidéos par chaîne",
    "trendsyoutube_list_subs": "Lister les Abonnements",
    "trendsyoutube_loading": "Chargement des données d'abonnement...",
    "trendsyoutube_error": "Erreur lors du chargement des données",
    "trendsyoutube_select_all": "Tout Sélectionner",
    "trendsyoutube_deselect_all": "Tout Désélectionner",
    "trendsyoutube_selected_count": "Vidéos sélectionnées : {}",
    "trendsyoutube_no_data": "Aucune donnée d'abonnement disponible",
    "trendsyoutube_sort_by": "Trier par",
    "trendsyoutube_relevance_score": "Score de Pertinence",
    "trendsyoutube_source": "Source des vidéos",
    "trendsyoutube_source_subscriptions": "Abonnements",
    "trendsyoutube_source_trending": "Tendances",
    "trendsyoutube_source_search": "Recherche par Mots-clés",
    "trendsyoutube_search_keywords": "Mots-clés",
    "trendsyoutube_search_order": "Ordre de recherche",
    "trendsyoutube_search_button": "Rechercher des vidéos",
    "trendsyoutube_source_keywords_compare": "Comparatif de mots-clés",
    "trendsyoutube_keywords_list": "Liste de mots-clés (séparés par des virgules)",
    "trendsyoutube_suggest_keywords": "Suggérer des mots-clés",
    "trendsyoutube_theme_for_suggestions": "Thème pour les suggestions",
    "trendsyoutube_add_to_targets": "Ajouter aux Cibles"
})


class TrendsyoutubePlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'subscription_videos' not in st.session_state:
            st.session_state.subscription_videos = []
        if 'selected_videos' not in st.session_state:
            st.session_state.selected_videos = {}
        if 'sort_column' not in st.session_state:
            st.session_state.sort_column = 'relevance_score'
        if 'sort_ascending' not in st.session_state:
            st.session_state.sort_ascending = False
        # Nouvelle variable pour stocker le DataFrame trié
        if 'sorted_df' not in st.session_state:
            st.session_state.sorted_df = None

    def get_config_fields(self):
        """Define configuration fields for the plugin."""
        return {
            "max_videos_per_channel": {
                "type": "number",
                "label": "Maximum videos to fetch per channel",
                "default": 5
            },
            "max_subscriptions": {
                "type": "number",
                "label": "Maximum subscriptions to analyze",
                "default": 150
            }
        }

    def get_tabs(self):
        """Define plugin tabs."""
        return [{"name": t("trendsyoutube_tab"), "plugin": "trendsyoutube"}]

    def sort_videos(self, column_name: str) -> None:
        """
        Trie les vidéos et force un rafraîchissement immédiat.

        Args:
            column_name: Nom de la colonne sur laquelle trier
        """
        # Inverse l'ordre si on clique sur la même colonne
        if st.session_state.sort_column == column_name:
            st.session_state.sort_ascending = not st.session_state.sort_ascending
        else:
            st.session_state.sort_column = column_name
            st.session_state.sort_ascending = False

        # Crée un nouveau DataFrame à partir des vidéos
        base_columns = [
            'title', 'video_id', 'channel_title', 'channel_id', 'view_count', 'like_count',
            'comment_count', 'days_old', 'published_at', 'url', 'language', 'subscriber_count',
            'relevance_score'
        ]
        if st.session_state.video_source == t("trendsyoutube_source_keywords_compare"):
            base_columns.append('search_keyword')

        df = pd.DataFrame(st.session_state.subscription_videos,
                          columns=base_columns)

        # Applique le tri
        st.session_state.sorted_df = df.sort_values(
            by=st.session_state.sort_column,
            ascending=st.session_state.sort_ascending
        )

        # Force le rafraîchissement de Streamlit
        st.rerun()

    def load_subscriptions(self, max_videos: int) -> None:
        """
        Load subscription videos and store them in session state.

        Args:
            max_videos: Maximum number of videos to fetch per channel
        """
        youtube_api = YoutubeAPI(self.plugin_manager.config)

        with st.spinner(t("trendsyoutube_loading")):
            # Get subscriptions
            subscriptions = youtube_api.get_subscriptions(
                max_results=int(
                    self.plugin_manager.config['trendsyoutube']['max_subscriptions'])
            )

            # Fetch recent videos for each subscription
            all_videos = []
            for sub in subscriptions:
                channel_videos = youtube_api.get_channel_recent_videos(
                    sub['channel_id'],
                    max_results=max_videos
                )
                all_videos.extend(channel_videos)

            # Store in session state
            st.session_state.subscription_videos = all_videos
            st.session_state.selected_videos = {
                i: False for i in range(len(all_videos))}

    def load_trending_videos(self, max_videos: int) -> None:
        """
        Load trending videos and store them in session state.

        Args:
            max_videos: Maximum number of videos to fetch
        """
        youtube_api = YoutubeAPI(self.plugin_manager.config)

        with st.spinner("Loading trending videos..."):
            trending_videos = youtube_api.get_trending_videos(
                language=st.session_state.lang, max_results=max_videos)
            st.session_state.subscription_videos = trending_videos
            st.session_state.selected_videos = {
                i: False for i in range(len(trending_videos))}

    def search_videos(self, keywords: str, max_videos: int, order: str) -> None:
        """
        Search videos by keywords and store them in session state.

        Args:
            keywords: Keywords to search for
            max_videos: Maximum number of videos to fetch
            order: Order of search results (relevance, date, viewCount, rating)
        """
        youtube_api = YoutubeAPI(self.plugin_manager.config)

        with st.spinner("Searching videos..."):
            search_results = youtube_api.search_videos(
                keywords, max_videos, order=order, language=st.session_state.lang)

            all_videos = []
            for video in search_results:
                video['relevance_score'] = youtube_api.calculate_relevance_score(
                    video)
                all_videos.append(video)

            st.session_state.subscription_videos = all_videos
            st.session_state.selected_videos = {
                i: False for i in range(len(all_videos))}

    def recalculate_scores(self, youtube_api: YoutubeAPI) -> None:
        """
        Recalcule les scores de pertinence pour toutes les vidéos en utilisant
        la méthode standardisée de l'API YouTube.
        """
        for video in st.session_state.subscription_videos:
            video['relevance_score'] = youtube_api.calculate_relevance_score(
                video)

        # Force le rafraîchissement du DataFrame trié
        if st.session_state.sorted_df is not None:
            st.session_state.sorted_df = pd.DataFrame(st.session_state.subscription_videos).sort_values(
                by=st.session_state.sort_column,
                ascending=st.session_state.sort_ascending
            )
        st.rerun()

    def search_multiple_keywords(self, keywords_list: List[str], max_videos: int, order: str) -> None:
        """
        Search videos for multiple keywords and store them in session state.

        Args:
            keywords_list: List of keywords to search for
            max_videos: Maximum number of videos to fetch per keyword
            order: Order of search results
        """
        youtube_api = YoutubeAPI(self.plugin_manager.config)

        all_videos = []
        with st.spinner("Searching videos for multiple keywords..."):
            for keyword in keywords_list:
                search_results = youtube_api.search_videos(
                    keyword.strip(),
                    max_videos,
                    order=order,
                    language=st.session_state.lang
                )

                for video in search_results:
                    # Add keyword information
                    video['search_keyword'] = keyword.strip()
                    all_videos.append(video)

            st.session_state.subscription_videos = all_videos
            st.session_state.selected_videos = {
                i: False for i in range(len(all_videos))}

    def get_keyword_suggestions(self, theme: str, config: dict) -> List[str]:
        """
        Get keyword suggestions from LLM based on a theme.

        Args:
            theme: Theme to generate keywords for
            config: Configuration dictionary

        Returns:
            List of suggested keywords
        """
        prompt = f"""Please suggest 5-10 relevant YouTube search keywords for the theme: {theme}.
        Provide only the keywords, separated by commas."""

        context = ""  # You can add relevant context if needed

        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        llm_response = ragllm_plugin.process_with_llm(
            prompt,
            config.get('llm', {}).get('llm_sys_prompt', ''),
            context
        )

        # Clean and parse the response
        keywords = [k.strip() for k in llm_response.split(',')]
        return keywords

    def reset_session_state(self) -> None:
        """Reset display and results related session state variables."""
        st.session_state.subscription_videos = []
        st.session_state.selected_videos = {}
        st.session_state.sort_column = 'relevance_score'
        st.session_state.sort_ascending = False
        st.session_state.sorted_df = None
        if 'suggested_keywords' in st.session_state:
            del st.session_state.suggested_keywords

    def display_subscriptions(self) -> None:
        """Display subscription videos in a sortable table with selection controls."""
        if not st.session_state.subscription_videos:
            st.warning(t("trendsyoutube_no_data"))
            return

        youtube_api = YoutubeAPI(self.plugin_manager.config)

        base_columns = [
            'title', 'video_id', 'channel_title', 'channel_id', 'view_count', 'like_count',
            'comment_count', 'days_old', 'published_at', 'url', 'language', 'subscriber_count',
            'relevance_score'
        ]

        # Utilise le DataFrame trié s'il existe, sinon crée un nouveau
        if st.session_state.sorted_df is None:
            df = pd.DataFrame(st.session_state.subscription_videos)
            if 'search_keyword' in base_columns and 'search_keyword' not in df.columns:
                df['search_keyword'] = ''
            st.session_state.sorted_df = df.sort_values(
                by=st.session_state.sort_column,
                ascending=st.session_state.sort_ascending
            )

        df = st.session_state.sorted_df

        # Extraire la liste des langues disponibles
        available_languages = df['language'].unique().tolist()
        available_languages.sort()  # Trier les langues par ordre alphabétique

        # Ajouter un filtre par langue
        selected_languages = st.multiselect(
            "Filter by language",
            options=available_languages,
            default=available_languages  # Par défaut, toutes les langues sont sélectionnées
        )

        # Filtrer le DataFrame en fonction des langues sélectionnées
        if selected_languages:
            df = df[df['language'].isin(selected_languages)]

        available_channels = df['channel_title'].unique().tolist()
        available_channels.sort()
        selected_channels = st.multiselect(
            "Filter by channel",
            options=available_channels,
            default=available_channels
        )
        if selected_channels:
            df = df[df['channel_title'].isin(selected_channels)]

        if st.session_state.video_source == t("trendsyoutube_source_keywords_compare"):
            available_keywords = df['search_keyword'].unique().tolist()
            available_keywords.sort()
            selected_keywords = st.multiselect(
                "Filter by keyword",
                options=available_keywords,
                default=available_keywords
            )
            if selected_keywords:
                df = df[df['search_keyword'].isin(selected_keywords)]

        # Add selection controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button(t("trendsyoutube_select_all")):
                st.session_state.selected_videos = {i: True for i in range(
                    len(st.session_state.subscription_videos))}
        with col2:
            if st.button(t("trendsyoutube_deselect_all")):
                st.session_state.selected_videos = {i: False for i in range(
                    len(st.session_state.subscription_videos))}
        with col3:
            st.write(t("trendsyoutube_selected_count").format(
                sum(st.session_state.selected_videos.values())
            ))
        with col4:
            if st.button("Recalculate Scores"):
                self.recalculate_scores(youtube_api)

        # Définition des en-têtes de colonnes avec leur fonction de tri
        headers = [
            ("", None),
            ("Title", "title"),
            ("Channel", "channel_title"),
            ("Views", "view_count"),
            ("Comments", "comment_count"),
            ("Age", "days_old"),
            ("Subscribers", "subscriber_count"),
            ("Likes", "like_count"),
            ("Score", "relevance_score"),
            ("Lang", "language"),
            ("Action", None),
        ]

        if st.session_state.video_source == t("trendsyoutube_source_keywords_compare"):
            headers.append(("Keyword", "search_keyword"))
            cols = st.columns([0.5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        else:
            cols = st.columns([0.5, 3, 2, 1, 1, 1, 1, 1, 1, 1])

        # Affiche les en-têtes triables
        for col, (header, column_name) in zip(cols, headers):
            if column_name:  # Skip the checkbox column
                # Ajoute un indicateur de tri sur la colonne active
                sort_indicator = ""
                if st.session_state.sort_column == column_name:
                    sort_indicator = " ↑" if st.session_state.sort_ascending else " ↓"

                if col.button(f"{header}{sort_indicator}"):
                    self.sort_videos(column_name)

        # Display video rows
        for index, video in df.iterrows():
            with st.container():
                columns_width = [0.5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1]
                if st.session_state.video_source == t("trendsyoutube_source_keywords_compare"):
                    columns_width.insert(-1, 1)
                    cols = st.columns(columns_width)
                else:
                    cols = st.columns(columns_width)

                original_index = next(
                    (i for i, v in enumerate(st.session_state.subscription_videos)
                        if v['video_id'] == video['video_id']),
                    index
                )

                cols[0].checkbox("", key=f"video_{original_index}", value=st.session_state.selected_videos.get(
                    original_index, False), on_change=lambda i=original_index: self._update_selection(i))
                cols[1].markdown(f"[{video['title']}]({video['url']})")
                cols[2].write(video['channel_title'])
                cols[3].write(youtube_api.format_count(video['view_count']))
                cols[4].write(str(video['comment_count']))
                cols[5].write(f"{video['days_old']}d")
                cols[6].write(youtube_api.format_count(
                    video['subscriber_count']))
                cols[7].write(youtube_api.format_count(video['like_count']))
                cols[8].write(f"{video['relevance_score']:.1f}")
                cols[9].write(video['language'])
                if st.session_state.video_source == t("trendsyoutube_source_keywords_compare"):
                    cols[10].write(video.get('search_keyword', ''))
                    action_col = cols[11]
                else:
                    action_col = cols[10]

                # Bouton "Add to Targets"
                if action_col.button("Add to Targets", key=f"add_target_{video['channel_id']}_{index}"):
                    channel_info = youtube_api.get_channel_info(
                        video['channel_id'])
                    if channel_info:
                        channel_url = f"https://www.youtube.com/channel/{video['channel_id']}"
                        add_target_channel(
                            channel_id=video['channel_id'],
                            channel_title=video['channel_title'],
                            channel_url=channel_url,
                            keywords=[],  # Mots-clés initiaux vides, modifiables dans Channel Manager
                            subscriber_count=channel_info['subscriber_count']
                        )
                        st.success(
                            f"Channel '{video['channel_title']}' added to target channels!")

    def _update_selection(self, index: int):
        """Update video selection in session state."""
        st.session_state.selected_videos[index] = not st.session_state.selected_videos.get(
            index, False)

    def extract_comments(self, youtube_api: YoutubeAPI, work_directory: str, max_comments: int) -> None:
        """
        Extrait les commentaires des vidéos sélectionnées et les écrit dans un fichier.

        Args:
            youtube_api: Instance de YoutubeAPI
            work_directory: Répertoire de travail pour sauvegarder le fichier
            max_comments: Nombre maximum de commentaires à extraire par vidéo
        """
        selected_indices = [
            i for i, selected in st.session_state.selected_videos.items() if selected]
        if not selected_indices:
            st.warning("Please select at least one video to extract comments.")
            return

        # Créer le fichier comments.txt
        comments_file = os.path.join(work_directory, "comments.txt")
        comment_id = 1  # Pour générer des IDs uniques

        with st.spinner("Extracting comments..."):
            with open(comments_file, "w", encoding="utf-8") as f:
                for index in selected_indices:
                    video = st.session_state.subscription_videos[index]

                    # Écrire l'en-tête de la vidéo
                    header = f"""
=== VIDEO INFORMATION ===
Title: {video['title']}
Channel: {video['channel_title']}
URL: {video['url']}
Subscribers: {youtube_api.format_count(video['subscriber_count'])}
Views: {youtube_api.format_count(video['view_count'])}
Likes: {youtube_api.format_count(video['like_count'])}
Age: {video['days_old']} days
Video Language: {video.get('language', 'unknown')}

--- COMMENTS ---
"""
                    f.write(header)

                    # Récupérer et écrire les commentaires
                    comments = youtube_api.get_comments(
                        video_id=video['video_id'],
                        max_results=max_comments,
                        order="relevance"  # Trier par pertinence
                    )

                    for comment in comments:
                        comment_text = f"""
[COMMENT ID: {comment['id']}]
Author: {comment['author']}
Posted: {comment['published_at']}
Content: {comment['text']}
----------------------------------------
"""
                        f.write(comment_text)
                        comment_id += 1

                    f.write("\n\n")  # Séparateur entre les vidéos

        st.success(f"Comments extracted and saved to {comments_file}")

    def run(self, config):
        """Main plugin execution."""
        st.header(t("trendsyoutube_subscriptions"))
        # youtube_api = YoutubeAPI(config)
        # stats = youtube_api.get_quota_usage(config)
        # st.info(f"Quota utilisé : {stats['usage_percentage']}%")

        if st.button("Reset Results"):
            self.reset_session_state()
            st.rerun()

        # Sélection de la source des vidéos
        video_source = st.radio(
            t("trendsyoutube_source"),
            options=[
                t("trendsyoutube_source_subscriptions"),
                t("trendsyoutube_source_trending"),
                t("trendsyoutube_source_search"),
                t("trendsyoutube_source_keywords_compare")  # New option
            ],
            index=0
        )
        st.session_state.video_source = video_source

        if video_source == t("trendsyoutube_source_subscriptions"):
            max_videos = st.number_input(
                t("trendsyoutube_num_videos"),
                min_value=1,
                max_value=10,
                value=int(config['trendsyoutube']['max_videos_per_channel'])
            )

            if st.button(t("trendsyoutube_list_subs")):
                self.load_subscriptions(int(max_videos))

        elif video_source == t("trendsyoutube_source_trending"):
            max_videos = st.number_input(
                "Number of trending videos to fetch",
                min_value=1,
                max_value=50,
                value=10
            )

            if st.button("Load Trending Videos"):
                self.load_trending_videos(int(max_videos))

        elif video_source == t("trendsyoutube_source_search"):
            keywords = st.text_input(t("trendsyoutube_search_keywords"))
            max_videos = st.number_input(
                "Number of videos to fetch",
                min_value=1,
                max_value=50,
                value=10
            )
            order = st.selectbox(
                t("trendsyoutube_search_order"),
                options=["relevance", "date", "viewCount", "rating"],
                index=1
            )

            if st.button(t("trendsyoutube_search_button")):
                if keywords:
                    self.search_videos(keywords, int(max_videos), order)
                else:
                    st.warning("Please enter keywords to search.")

        elif video_source == t("trendsyoutube_source_keywords_compare"):
            # Add theme input and suggestion button
            col1, col2 = st.columns([3, 1])
            with col1:
                theme = st.text_input(t("trendsyoutube_theme_for_suggestions"))
            with col2:
                if st.button(t("trendsyoutube_suggest_keywords")) and theme:
                    suggested_keywords = self.get_keyword_suggestions(
                        theme, config)
                    st.session_state.suggested_keywords = ", ".join(
                        suggested_keywords)

            # Keywords input
            keywords = st.text_input(
                t("trendsyoutube_keywords_list"),
                value=getattr(st.session_state, 'suggested_keywords', '')
            )

            max_videos = st.number_input(
                "Number of videos to fetch per keyword",
                min_value=1,
                max_value=50,
                value=10
            )

            order = st.selectbox(
                t("trendsyoutube_search_order"),
                options=["relevance", "date", "viewCount", "rating"],
                index=1
            )

            if st.button(t("trendsyoutube_search_button")):
                if keywords:
                    keywords_list = [k.strip()
                                     for k in keywords.split(',') if k.strip()]
                    self.search_multiple_keywords(
                        keywords_list, int(max_videos), order)
                else:
                    st.warning("Please enter keywords to search.")

        if st.session_state.subscription_videos:
            self.display_subscriptions()

            # Ajouter les contrôles pour l'extraction des commentaires
            st.write("---")
            st.subheader("Comment Extraction")

            max_comments = st.number_input(
                "Number of comments to extract per video",
                min_value=1,
                max_value=50,
                value=10
            )

            if st.button("Extract Comments"):
                youtube_api = YoutubeAPI(config)
                work_directory = config['common']['work_directory']
                self.extract_comments(
                    youtube_api, work_directory, max_comments)
