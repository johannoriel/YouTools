from global_vars import translations, t
from app import Plugin
import streamlit as st
from youtube_api import YoutubeAPI
from youtube_db import *
from plugins.ragllm import RagllmPlugin
from typing import List, Dict, Any

translations["en"].update({
    "marketyoutube_tab_videos": "Videos Database",
    "marketyoutube_tab_stats": "Video Statistics",
    "marketyoutube_header_videos": "Manage Video Database",
    "marketyoutube_header_stats": "Video Statistics and Insights",
    "marketyoutube_sync": "Sync with YouTube",
    "marketyoutube_filter_label": "Filter by",
    "marketyoutube_filter_title": "Title",
    "marketyoutube_filter_title_desc": "Title + Description",
    "marketyoutube_filter_all": "Title + Description + Transcript",
    "marketyoutube_keyword": "Keyword",
    "marketyoutube_page": "Page",
    "marketyoutube_video_count": "Total Videos: {}",
    "marketyoutube_views": "Views",
    "marketyoutube_subscriber_gains": "Subscriber Gains",
    "marketyoutube_retention_rate": "Retention Rate (%)",
    "marketyoutube_avg_view_duration": "Avg View Duration (s)",
    "marketyoutube_syncing": "Syncing with YouTube...",
    "marketyoutube_sync_complete": "Sync completed successfully!",
    "marketyoutube_tab_campaigns": "Marketing Campaigns",
    "marketyoutube_header_campaigns": "Automated YouTube Marketing",
    "marketyoutube_select_video": "Select Video to Promote",
    "marketyoutube_keywords": "Keywords for Campaign",
    "marketyoutube_max_videos": "Max Videos to Search",
    "marketyoutube_max_comments": "Max Comments per Video",
    "marketyoutube_start_campaign": "Start Campaign",
    "marketyoutube_searching": "Searching for relevant videos...",
    "marketyoutube_fetching_comments": "Fetching comments...",
    "marketyoutube_generating_responses": "Generating responses...",
    "marketyoutube_validating": "Validate Responses",
    "marketyoutube_posting": "Posting responses...",
    "marketyoutube_campaign_complete": "Campaign completed successfully!",
    "marketyoutube_response_edit": "Edit Response for Comment {}",
    "marketyoutube_response_select": "Select to Post",
    "marketyoutube_progress": "Processing {}/{}",
    "marketyoutube_annotation_click_through_rate": "Annotation Click-Through Rate (%)",
    "marketyoutube_annotation_close_rate": "Annotation Close Rate (%)",
    "marketyoutube_comments": "Comments",
    "marketyoutube_dislikes": "Dislikes",
    "marketyoutube_estimated_minutes_watched": "Estimated Minutes Watched",
    "marketyoutube_estimated_revenue": "Estimated Revenue ($)",
    "marketyoutube_likes": "Likes",
    "marketyoutube_shares": "Shares",
    "marketyoutube_subscribers_lost": "Subscribers Lost",
    "marketyoutube_viewer_percentage": "Viewer Percentage (%)",
    "marketyoutube_average_view_percentage": "Average View Percentage (%)",
    "marketyoutube_audience_watch_ratio": "Audience Watch Ratio",
    "marketyoutube_relative_retention_performance": "Relative Retention Performance",
    "marketyoutube_estimated_ad_revenue": "Estimated Ad Revenue ($)",
})

translations["fr"].update({
    "marketyoutube_tab_videos": "Base de données des vidéos",
    "marketyoutube_tab_stats": "Statistiques des vidéos",
    "marketyoutube_header_videos": "Gérer la base de données des vidéos",
    "marketyoutube_header_stats": "Statistiques et analyses des vidéos",
    "marketyoutube_sync": "Synchroniser avec YouTube",
    "marketyoutube_filter_label": "Filtrer par",
    "marketyoutube_filter_title": "Titre",
    "marketyoutube_filter_title_desc": "Titre + Description",
    "marketyoutube_filter_all": "Titre + Description + Transcription",
    "marketyoutube_keyword": "Mot-clé",
    "marketyoutube_page": "Page",
    "marketyoutube_video_count": "Total des vidéos : {}",
    "marketyoutube_views": "Vues",
    "marketyoutube_subscriber_gains": "Abonnés gagnés",
    "marketyoutube_retention_rate": "Taux de rétention (%)",
    "marketyoutube_avg_view_duration": "Durée moyenne de visionnage (s)",
    "marketyoutube_syncing": "Synchronisation avec YouTube...",
    "marketyoutube_sync_complete": "Synchronisation terminée avec succès !",
    "marketyoutube_tab_campaigns": "Campagnes Marketing",
    "marketyoutube_header_campaigns": "Marketing Automatisé sur YouTube",
    "marketyoutube_select_video": "Sélectionner la vidéo à promouvoir",
    "marketyoutube_keywords": "Mots-clés pour la campagne",
    "marketyoutube_max_videos": "Nombre max de vidéos à rechercher",
    "marketyoutube_max_comments": "Nombre max de commentaires par vidéo",
    "marketyoutube_start_campaign": "Lancer la campagne",
    "marketyoutube_searching": "Recherche de vidéos pertinentes...",
    "marketyoutube_fetching_comments": "Récupération des commentaires...",
    "marketyoutube_generating_responses": "Génération des réponses...",
    "marketyoutube_validating": "Valider les réponses",
    "marketyoutube_posting": "Publication des réponses...",
    "marketyoutube_campaign_complete": "Campagne terminée avec succès !",
    "marketyoutube_response_edit": "Modifier la réponse pour le commentaire {}",
    "marketyoutube_response_select": "Sélectionner pour poster",
    "marketyoutube_progress": "Traitement {}/{}",
    "marketyoutube_annotation_click_through_rate": "Taux de clic sur annotations (%)",
    "marketyoutube_annotation_close_rate": "Taux de fermeture des annotations (%)",
    "marketyoutube_comments": "Commentaires",
    "marketyoutube_dislikes": "Dislikes",
    "marketyoutube_estimated_minutes_watched": "Minutes estimées regardées",
    "marketyoutube_estimated_revenue": "Revenus estimés ($)",
    "marketyoutube_likes": "Likes",
    "marketyoutube_shares": "Partages",
    "marketyoutube_subscribers_lost": "Abonnés perdus",
    "marketyoutube_viewer_percentage": "Pourcentage de spectateurs (%)",
    "marketyoutube_average_view_percentage": "Pourcentage moyen de visionnage (%)",
    "marketyoutube_audience_watch_ratio": "Ratio de visionnage de l'audience",
    "marketyoutube_relative_retention_performance": "Performance relative de rétention",
    "marketyoutube_estimated_ad_revenue": "Revenus publicitaires estimés ($)",
})


class MarketyoutubePlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        initialize_database()
        self.youtube_api = YoutubeAPI(self.plugin_manager.config)
        self.ragllm_plugin = self.plugin_manager.get_plugin('ragllm')
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'campaign_responses' not in st.session_state:
            st.session_state.campaign_responses = []
        if 'selected_responses' not in st.session_state:
            st.session_state.selected_responses = {}

    def get_config_fields(self):
        return {
            "campaign_keywords": {
                "type": "text",
                "label": "Default Campaign Keywords",
                "default": ""
            },
            "max_campaign_videos": {
                "type": "number",
                "label": "Default Max Videos for Campaign",
                "default": 10
            },
            "max_campaign_comments": {
                "type": "number",
                "label": "Default Max Comments per Video",
                "default": 2
            },
            "response_prompt": {
                "type": "textarea",
                "label": "LLM Prompt for Campaign Responses",
                "default": """Suggest a concise response (<500 chars) to this comment, promoting the video at {url} (mention it). Use a direct tone, as if you're the commenter, inspired by this transcript: {transcript}"""
            }
        }

    def get_tabs(self):
        return [
            {"name": t("marketyoutube_tab_videos"), "plugin": "marketyoutube"},
            {"name": t("marketyoutube_tab_stats"),
             "plugin": "marketyoutube"},
            {"name": t("marketyoutube_tab_campaigns"),
             "plugin": "marketyoutube"},
            {"name": "Debug Stats API", "plugin": "marketyoutube"}
        ]

    def display_videos(self, tab: str, filter_type: str, keyword: str, page: int):
        videos = get_videos(filter_type, keyword, page)
        total_videos = len(videos)

        st.write(t("marketyoutube_video_count").format(total_videos))

        if tab == t("marketyoutube_tab_videos"):
            for video in videos:
                col1, col2 = st.columns([1, 3])
                col1.image(video['thumbnail_url'], width=120)
                col2.markdown(f"[{video['title']}]({video['url']})")
                col2.write(f"Published: {video['published_at']}")
                col2.write(f"Status: {video['status']}")
        elif tab == t("marketyoutube_tab_stats"):
            stats_data = []
            for video in videos:
                latest_stats = get_latest_stats(video['video_id'])
                stats_data.append({
                    'Title': video['title'],
                    'URL': video['url'],
                    'Published': video['published_at'],
                    'Status': video['status'],
                    'Views': latest_stats['view_count'] if latest_stats else 0,
                    'Subscribers Gained': latest_stats['subscribers_gained'] if latest_stats else 0,
                    'Subscribers Lost': latest_stats['subscribers_lost'] if latest_stats else 0,
                    'Retention Rate (%)': latest_stats['retention_rate'] if latest_stats else 0.0,
                    'Avg View Duration (s)': latest_stats['avg_view_duration'] if latest_stats else 0.0,
                    'Average View Percentage (%)': latest_stats['average_view_percentage'] if latest_stats else 0.0,
                    'Annotation Click-Through Rate (%)': latest_stats['annotation_click_through_rate'] if latest_stats else 0.0,
                    'Annotation Close Rate (%)': latest_stats['annotation_close_rate'] if latest_stats else 0.0,
                    'Comments': latest_stats['comments'] if latest_stats else 0,
                    'Dislikes': latest_stats['dislikes'] if latest_stats else 0,
                    'Estimated Minutes Watched': latest_stats['estimated_minutes_watched'] if latest_stats else 0.0,
                    'Estimated Ad Revenue ($)': latest_stats['estimated_ad_revenue'] if latest_stats else 0.0,
                    'Likes': latest_stats['likes'] if latest_stats else 0,
                    'Shares': latest_stats['shares'] if latest_stats else 0
                })

            st.dataframe(
                stats_data,
                column_config={
                    'Title': st.column_config.TextColumn("Title"),
                    'URL': st.column_config.LinkColumn("URL", width="small"),
                    'Published': st.column_config.TextColumn("Published"),
                    'Status': st.column_config.TextColumn("Status"),
                    'Views': st.column_config.NumberColumn(t("marketyoutube_views")),
                    'Subscribers Gained': st.column_config.NumberColumn(t("marketyoutube_subscribers_gained")),
                    'Subscribers Lost': st.column_config.NumberColumn(t("marketyoutube_subscribers_lost")),
                    'Retention Rate (%)': st.column_config.NumberColumn(t("marketyoutube_retention_rate"), format="%.1f"),
                    'Avg View Duration (s)': st.column_config.NumberColumn(t("marketyoutube_avg_view_duration"), format="%.1f"),
                    'Average View Percentage (%)': st.column_config.NumberColumn(t("marketyoutube_average_view_percentage"), format="%.1f"),
                    'Annotation Click-Through Rate (%)': st.column_config.NumberColumn(t("marketyoutube_annotation_click_through_rate"), format="%.2f"),
                    'Annotation Close Rate (%)': st.column_config.NumberColumn(t("marketyoutube_annotation_close_rate"), format="%.2f"),
                    'Comments': st.column_config.NumberColumn(t("marketyoutube_comments")),
                    'Dislikes': st.column_config.NumberColumn(t("marketyoutube_dislikes")),
                    'Estimated Minutes Watched': st.column_config.NumberColumn(t("marketyoutube_estimated_minutes_watched"), format="%.1f"),
                    'Estimated Ad Revenue ($)': st.column_config.NumberColumn(t("marketyoutube_estimated_ad_revenue"), format="%.2f"),
                    'Likes': st.column_config.NumberColumn(t("marketyoutube_likes")),
                    'Shares': st.column_config.NumberColumn(t("marketyoutube_shares"))
                },
                use_container_width=True
            )

    def generate_campaign_responses(self, config, campaign_video: Dict[str, Any], comments: List[Dict[str, Any]]):
        responses = []
        total_comments = len(comments)
        progress_bar = st.progress(0)
        progress_text = st.empty()

        prompt = config['marketyoutube']['response_prompt'].format(
            url=campaign_video['url'],
            transcript=campaign_video['transcript']
        )

        for idx, comment in enumerate(comments):
            progress = (idx + 1) / total_comments
            progress_bar.progress(progress)
            progress_text.text(
                t("marketyoutube_progress").format(idx + 1, total_comments))

            comment_context = f"Comment by {comment['author']} on {comment['video_title']} from {comment['channel_title']}:\n{comment['text']}"
            try:
                llm_response = self.ragllm_plugin.process_with_llm(
                    prompt,
                    config.get('ragllm', {}).get('llm_sys_prompt', ''),
                    comment_context
                )
                clean_response = llm_response.strip()
                if clean_response.startswith('"') and clean_response.endswith('"'):
                    clean_response = clean_response[1:-1]
                responses.append({
                    'comment_id': comment['id'],
                    'response': clean_response,
                    'target_video_id': comment['video_id'],
                    'comment_text': comment['text']
                })
            except Exception as e:
                responses.append({
                    'comment_id': comment['id'],
                    'response': f"Error: {str(e)}",
                    'target_video_id': comment['video_id'],
                    'comment_text': comment['text']
                })

        progress_bar.empty()
        progress_text.empty()
        return responses

    def run(self, config):
        tab1, tab2, tab3, tab4 = st.tabs([t("marketyoutube_tab_videos"), t(
            "marketyoutube_tab_stats"), t("marketyoutube_tab_campaigns"), "Debug Stats API"])

        filter_options = {
            t("marketyoutube_filter_title"): "title",
            t("marketyoutube_filter_title_desc"): "title_description",
            t("marketyoutube_filter_all"): "all"
        }

        # Tab 1: Videos
        with tab1:
            st.header(t("marketyoutube_header_videos"))
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("marketyoutube_sync")):
                    with st.spinner(t("marketyoutube_syncing")):
                        sync_videos(config['common']
                                    ['channel_id'], self.youtube_api)
                        st.success(t("marketyoutube_sync_complete"))
            with col2:
                if st.button("Reset Database Structure"):
                    with st.spinner("Resetting database..."):
                        reset_database()
                        st.success("Database structure reset successfully!")

            filter_type = st.selectbox(
                t("marketyoutube_filter_label"),
                options=list(filter_options.keys()),
                key="filter_type_videos"
            )
            keyword = st.text_input(
                t("marketyoutube_keyword"), key="keyword_videos")
            page = st.number_input(
                t("marketyoutube_page"), min_value=1, value=1, key="page_videos")
            self.display_videos(t("marketyoutube_tab_videos"),
                                filter_options[filter_type], keyword, page)

        # Tab 2: Stats (inchangé sauf ajout du statut dans le tableau)
        with tab2:
            st.header(t("marketyoutube_header_stats"))
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("marketyoutube_sync"), key="sync_stats"):
                    with st.spinner(t("marketyoutube_syncing")):
                        progress_bar = st.progress(0)

                        def update_progress(progress):
                            progress_bar.progress(progress)
                        self.youtube_api.sync_stats(
                            config['common']['channel_id'], update_progress)
                        progress_bar.empty()
                        st.success(t("marketyoutube_sync_complete"))
            with col2:
                timestamps = get_stats_snapshots_timestamps()
                if timestamps:
                    selected_timestamp = st.selectbox(
                        "Select Snapshot to Delete", timestamps)
                    if st.button("Delete Snapshot", key="delete_snapshot"):
                        delete_stats_snapshot(selected_timestamp)
                        st.success(
                            f"Snapshot at {selected_timestamp} deleted!")
                else:
                    st.write("No snapshots available.")

            filter_type = st.selectbox(
                t("marketyoutube_filter_label"),
                options=list(filter_options.keys()),
                key="filter_type_stats"
            )
            keyword = st.text_input(
                t("marketyoutube_keyword"), key="keyword_stats")
            page = st.number_input(
                t("marketyoutube_page"), min_value=1, value=1, key="page_stats")
            self.display_videos(t("marketyoutube_tab_stats"),
                                filter_options[filter_type], keyword, page)

        # Tab 3: Campaigns
        with tab3:
            st.header(t("marketyoutube_header_campaigns"))

            # Select video to promote
            videos = get_videos()
            video_options = {
                f"{v['title']} ({v['published_at']})": v for v in videos}
            selected_video_title = st.selectbox(
                t("marketyoutube_select_video"),
                options=list(video_options.keys())
            )
            campaign_video = video_options[selected_video_title]

            # Campaign parameters
            keywords = st.text_input(
                t("marketyoutube_keywords"),
                value=config['marketyoutube']['campaign_keywords']
            )
            max_videos = st.number_input(
                t("marketyoutube_max_videos"),
                min_value=1,
                max_value=50,
                value=int(config['marketyoutube']['max_campaign_videos'])
            )
            max_comments = st.number_input(
                t("marketyoutube_max_comments"),
                min_value=1,
                max_value=10,
                value=int(config['marketyoutube']['max_campaign_comments'])
            )

            if st.button(t("marketyoutube_start_campaign")):
                with st.spinner(t("marketyoutube_searching")):
                    # Step 1: Search for relevant videos
                    target_videos = self.youtube_api.search_videos(
                        keywords, max_videos)

                with st.spinner(t("marketyoutube_fetching_comments")):
                    # Step 2: Fetch comments
                    comments = []
                    for video in target_videos:
                        video_comments = self.youtube_api.get_comments(
                            video['video_id'], max_comments)
                        for comment in video_comments:
                            comment['video_title'] = video['title']
                            comment['channel_title'] = video['channel_title']
                            cache_campaign_data(
                                campaign_video['video_id'], video['video_id'], comment['id'], comment['text'])
                        comments.extend(video_comments)

                with st.spinner(t("marketyoutube_generating_responses")):
                    # Step 3: Generate responses
                    st.session_state.campaign_responses = self.generate_campaign_responses(
                        config, campaign_video, comments)
                    for resp in st.session_state.campaign_responses:
                        cache_campaign_data(campaign_video['video_id'], resp['target_video_id'],
                                            resp['comment_id'], resp['comment_text'], resp['response'], "pending")
                    st.session_state.selected_responses = {
                        i: False for i in range(len(st.session_state.campaign_responses))}

            # Step 4: Display and validate responses
            if st.session_state.campaign_responses:
                st.subheader(t("marketyoutube_validating"))
                for i, resp in enumerate(st.session_state.campaign_responses):
                    st.write(f"Comment: {resp['comment_text']}")
                    edited_response = st.text_area(
                        t("marketyoutube_response_edit").format(i + 1),
                        value=resp['response'],
                        key=f"resp_edit_{i}"
                    )
                    st.session_state.campaign_responses[i]['response'] = edited_response
                    st.session_state.selected_responses[i] = st.checkbox(
                        t("marketyoutube_response_select"),
                        value=st.session_state.selected_responses[i],
                        key=f"resp_select_{i}"
                    )
                    cached_data = get_campaign_data(
                        campaign_video['video_id'], "pending")
                    if cached_data:
                        for cache in cached_data:
                            if cache['comment_id'] == resp['comment_id']:
                                update_campaign_response(
                                    cache['cache_id'], edited_response)

                # Step 5: Post validated responses
                if st.button(t("marketyoutube_posting")):
                    with st.spinner(t("marketyoutube_posting")):
                        selected_responses = [r for i, r in enumerate(
                            st.session_state.campaign_responses) if st.session_state.selected_responses[i]]
                        for resp in selected_responses:
                            try:
                                self.youtube_api.post_comment_reply(
                                    resp['comment_id'], resp['response'])
                                cached_data = get_campaign_data(
                                    campaign_video['video_id'], "validated")
                                for cache in cached_data:
                                    if cache['comment_id'] == resp['comment_id']:
                                        mark_campaign_posted(cache['cache_id'])
                            except Exception as e:
                                st.error(f"Error posting response: {str(e)}")
                        st.success(t("marketyoutube_campaign_complete"))

        # Tab 4: Debug Stats API
        with tab4:
            st.header("Debug YouTube Analytics API")

            videos = get_videos(page=1, per_page=1)
            if not videos:
                st.warning("No videos in database. Please sync videos first.")
            else:
                last_video = videos[0]
                st.write(
                    f"Testing on video: **{last_video['title']}** (ID: {last_video['video_id']})")

                available_metrics = [
                    "annotationClickThroughRate", "annotationCloseRate", "averageViewDuration",
                    "averageViewPercentage", "comments", "dislikes", "estimatedMinutesWatched",
                    "estimatedAdRevenue", "likes", "shares", "subscribersGained", "subscribersLost",
                    "views"
                ]

                selected_metrics = []
                st.write("Select metrics to fetch:")
                for metric in available_metrics:
                    if st.checkbox(metric, key=f"metric_{metric}"):
                        selected_metrics.append(metric)

                if st.button("Fetch Debug Stats"):
                    if not selected_metrics:
                        st.warning("Please select at least one metric.")
                    else:
                        with st.spinner("Fetching debug stats..."):
                            result = self.youtube_api.debug_advanced_video_stats(
                                last_video['video_id'], selected_metrics)
                            if result["error"]:
                                st.error(f"Error: {result['error']}")
                            else:
                                st.write("**Analytics Response:**")
                                st.json(result["analytics_response"])
                                st.write("**Basic Stats (from videos.list):**")
                                st.json(result["basic_stats"])
                                if 'calculated_retention_rate' in result:
                                    st.write(
                                        f"**Calculated Retention Rate:** {result['calculated_retention_rate']:.1f}%")
