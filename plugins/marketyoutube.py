from global_vars import translations, t
from app import Plugin
import streamlit as st
from youtube_api import YoutubeAPI
from youtube_db import *
from plugins.ragllm import RagllmPlugin
from typing import List, Dict, Any
from plugins.promoteyoutube import PromoteyoutubePlugin

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
    "marketyoutube_subscribers_gained": "Subscribers gained",
    "marketyoutube_subscribers_lost": "Subscribers Lost",
    "marketyoutube_viewer_percentage": "Viewer Percentage (%)",
    "marketyoutube_average_view_percentage": "Average View Percentage (%)",
    "marketyoutube_audience_watch_ratio": "Audience Watch Ratio",
    "marketyoutube_relative_retention_performance": "Relative Retention Performance",
    "marketyoutube_estimated_ad_revenue": "Estimated Ad Revenue ($)",
    "marketyoutube_tab_channel_manager": "Channel Manager",
    "marketyoutube_add_channel": "Add New Target Channel",
    "marketyoutube_channel_url": "Channel URL",
    "marketyoutube_initial_keywords": "Initial Keywords (comma-separated)",
    "marketyoutube_target_channels": "Target Channels",
    "marketyoutube_update_keywords": "Update Keywords",
    "marketyoutube_delete_channel": "Delete Channel",
    "marketyoutube_keywords": "Keywords",
    "marketyoutube_edit_keywords": "Edit Keywords",
    "marketyoutube_suggest_keywords": "Suggest Keywords",
    "marketyoutube_filter_keywords": "Filter by Keywords",
    "marketyoutube_title": "Title",
    "marketyoutube_url": "URL",
    "marketyoutube_published": "Published",
    "marketyoutube_status": "Status",
    "marketyoutube_sync_transcripts": "Sync Transcripts",
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
    "marketyoutube_retention_rate": "Rétention%",
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
    "marketyoutube_subscribers_gained": "Abonnés gagnés",
    "marketyoutube_subscribers_lost": "Abonnés perdus",
    "marketyoutube_viewer_percentage": "Pourcentage de spectateurs (%)",
    "marketyoutube_average_view_percentage": "Pourcentage moyen de visionnage (%)",
    "marketyoutube_audience_watch_ratio": "Ratio de visionnage de l'audience",
    "marketyoutube_relative_retention_performance": "Performance relative de rétention",
    "marketyoutube_estimated_ad_revenue": "Revenus publicitaires estimés ($)",
    "marketyoutube_tab_channel_manager": "Gestion des Chaînes",
    "marketyoutube_add_channel": "Ajouter une Nouvelle Chaîne Cible",
    "marketyoutube_channel_url": "URL de la Chaîne",
    "marketyoutube_initial_keywords": "Mots-clés Initiaux (séparés par des virgules)",
    "marketyoutube_target_channels": "Chaînes Cibles",
    "marketyoutube_update_keywords": "Mettre à Jour les Mots-clés",
    "marketyoutube_delete_channel": "Supprimer la Chaîne",
    "marketyoutube_keywords": "Mots-clés",
    "marketyoutube_edit_keywords": "Modifier les mots-clés",
    "marketyoutube_suggest_keywords": "Suggérer des mots-clés",
    "marketyoutube_filter_keywords": "Filtrer par mots-clés",
    "marketyoutube_title": "Titre",
    "marketyoutube_url": "URL",
    "marketyoutube_published": "Publié",
    "marketyoutube_status": "Statut",
    "marketyoutube_sync_transcripts": "Synchroniser les transcripts",
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
            {"name": "Channel Manager", "plugin": "marketyoutube"},
            {"name": "Debug Stats API", "plugin": "marketyoutube"}
        ]

    def format_count(self, count: int) -> str:
        """Formate un nombre en K/M si > 1000."""
        if count >= 1_000_000:
            return f"{count/1_000_000:.1f}M"
        elif count >= 1000:
            return f"{count/1000:.1f}K"
        return str(count)

    def suggest_keywords(self, title: str, description: str, transcript: str) -> List[str]:
        """Suggère des mots-clés via LLM."""
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        prompt = """
        Suggest 5-10 relevant keywords for a YouTube video based on the following:
        Title: {title}
        Description: {description}
        Transcript: {transcript}
        Return the keywords as a comma-separated list.
        """
        context = f"Title: {title}\nDescription: {description}\nTranscript: {transcript}"
        llm_response = ragllm_plugin.process_with_llm(
            prompt.format(title=title, description=description,
                          transcript=transcript),
            "",
            context
        )
        return [kw.strip() for kw in llm_response.split(",")]

    def display_video_database(self, config, filter_type: str, keyword: str, page: int, keyword_filter: List[str] = None):
        videos = get_videos(filter_type, keyword, page,
                            keyword_filter=keyword_filter)
        total_videos = len(videos)

        st.write(t("marketyoutube_video_count").format(total_videos))

        for video in videos:
            keywords_str = ", ".join(
                video['keywords']) if video['keywords'] else "--"
            with st.expander(f"{video['title']} ({keywords_str})"):
                col1, col2 = st.columns([1, 3])
                col1.image(video['thumbnail_url'], width=120)
                col2.markdown(f"[{video['title']}]({video['url']})")
                col2.write(f"Published: {video['published_at']}")
                col2.write(f"Status: {video['status']}")
                transcript = get_video_transcript(video['video_id'])
                if transcript:
                    st.text_area("Transcription de la vidéo sélectionnée",
                                 value=transcript, key=f"video_transcript_{video['video_id']}", height=150, disabled=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(t("Copy"), key=f"video_transcript_copy_{video['video_id']}"):
                            st.code(transcript)
                    with col2:
                        st.download_button(
                            label=t("Download"),
                            key=f"video_transcript_download_{video['video_id']}",
                            data=transcript,
                            file_name=f"transcript_{video['video_id']}.txt",
                            mime="text/plain"
                        )
                else:
                    if st.button("Transcript>>>", key=f"get_transcript_{video['video_id']}"):
                        transcript, lang = self.youtube_api.get_transcript(
                            video['video_id'], config['common']['language'])
                        save_transcript(video['video_id'], transcript)
                        st.success("Transcript saved.")

                current_keywords = ", ".join(
                    video['keywords']) if video['keywords'] else "No keywords"
                col2.write(
                    f"{t('marketyoutube_keywords')}: {current_keywords}")

                new_keywords = st.text_input(
                    t("marketyoutube_edit_keywords"),
                    value=current_keywords,
                    key=f"edit_keywords_{video['video_id']}"
                )
                if st.button(t("marketyoutube_edit_keywords"), key=f"save_keywords_{video['video_id']}"):
                    updated_keywords = [
                        kw.strip() for kw in new_keywords.split(",") if kw.strip()]
                    update_video_keywords(video['video_id'], updated_keywords)
                    st.success(f"Keywords updated for {video['title']}")
                    st.rerun()

                if st.button(t("marketyoutube_suggest_keywords"), key=f"suggest_keywords_{video['video_id']}"):
                    suggested_keywords = self.suggest_keywords(
                        video['title'], video['description'], video['transcript'])
                    update_video_keywords(
                        video['video_id'], suggested_keywords)
                    st.success(
                        f"Suggested keywords applied for {video['title']}")
                    st.rerun()

    def display_video_stats(self, filter_type: str, keyword: str, keyword_filter: List[str] = None):
        from datetime import datetime  # Importer datetime pour formater la date

        videos = get_videos(filter_type, keyword,
                            keyword_filter=keyword_filter)
        total_videos = len(videos)

        st.write(t("marketyoutube_video_count").format(total_videos))

        stats_data = []
        advanced_stats_list = self.youtube_api.get_advanced_stats_list()
        for video in videos:
            latest_stats = get_latest_stats(video['video_id'])
            keywords_str = ", ".join(
                video['keywords']) if video['keywords'] else "--"
            # Formater la date pour n'afficher que le jour (YYYY-MM-DD)
            published_date = datetime.strptime(
                video['published_at'], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d") if video['published_at'] else "--"
            row = {
                t("marketyoutube_title"): video['title'],
                t("marketyoutube_url"): video['url'],
                t("marketyoutube_published"): published_date,  # Uniquement la date
                t("marketyoutube_status"): video['status'],
                t("marketyoutube_keywords"): keywords_str,
                t("marketyoutube_views"): latest_stats['view_count'] if latest_stats else 0,
                t("marketyoutube_retention_rate"): latest_stats['retention_rate'] if latest_stats else 0.0,
            }
            if latest_stats and 'advanced_stats' in latest_stats:
                for stat in advanced_stats_list:
                    translation_key = f"marketyoutube_{stat.lower()}"
                    label = t(translation_key) if translation_key in translations["en"] else stat.replace(
                        "Rate", " Rate (%)")
                    value = latest_stats['advanced_stats'].get(stat, 0)
                    row[label] = value
            else:
                for stat in advanced_stats_list:
                    translation_key = f"marketyoutube_{stat.lower()}"
                    label = t(translation_key) if translation_key in translations["en"] else stat.replace(
                        "Rate", " Rate (%)")
                    row[label] = 0
            stats_data.append(row)

        column_config = {
            t("marketyoutube_title"): st.column_config.TextColumn(
                t("marketyoutube_title"), width="small"),  # Réduction de la largeur
            t("marketyoutube_url"): st.column_config.LinkColumn(
                t("marketyoutube_url"), width="small"),
            t("marketyoutube_published"): st.column_config.TextColumn(
                t("marketyoutube_published")),
            t("marketyoutube_status"): st.column_config.TextColumn(
                t("marketyoutube_status")),
            t("marketyoutube_keywords"): st.column_config.TextColumn(
                t("marketyoutube_keywords")),
            t("marketyoutube_views"): st.column_config.NumberColumn(
                t("marketyoutube_views")),
            t("marketyoutube_retention_rate"): st.column_config.NumberColumn(
                t("marketyoutube_retention_rate"), format="%.1f"),
        }
        for stat in advanced_stats_list:
            translation_key = f"marketyoutube_{stat.lower()}"
            label = t(translation_key) if translation_key in translations["en"] else stat.replace(
                "Rate", " Rate (%)")
            format_str = "%.2f" if "Rate" in stat or "Percentage" in stat else "%.1f" if stat == "estimatedMinutesWatched" else "%.2f" if stat == "estimatedAdRevenue" else None
            column_config[label] = st.column_config.NumberColumn(
                label, format=format_str)

        st.dataframe(stats_data, column_config=column_config,
                     use_container_width=True)

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

    def sync_stats(self, channel_id: str, progress_callback=None):
        """Sync stats for all videos with progress callback."""
        videos = self.youtube_api.get_channel_videos(channel_id)
        total_videos = len(videos)
        timestamp = datetime.now(pytz.UTC).isoformat()

        for i, video in enumerate(videos):
            stats = self.youtube_api.get_advanced_video_stats(
                video['video_id'])
            if stats:
                insert_stats_snapshot(video['video_id'], timestamp, stats)
            if progress_callback:
                progress_callback((i + 1) / total_videos)

    def sync_transcripts(self, channel_id: str, youtube_api, config):
        """Synchronise les transcripts pour toutes les vidéos du canal qui n'en ont pas encore."""
        videos = get_videos()  # Récupère toutes les vidéos de la base
        total_videos = len(videos)
        processed = 0
        successes = 0
        errors = []

        with st.spinner(t("marketyoutube_syncing")):
            # Créer la barre de progression une seule fois avant la boucle
            progress_bar = st.progress(0)

            for video in videos:
                # Vérifie si le transcript est vide ou inexistant
                current_transcript = get_video_transcript(video['video_id'])
                if not current_transcript:
                    try:
                        transcript, lang = youtube_api.get_transcript(
                            video['video_id'],
                            config['common']['language']
                        )
                        if transcript:
                            save_transcript(video['video_id'], transcript)
                            successes += 1
                        else:
                            errors.append(
                                f"{video['title']} ({video['video_id']}): No transcript available")
                    except Exception as e:
                        error_msg = f"{video['title']} ({video['video_id']}): {str(e)}"
                        errors.append(error_msg)
                        # Optionnel : pour debug, tu peux afficher chaque erreur immédiatement
                        # st.warning(error_msg)

                processed += 1
                # Mettre à jour la barre existante
                progress_bar.progress(processed / total_videos)

            # Nettoyer la barre de progression
            progress_bar.empty()

            # Afficher un résumé des résultats
            if total_videos > 0:
                st.success(t("marketyoutube_sync_complete"))
                st.write(
                    f"Transcripts synchronisés avec succès : {successes}/{total_videos}")
                if errors:
                    with st.expander("Détails des erreurs"):
                        for error in errors:
                            st.write(error)
            else:
                st.info("Aucune vidéo à synchroniser.")

    def display_channel_manager(self, config):
        youtube_api = YoutubeAPI(self.plugin_manager.config)
        st.header("Channel Manager")

        # Section pour ajouter une chaîne manuellement
        st.subheader("Add New Target Channel")
        channel_url = st.text_input(
            "Channel URL (e.g., https://www.youtube.com/@ChannelName)", "")
        initial_keywords = st.text_input(
            "Initial Keywords (comma-separated)", "")

        if st.button("Add Channel"):
            if channel_url:
                try:
                    if "/@" in channel_url:
                        handle = channel_url.split("/@")[1].split("/")[0]
                        request = youtube_api.youtube.channels().list(
                            part="id,snippet,statistics",
                            forHandle=handle
                        )
                    elif "/channel/" in channel_url:
                        channel_id = channel_url.split(
                            "/channel/")[1].split("/")[0]
                        request = youtube_api.youtube.channels().list(
                            part="id,snippet,statistics",
                            id=channel_id
                        )
                    else:
                        st.error("Invalid channel URL format.")
                        return

                    response = request.execute()
                    if response['items']:
                        channel = response['items'][0]
                        channel_id = channel['id']
                        channel_title = channel['snippet']['title']
                        subscriber_count = int(
                            channel['statistics'].get('subscriberCount', 0))
                        keywords = [k.strip() for k in initial_keywords.split(
                            ",")] if initial_keywords else []

                        add_target_channel(
                            channel_id, channel_title, channel_url, keywords, subscriber_count)
                        st.success(
                            f"Channel '{channel_title}' added successfully!")
                    else:
                        st.error("Channel not found.")
                except Exception as e:
                    st.error(f"Error adding channel: {str(e)}")
            else:
                st.warning("Please provide a channel URL.")

        # Liste des chaînes cibles
        st.subheader("Target Channels")
        channels = get_target_channels()
        if not channels:
            st.info("No target channels added yet.")
        else:
            for channel in channels:
                # Ajouter les mots-clés entre parenthèses et formater le nombre d'abonnés
                keywords_str = ", ".join(
                    channel['keywords']) if channel['keywords'] else "--"
                subscriber_count_str = self.format_count(
                    channel['subscriber_count'])
                with st.expander(f"{channel['channel_title']} ({subscriber_count_str} subscribers) ({keywords_str}) "):
                    st.write(f"URL: {channel['channel_url']}")
                    st.write(f"Added: {channel['added_at']}")
                    st.write(f"Last Updated: {channel['last_updated']}")

                    current_keywords = ", ".join(channel['keywords'])
                    new_keywords = st.text_input(
                        f"Keywords for {channel['channel_title']}",
                        value=current_keywords,
                        key=f"keywords_{channel['channel_id']}"
                    )
                    if st.button("Update Keywords", key=f"update_{channel['channel_id']}"):
                        updated_keywords = [k.strip()
                                            for k in new_keywords.split(",")]
                        update_target_channel_keywords(
                            channel['channel_id'], updated_keywords)
                        st.success(
                            f"Keywords updated for {channel['channel_title']}!")

                    if st.button("Delete Channel", key=f"delete_{channel['channel_id']}"):
                        delete_target_channel(channel['channel_id'])
                        st.success(
                            f"Channel '{channel['channel_title']}' deleted!")
                        st.rerun()

    def display_campaign_tab(self, config, tab):
        with tab:
            st.header(t("marketyoutube_header_campaigns"))

            target_source = st.radio(
                "Target Source",
                options=["Search by Keywords", "Target Channels"],
                index=0,
                key="campaign_target_source"
            )

            # Récupérer toutes les vidéos
            videos = get_videos()

            # Ajouter un champ de recherche pour filtrer les vidéos à promouvoir
            search_keyword = st.text_input(
                "Search video by keyword",
                value="",
                key="campaign_video_search_keyword",
                help="Enter a keyword to filter videos by title or keywords"
            )

            # Filtrer les vidéos en fonction du mot-clé saisi
            if search_keyword:
                search_keyword = search_keyword.lower()
                filtered_videos = [
                    v for v in videos
                    if search_keyword in v['title'].lower() or
                    any(search_keyword in kw.lower() for kw in v['keywords'])
                ]
            else:
                filtered_videos = videos

            # Créer les options pour le selectbox avec titre, date et mots-clés
            video_options = {
                f"{v['title']} ({v['published_at']}) ({', '.join(v['keywords']) if v['keywords'] else '--'})": v
                for v in filtered_videos
            }

            # Si aucune vidéo ne correspond au filtre, afficher un message
            if not video_options:
                st.warning("No videos match your search keyword.")
                selected_video_title = None
                campaign_video = None
            else:
                selected_video_title = st.selectbox(
                    t("marketyoutube_select_video"),
                    options=list(video_options.keys()),
                    key="campaign_select_video"
                )
                campaign_video = video_options.get(selected_video_title)

            if "campaign_target_videos" not in st.session_state:
                st.session_state["campaign_target_videos"] = []

            if target_source == "Search by Keywords":
                keywords = st.text_input(
                    t("marketyoutube_keywords"),
                    value=config['marketyoutube']['campaign_keywords'],
                    key="campaign_keywords_search"
                )
                max_videos = st.number_input(
                    t("marketyoutube_max_videos"),
                    min_value=1,
                    max_value=50,
                    value=int(config['marketyoutube']['max_campaign_videos']),
                    key="campaign_max_videos_search"
                )
            else:
                target_channels = get_target_channels()
                if not target_channels:
                    st.warning(
                        "No target channels available. Please add some in the Channel Manager tab.")
                    return

                all_keywords = set()
                for channel in target_channels:
                    all_keywords.update(channel['keywords'])
                all_keywords = sorted(list(all_keywords))

                selected_keywords = st.multiselect(
                    "Select Keywords to Filter Channels",
                    options=all_keywords,
                    key="campaign_keywords_filter"
                )

                filtered_channels = [
                    ch for ch in target_channels
                    if not selected_keywords or any(kw in ch['keywords'] for kw in selected_keywords)
                ]

                selected_channels = st.multiselect(
                    "Select Target Channels",
                    options=[
                        f"{ch['channel_title']} ({', '.join(ch['keywords']) if ch['keywords'] else '--'}) ({ch['subscriber_count']} subscribers)"
                        for ch in filtered_channels
                    ],
                    default=[
                        f"{ch['channel_title']} ({', '.join(ch['keywords']) if ch['keywords'] else '--'}) ({ch['subscriber_count']} subscribers)"
                        for ch in filtered_channels
                    ],
                    key="campaign_select_channels"
                )

                max_videos_per_channel = st.number_input(
                    "Max Videos per Channel",
                    min_value=1,
                    max_value=50,
                    value=5,
                    key="campaign_max_videos_per_channel"
                )

            max_comments = st.number_input(
                t("marketyoutube_max_comments"),
                min_value=1,
                max_value=10,
                value=int(config['marketyoutube']['max_campaign_comments']),
                key="campaign_max_comments"
            )

            if st.button(t("marketyoutube_start_campaign"), key="campaign_start_button"):
                if not campaign_video:
                    st.error("Please select a video to promote.")
                else:
                    with st.spinner(t("marketyoutube_searching")):
                        if target_source == "Search by Keywords":
                            target_videos = self.youtube_api.search_videos(
                                keywords, max_videos)
                        else:
                            target_videos = []
                            for channel in filtered_channels:
                                if f"{channel['channel_title']} ({', '.join(channel['keywords']) if channel['keywords'] else '--'}) ({channel['subscriber_count']} subscribers)" in selected_channels:
                                    channel_videos = self.youtube_api.get_channel_recent_videos(
                                        channel['channel_id'],
                                        max_results=max_videos_per_channel
                                    )
                                    target_videos.extend(channel_videos)
                        st.session_state["campaign_target_videos"] = target_videos
                        prefix = "campaign_"
                        if f"{prefix}videos" in st.session_state:
                            del st.session_state[f"{prefix}videos"]
                        if f"{prefix}original_order" in st.session_state:
                            del st.session_state[f"{prefix}original_order"]
                        if f"{prefix}selected_videos" in st.session_state:
                            del st.session_state[f"{prefix}selected_videos"]
                        if f"{prefix}selected_video_indices" in st.session_state:
                            del st.session_state[f"{prefix}selected_video_indices"]
                        if f"{prefix}comments" in st.session_state:
                            del st.session_state[f"{prefix}comments"]
                        if f"{prefix}selected_comments" in st.session_state:
                            del st.session_state[f"{prefix}selected_comments"]
                        if f"{prefix}generated_responses" in st.session_state:
                            del st.session_state[f"{prefix}generated_responses"]
                        if f"{prefix}selected_responses" in st.session_state:
                            del st.session_state[f"{prefix}selected_responses"]
                        if f"{prefix}campaign_id" in st.session_state:
                            del st.session_state[f"{prefix}campaign_id"]

            if st.session_state["campaign_target_videos"] and campaign_video:
                promoteyoutube = PromoteyoutubePlugin(
                    "promoteyoutube", self.plugin_manager)
                promoteyoutube.run_campaign(
                    config=config,
                    target_videos=st.session_state["campaign_target_videos"],
                    campaign_video=campaign_video,
                    max_comments=max_comments,
                    prefix="campaign_"
                )

    def run(self, config):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([t("marketyoutube_tab_videos"), t(
            "marketyoutube_tab_stats"), t("marketyoutube_tab_campaigns"), "Channel Manager", "Debug Stats API"])

        filter_options = {
            t("marketyoutube_filter_title"): "title",
            t("marketyoutube_filter_title_desc"): "title_description",
            t("marketyoutube_filter_all"): "all"
        }

        # Tab 1: Videos
        with tab1:
            st.header(t("marketyoutube_header_videos"))
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button(t("marketyoutube_sync")):
                    with st.spinner(t("marketyoutube_syncing")):
                        sync_videos(
                            config['common']['channel_id'], self.youtube_api)
                        st.success(t("marketyoutube_sync_complete"))
            with col2:
                if st.button(t("marketyoutube_sync_transcripts")):  # Nouveau bouton
                    self.sync_transcripts(
                        config['common']['channel_id'], self.youtube_api, config)
            with col3:
                if st.button("Reset Database Structure"):
                    with st.spinner("Resetting database..."):
                        reset_database()
                        st.success(
                            "Database structure reset successfully!")
            with col4:
                if st.button("Upgrade Database Structure"):
                    try:
                        auto_upgrade_database()
                        st.info("Database upgraded")
                    except Exception as e:
                        print(f"Database error: {str(e)}")

            filter_type = st.selectbox(
                t("marketyoutube_filter_label"),
                options=list(filter_options.keys()),
                key="filter_type_videos"
            )
            keyword = st.text_input(
                t("marketyoutube_keyword"),
                key="keyword_videos"
            )

            all_keywords = set()
            for video in get_videos():
                all_keywords.update(video['keywords'])
            all_keywords = sorted(list(all_keywords))
            selected_keyword_filter = st.multiselect(
                t("marketyoutube_filter_keywords"),
                options=all_keywords,
                key="keyword_filter_videos"
            )

            page = st.number_input(
                t("marketyoutube_page"),
                min_value=1,
                value=1,
                key="page_videos"
            )
            self.display_video_database(config,
                                        filter_options[filter_type],
                                        keyword,
                                        page,
                                        keyword_filter=selected_keyword_filter if selected_keyword_filter else None
                                        )

        # Tab 2: Stats
        with tab2:
            st.header(t("marketyoutube_header_stats"))
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("marketyoutube_sync"), key="sync_stats"):
                    with st.spinner(t("marketyoutube_syncing")):
                        progress_bar = st.progress(0)

                        def update_progress(progress):
                            progress_bar.progress(progress)
                        self.sync_stats(
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
            # Plus de pagination dans les stats
            self.display_video_stats(
                filter_options[filter_type],
                keyword,
                keyword_filter=selected_keyword_filter if selected_keyword_filter else None
            )

        # Tab 3: Campaigns
        self.display_campaign_tab(config, tab3)

        # Tab 4: Channel Manager
        with tab4:
            self.display_channel_manager(config)

            # Tab 5: Debug Stats API (inchangé)
        with tab5:
            st.header("Debug YouTube Analytics API")

            st.subheader("Gestion du Quota YouTube")
            st.write(
                "Vous pouvez vérifier l'usage réel du quota ici : [Google Cloud Console Quotas](https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas?hl=fr&inv=1&invt=AbrCIQ&pageState=(%22allQuotasTable%22%253A(%22c%22%253A%5B%22displayDimensions%22%5D)))")
            # Création d'une instance pour accéder à set_global_quota_usage
            youtube_api = YoutubeAPI(config)
            current_quota = youtube_api.get_quota_usage()['quota_usage']
            st.write(f"Quota estimé actuel : {current_quota} unités")
            forced_quota = st.number_input(
                "Forcer la valeur du quota utilisé (unités)",
                min_value=0,
                value=current_quota,
                step=1,
                key="forced_quota_usage"
            )
            if st.button("Mettre à jour le quota"):
                youtube_api.set_global_quota_usage(forced_quota)
                st.success(f"Quota global mis à jour à {forced_quota} unités")

            videos = get_videos(page=1, per_page=1)
            if not videos:
                st.warning("No videos in database. Please sync videos first.")
            else:
                last_video = videos[0]
                st.write(
                    f"Testing on video: **{last_video['title']}** (ID: {last_video['video_id']})")

                available_metrics = self.youtube_api.get_advanced_stats_list()
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
