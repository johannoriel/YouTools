from global_vars import translations, t
from app import Plugin
import streamlit as st
from youtube_api import YoutubeAPI
from youtube_db import *
from typing import List, Dict, Any, Optional
from datetime import datetime
import pytz
from langdetect import detect
from widgets.yt_responses import ResponseDBDisplayWidget

# Ajout des traductions spécifiques au plugin Automarket
translations["en"].update({
    "automarket_tab": "Automarket",
    "automarket_header": "Automated YouTube Marketing Campaign",
    "automarket_select_video": "Select Video to Promote",
    "automarket_comments_per_video": "Comments per Video",
    "automarket_min_subscribers": "Minimum Subscribers",
    "automarket_max_videos_per_keyword": "Max Videos per Keyword",
    "automarket_expiry_days": "Expiry Days",
    "automarket_view_threshold": "View Threshold",
    "automarket_trusted_channel_videos": "Videos from Trusted Channels",
    "automarket_start_campaign": "Start Campaign",
    "automarket_processing": "Processing campaign...",
    "automarket_generating_responses": "Generating reséponses...",
    "automarket_rejected_videos": "Rejected Videos (Log)",
    "automarket_responses": "Generated Responses",
    "automarket_post_responses": "Post Responses",
    "automarket_channel": "Channel",
    "automarket_video": "Video",
    "automarket_comment": "Comment",
    "automarket_response": "Response",
    "automarket_add_to_trusted": "Add to Trusted Channels",
    "automarket_exclude_response": "Exclude Response",
    "automarket_campaign_complete": "Campaign completed successfully!",
    "automarket_no_videos_with_keywords": "No videos with keywords found in the database.",
    "automarket_rejected_reason": "Rejected: {}",
    "automarket_keyword": "Keyword",
    "automarket_criterion": "Criterion",
    "automarket_expand_all": "Expand All",
    "automarket_collapse_all": "Collapse All",
    "automarket_debug_mode": "Debug Mode",
    "automarket_max_comments_debug": "Max Comments in Debug Mode",
    "automarket_excluded_comments": "Excluded Comments (Log)",
    "automarket_exclusion_keyword": "Exclusion Keyword",
    "automarket_quota_consumed": "Quota consumed during campaign: {units} units",
    "automarket_progress": "Progression...",
    "automarket_search_keywords": "Search by Keywords",
    "automarket_search_trusted": "Search in Trusted Channels",
    "automarket_reponse_to_comment": "Response to:",
    "automarket_posting": "Posting responses...",
    "automarket_select_all": "Select All",
    "automarket_deselect_all": "Deselect All",
    "monitor_trends_tab": "Monitor Trends",
    "monitor_trends_header": "Monitor YouTube Trends",
    "monitor_trends_select_keyword": "Select Campaign Keyword",
    "monitor_trends_search_criteria": "Search Criteria",
    "monitor_trends_time": "Recent Videos (Time)",
    "monitor_trends_relevant": "Relevant Videos",
    "monitor_trends_trusted": "Trusted Channels",
    "automarket_selected_videos": "Selected Videos (Log)",
    "automarket_moderated_responses": "Responses in moderation",
})

translations["fr"].update({
    "automarket_tab": "Automarket",
    "automarket_header": "Campagne Marketing Automatisée sur YouTube",
    "automarket_select_video": "Sélectionner la vidéo à promouvoir",
    "automarket_comments_per_video": "Commentaires par vidéo",
    "automarket_min_subscribers": "Abonnés minimum",
    "automarket_max_videos_per_keyword": "Vidéos max par mot-clé",
    "automarket_expiry_days": "Jours d'expiration",
    "automarket_view_threshold": "Seuil de vues",
    "automarket_trusted_channel_videos": "Vidéos des chaînes de confiance",
    "automarket_start_campaign": "Lancer la campagne",
    "automarket_processing": "Traitement de la campagne...",
    "automarket_generating_responses": "Génération des réponses...",
    "automarket_rejected_videos": "Vidéos rejetées (Log)",
    "automarket_responses": "Réponses générées",
    "automarket_post_responses": "Poster les réponses",
    "automarket_channel": "Chaîne",
    "automarket_video": "Vidéo",
    "automarket_comment": "Commentaire",
    "automarket_response": "Réponse",
    "automarket_add_to_trusted": "Ajouter aux chaînes de confiance",
    "automarket_exclude_response": "Exclure la réponse",
    "automarket_campaign_complete": "Campagne terminée avec succès !",
    "automarket_no_videos_with_keywords": "Aucune vidéo avec mots-clés trouvée dans la base de données.",
    "automarket_rejected_reason": "Rejetée : {}",
    "automarket_keyword": "Mot-clé",
    "automarket_criterion": "Critère",
    "automarket_expand_all": "Tout déplier",
    "automarket_collapse_all": "Tout replier",
    "automarket_debug_mode": "Mode Debug",
    "automarket_max_comments_debug": "Nombre max de commentaires en mode Debug",
    "automarket_excluded_comments": "Commentaires Exclus (Log)",
    "automarket_exclusion_keyword": "Mot-clé d'Exclusion",
    "automarket_quota_consumed": "Quota consommé pendant la campagne : {units} unités",
    "automarket_progress": "Progression...",
    "automarket_search_keywords": "Recherche par Mots-clés",
    "automarket_search_trusted": "Recherche dans les Chaînes de Confiance",
    "automarket_reponse_to_comment": "Réponse à:",
    "automarket_posting": "Post des réponses ...",
    "automarket_select_all": "Tout sélectionner",
    "automarket_deselect_all": "Tout désélectionner",
    "monitor_trends_tab": "Surveiller les Tendances",
    "monitor_trends_header": "Surveiller les Tendances YouTube",
    "monitor_trends_select_keyword": "Sélectionner un Mot-clé de Campagne",
    "monitor_trends_search_criteria": "Critères de Recherche",
    "monitor_trends_time": "Vidéos Récentes (Temps)",
    "monitor_trends_relevant": "Vidéos Pertinentes",
    "monitor_trends_trusted": "Chaînes de Confiance",
    "automarket_selected_videos": "Vidéos retenues (Log)",
    "automarket_moderated_responses": "Réponses en modération",
})


class AutomarketPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.youtube_api = YoutubeAPI(self.plugin_manager.config)
        self.response_db_widget = ResponseDBDisplayWidget("response_db_display", "automarket", plugin_manager)
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'campaign_responses' not in st.session_state:
            st.session_state.campaign_responses = []
        if 'selected_responses' not in st.session_state:
            st.session_state.selected_responses = {}
        if 'rejected_videos' not in st.session_state:
            st.session_state.rejected_videos = []
        if 'expand_all' not in st.session_state:
            st.session_state.expand_all = True
        if 'excluded_comments' not in st.session_state:
            st.session_state.excluded_comments = []
        if 'campaign_target_videos' not in st.session_state:
            st.session_state.campaign_target_videos = []
        if 'campaign_current_comments' not in st.session_state:
            st.session_state.campaign_current_comments = []
        if 'campaign_responses' not in st.session_state:
            st.session_state.campaign_responses = []
        if 'campaign_excluded_comments' not in st.session_state:
            st.session_state.campaign_excluded_comments = []
        # Add this line to initialize target_videos
        if 'target_videos' not in st.session_state:
            st.session_state.target_videos = []

    def get_config_fields(self):
        return {
            "comments_per_video": {
                "type": "number",
                "label": "Default Comments per Video",
                "default": 2
            },
            "min_subscribers": {
                "type": "number",
                "label": "Minimum Subscribers",
                "default": 10000
            },
            "max_videos_per_keyword": {
                "type": "number",
                "label": "Max Videos per Keyword",
                "default": 10
            },
            "expiry_days": {
                "type": "number",
                "label": "Expiry Days",
                "default": 7
            },
            "view_threshold": {
                "type": "number",
                "label": "View Threshold",
                "default": 1000
            },
            "trusted_channel_videos": {
                "type": "number",
                "label": "Videos from Trusted Channels",
                "default": 3
            },
            "response_prompt": {
                "type": "textarea",
                "label": "LLM Prompt for Responses",
                "default": """Suggest a concise response (<500 chars) to this comment, promoting the video at {url} (mention it). Use a direct tone, as if you're the commenter, inspired by this transcript: {transcript}"""
            },
            "exclusion_keyword": {
                "type": "text",
                "label": "Keyword to Exclude Comments",
                "default": "Stop"
            },
        }

    def get_tabs(self):
        return [
            {"name": "Lancer une campagne", "plugin": "automarket"},
            {"name": "Réponses existantes", "plugin": "automarket"},
            {"name": "Surveiller les tendances", "plugin": "automarket"}
        ]

    def fetch_videos_for_keyword(self, keyword: str, max_videos: int, min_subscribers: int, expiry_days: int, view_threshold: int, combine_keywords: bool = False) -> List[Dict[str, Any]]:
        """Récupère les vidéos pour un mot-clé avec filtres et complète avec get_video_infos."""
        videos = []
        for order in ["relevance", "date"]:
            search_results = self.youtube_api.search_videos(
                keyword, max_videos * 2, order=order, language=st.session_state.lang, combine_keywords=combine_keywords)
            st.info(
                f"Nombre de vidéos trouvées par l'API pour '{keyword}' (ordre: {order}) : {len(search_results)}")
            rejected = 0

            for video in search_results:
                # Compléter les informations de la vidéo avec get_video_infos
                normalized_video = self.youtube_api.get_video_infos(video)

                if normalized_video['language'] != st.session_state.lang:
                    st.session_state.rejected_videos.append({
                        'title': normalized_video['title'],
                        'url': normalized_video['url'],
                        'channel_title': normalized_video['channel_title'],
                        'channel_url': f"https://www.youtube.com/channel/{normalized_video['channel_id']}",
                        'reason': f"Language ({normalized_video['language']} != {st.session_state.lang})",
                        'stats': {'language': normalized_video['language']},
                        'keyword': keyword,
                        'criterion': order
                    })
                    continue

                published_at = datetime.strptime(
                    normalized_video['published_at'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
                days_old = (datetime.now(pytz.UTC) - published_at).days
                last_comment = self.youtube_api.get_comments(
                    normalized_video['video_id'], max_results=1, order="time")
                last_comment_days = (datetime.now(pytz.UTC) - datetime.strptime(
                    last_comment[0]['published_at'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)).days if last_comment else expiry_days + 1

                reject_reason = []
                if normalized_video['subscriber_count'] < min_subscribers:
                    reject_reason.append(
                        f"Subscribers ({normalized_video['subscriber_count']} < {min_subscribers})")
                if last_comment_days > expiry_days:
                    reject_reason.append(
                        f"Last comment ({last_comment_days} days > {expiry_days})")
                if days_old > 1 and normalized_video['view_count'] < view_threshold:
                    reject_reason.append(
                        f"Views ({normalized_video['view_count']} < {view_threshold})")

                if not reject_reason:
                    videos.append(normalized_video)
                else:
                    rejected += 1
                    st.session_state.rejected_videos.append({
                        'title': normalized_video['title'],
                        'url': normalized_video['url'],
                        'channel_title': normalized_video['channel_title'],
                        'channel_url': f"https://www.youtube.com/channel/{normalized_video['channel_id']}",
                        'reason': ", ".join(reject_reason),
                        'stats': {
                            'subscribers': normalized_video['subscriber_count'],
                            'views': normalized_video['view_count'],
                            'days_old': days_old,
                            'last_comment_days': last_comment_days
                        },
                        'keyword': keyword,
                        'criterion': order
                    })

                if len(videos) >= max_videos:
                    break
            if len(videos) >= max_videos:
                break
            st.info(f"Number of rejected videos : {rejected}")
        return videos[:max_videos]

    def fetch_videos_from_trusted_channels(self, keyword: str, max_videos: int, min_subscribers: int, expiry_days: int, view_threshold: int) -> List[Dict[str, Any]]:
        """Récupère les vidéos récentes des chaînes de confiance pour un mot-clé avec détection de langue."""
        trusted_channels = [
            ch for ch in get_target_channels() if keyword in ch['keywords']]
        videos = []
        for channel in trusted_channels:
            channel_videos = self.youtube_api.get_channel_recent_videos(
                channel['channel_id'], max_results=max_videos * 2)
            for video in channel_videos:
                # Détection de la langue
                normalized_video = self.youtube_api.get_video_infos(video)
                video_language = detect(
                    normalized_video['title'] + " " + normalized_video.get('description', 'No description'))
                if video_language != st.session_state.lang:
                    st.session_state.rejected_videos.append({
                        'title': normalized_video['title'],
                        'url': normalized_video['url'],
                        'channel_title': normalized_video['channel_title'],
                        'channel_url': f"https://www.youtube.com/channel/{normalized_video['channel_id']}",
                        'reason': f"Language ({video_language} != {st.session_state.lang})",
                        'stats': {'language': video_language},
                        'keyword': keyword,
                        'criterion': 'trust'
                    })
                    continue

                published_at = datetime.strptime(
                    normalized_video['published_at'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
                days_old = (datetime.now(pytz.UTC) - published_at).days
                last_comment = self.youtube_api.get_comments(
                    normalized_video['video_id'], max_results=1, order="time")
                last_comment_days = (datetime.now(pytz.UTC) - datetime.strptime(
                    last_comment[0]['published_at'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)).days if last_comment else expiry_days + 1

                reject_reason = []
                if normalized_video['subscriber_count'] < min_subscribers:
                    reject_reason.append(
                        f"Subscribers ({normalized_video['subscriber_count']} < {min_subscribers})")
                if last_comment_days > expiry_days:
                    reject_reason.append(
                        f"Last comment ({last_comment_days} days > {expiry_days})")
                if days_old > 1 and normalized_video['view_count'] < view_threshold:
                    reject_reason.append(
                        f"Views ({normalized_video['view_count']} < {view_threshold})")

                if not reject_reason:
                    videos.append(normalized_video)
                else:
                    st.session_state.rejected_videos.append({
                        'title': normalized_video['title'],
                        'url': normalized_video['url'],
                        'channel_title': normalized_video['channel_title'],
                        'channel_url': f"https://www.youtube.com/channel/{normalized_video['channel_id']}",
                        'reason': ", ".join(reject_reason),
                        'stats': {
                            'subscribers': normalized_video['subscriber_count'],
                            'views': normalized_video['view_count'],
                            'days_old': days_old,
                            'last_comment_days': last_comment_days
                        },
                        'keyword': keyword,
                        'criterion': 'trust'
                    })

                if len(videos) >= max_videos:
                    break
            if len(videos) >= max_videos:
                break
        return videos[:max_videos]

    def generate_responses(self, config, campaign_video: Dict[str, Any], comments: List[Dict[str, Any]], max_comments_debug: int = None):
        """Génère les réponses tout en excluant celles marquées par le mot-clé d'exclusion."""
        responses = []
        total_comments = min(
            len(comments), max_comments_debug) if max_comments_debug else len(comments)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        exclusion_keyword = config['automarket']['exclusion_keyword'].strip(
        )

        prompt = config['automarket']['response_prompt'].format(
            url=campaign_video['url'],
            transcript=campaign_video.get('transcript', '')
        )

        for idx, comment in enumerate(comments[:total_comments]):
            progress = (idx + 1) / total_comments
            progress_bar.progress(progress)
            progress_text.text(
                t("automarket_progress").format(idx + 1, total_comments))

            comment_context = f"Comment by {comment['author']} on {comment['video_title']} from {comment['channel_title']}:\n{comment['text']}"
            try:
                llm_response = self.process_with_llm(
                    prompt,
                    config.get('llm', {}).get('llm_sys_prompt', ''),
                    comment_context
                )
                clean_response = llm_response.strip().strip(".")
                if clean_response.startswith('"') and clean_response.endswith('"'):
                    clean_response = clean_response[1:-1]

                # Vérification du mot-clé d'exclusion
                if clean_response == exclusion_keyword:
                    st.session_state.excluded_comments.append({
                        'text': comment['text'],
                        'author': comment['author'],
                        'video_title': comment['video_title'],
                        'video_id': comment['video_id'],
                        'published_at': comment['published_at']
                    })
                    continue

                responses.append({
                    'comment_id': comment['id'],
                    'response': clean_response,
                    'target_video_id': comment['video_id'],
                    'comment_text': comment['text'],
                    'channel_id': comment['channel_id'],
                    'channel_title': comment['channel_title'],
                    'video_title': comment['video_title'],
                    'view_count': comment.get('view_count', 0),
                    'like_count': comment.get('like_count', 0),
                    'comment_count': comment.get('comment_count', 0),
                    'days_old': comment.get('days_old', 0),
                    'subscriber_count': comment.get('subscriber_count', 0),
                    'keyword': comment.get('keyword', 'unknown'),
                    'criterion': comment.get('criterion', 'unknown')
                })
            except Exception as e:
                responses.append({
                    'comment_id': comment['id'],
                    'response': f"Error: {str(e)}",
                    'target_video_id': comment['video_id'],
                    'comment_text': comment['text'],
                    'channel_id': comment['channel_id'],
                    'channel_title': comment['channel_title'],
                    'video_title': comment['video_title'],
                    'view_count': comment.get('view_count', 0),
                    'like_count': comment.get('like_count', 0),
                    'comment_count': comment.get('comment_count', 0),
                    'days_old': comment.get('days_old', 0),
                    'subscriber_count': comment.get('subscriber_count', 0),
                    'keyword': comment.get('keyword', 'unknown'),
                    'criterion': comment.get('criterion', 'unknown')
                })

        progress_bar.empty()
        progress_text.empty()
        return responses

    def post_responses(self, config, selected_responses, campaign_timestamp: str):
        """Poste les réponses et les sauvegarde dans la base."""
        moderated_count = 0
        for response in selected_responses:
            video_id = response['target_video_id']
            comment_id = response['comment_id']
            if not check_existing_response(video_id, comment_id):
                try:
                    api_response = self.youtube_api.post_comment_reply(
                        comment_id, response['response'])
                    if api_response:
                        response_id = api_response.get('id')
                        moderation_status = api_response.get(
                            'moderation_status', 'unknown')
                        if moderation_status != 'published':  # Si différent de published, on compte comme modéré
                            moderated_count += 1
                        save_response(
                            campaign_timestamp=campaign_timestamp,
                            video_id=video_id,
                            comment_id=comment_id,
                            response_id=response_id,
                            channel_id=response['channel_id'],
                            keyword=response['keyword'],
                            response_text=response['response'],
                            moderation_status=moderation_status
                        )
                except Exception as e:
                    print(f"Error posting response to {comment_id}: {str(e)}")

        # Calcul des statistiques
        total_videos = len(set(r['target_video_id']
                           for r in selected_responses)) + len(st.session_state.rejected_videos)
        excluded_videos = len(st.session_state.rejected_videos)
        total_comments = len(
            st.session_state.current_comments) if 'current_comments' in st.session_state else len(selected_responses)
        stop_comments = len([c for c in st.session_state.excluded_comments if c['text'].strip(
        ).lower() == config['automarket']['exclusion_keyword'].strip().lower()])
        excluded_comments = len(st.session_state.excluded_comments)
        # Note : semble être une erreur dans ton code original, devrait être len(responses) - len(selected_responses)
        refused_responses = len(selected_responses) - len(selected_responses)
        posted_responses = len(selected_responses)

        save_campaign_stats(campaign_timestamp, {
            'total_videos': total_videos,
            'excluded_videos': excluded_videos,
            'total_comments': total_comments,
            'stop_comments': stop_comments,
            'excluded_comments': excluded_comments,
            'refused_responses': refused_responses,
            'posted_responses': posted_responses,
            'moderated_responses': moderated_count
        })

    def display_responses(self, prefix: str, config: dict, campaign_video: Dict[str, Any], responses: List[Dict[str, Any]], campaign_timestamp: str):
        """Affiche et gère les réponses générées avec préfixe pour les éléments UI"""
        st.subheader(t("automarket_responses"))
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("automarket_expand_all"), key=f"{prefix}_expand_all"):
                st.session_state.expand_all = True
        with col2:
            if st.button(t("automarket_collapse_all"), key=f"{prefix}_collapse_all"):
                st.session_state.expand_all = False

        default_prompt = config['automarket']['response_prompt']
        new_prompt = st.text_area(
            "Nouveau prompt pour regénérer les réponses",
            value=default_prompt,
            height=150,
            key=f"{prefix}_regen_prompt"
        )

        if st.button("Regénérer les réponses", key=f"{prefix}_regen_button"):
            if 'current_comments' in st.session_state:
                with st.spinner("Regénération des réponses..."):
                    original_prompt = config['automarket']['response_prompt']
                    config['automarket']['response_prompt'] = new_prompt
                    st.session_state.campaign_responses = self.generate_responses(
                        config, campaign_video, st.session_state.current_comments)
                    st.session_state.selected_responses = {
                        i: True for i in range(len(st.session_state.campaign_responses))}
                    config['automarket']['response_prompt'] = original_prompt
                    st.success("Réponses regénérées avec succès !")
            else:
                st.warning(
                    "Aucune campagne précédente trouvée pour regénération.")

        col3, col4 = st.columns(2)
        with col3:
            if st.button(t("automarket_select_all"), key=f"{prefix}_select_all"):
                for i in range(len(st.session_state.campaign_responses)):
                    st.session_state.selected_responses[i] = False
                st.rerun()
        with col4:
            if st.button(t("automarket_deselect_all"), key=f"{prefix}_deselect_all"):
                for i in range(len(st.session_state.campaign_responses)):
                    st.session_state.selected_responses[i] = True
                st.rerun()

        for i, response in enumerate(responses):
            title = f"{t('automarket_reponse_to_comment').format(i+1)} : {response['video_title']}"
            with st.expander(title, expanded=st.session_state.expand_all):
                st.write(f"{t('automarket_channel')}: {response['channel_title']} "
                         f"({self.youtube_api.format_count(response['subscriber_count'])} subscribers)")
                if st.button(t("automarket_add_to_trusted"), key=f"{prefix}_add_trusted_{i}"):
                    channel_url = f"https://www.youtube.com/channel/{response['channel_id']}"
                    add_target_channel(
                        channel_id=response['channel_id'],
                        channel_title=response['channel_title'],
                        channel_url=channel_url,
                        keywords=[response['keyword']],
                        subscriber_count=response['subscriber_count']
                    )
                    st.success(
                        f"Added {response['channel_title']} to trusted channels!")

                st.write(
                    f"{t('automarket_video')}: [{response['video_title']}](https://www.youtube.com/watch?v={response['target_video_id']})")
                st.write(f"Views: {self.youtube_api.format_count(response['view_count'])}, "
                         f"Likes: {self.youtube_api.format_count(response['like_count'])}, "
                         f"Comments: {response['comment_count']}, "
                         f"Age: {response['days_old']} days")
                st.write(f"{t('automarket_keyword')}: {response['keyword']}")
                st.write(
                    f"{t('automarket_criterion')}: {response['criterion']}")
                st.write(
                    f"{t('automarket_comment')}: {response['comment_text']}")
                st.write(f"{t('automarket_response')}: {response['response']}")
                moderation_status = get_response_moderation_status(
                    response['target_video_id'], response['comment_id'])
                st.write(f"Moderation Status: {moderation_status}")
                st.write(f"Promoted Video: {campaign_video['title']}")

                current_value = st.session_state.selected_responses.get(
                    i, False)
                new_value = st.checkbox(
                    t("automarket_exclude_response"),
                    value=current_value,
                    key=f"{prefix}_exclude_{i}"
                )
                st.session_state.selected_responses[i] = new_value

        if st.button(t("automarket_post_responses"), key=f"{prefix}_post_responses"):
            with st.spinner(t("automarket_posting")):
                selected_responses = [r for i, r in enumerate(responses)
                                      if not st.session_state.selected_responses.get(i, False)]
                self.post_responses(
                    config, selected_responses, campaign_timestamp)

                # Calcul des statistiques
                total_videos = len(set(
                    r['target_video_id'] for r in responses)) + len(st.session_state.rejected_videos)
                excluded_videos = len(st.session_state.rejected_videos)
                total_comments = len(
                    st.session_state.current_comments) if 'current_comments' in st.session_state else len(responses)
                stop_comments = len([c for c in st.session_state.excluded_comments if c['text'].strip(
                ).lower() == config['automarket']['exclusion_keyword'].strip().lower()])
                excluded_comments = len(st.session_state.excluded_comments)
                refused_responses = len(responses) - len(selected_responses)
                posted_responses = len(selected_responses)
                moderated_responses = sum(1 for r in selected_responses if self.youtube_api.post_comment_reply(
                    r['comment_id'], r['response']) and self.youtube_api.post_comment_reply(r['comment_id'], r['response']).get('moderation_status', 'unknown') != 'published')
                published_responses = posted_responses - \
                    moderated_responses  # Calcul des réponses publiées

                save_campaign_stats(campaign_timestamp, {
                    'total_videos': total_videos,
                    'excluded_videos': excluded_videos,
                    'total_comments': total_comments,
                    'stop_comments': stop_comments,
                    'excluded_comments': excluded_comments,
                    'refused_responses': refused_responses,
                    'posted_responses': posted_responses,
                    'moderated_responses': moderated_responses
                })

                st.success(t("automarket_campaign_complete"))

                with st.expander(f"Campaign Statistics ({posted_responses + moderated_responses + refused_responses} actions)"):
                    st.write(f"Total videos found: {total_videos}")
                    st.write(f"Excluded videos: {excluded_videos}")
                    st.write(f"Total comments analyzed: {total_comments}")
                    st.write(f"Comments marked as 'stop': {stop_comments}")
                    st.write(f"Total excluded comments: {excluded_comments}")
                    st.write(f"Refused responses: {refused_responses}")
                    st.write(f"Responses posted: {posted_responses}")
                    st.write(
                        f"{t('automarket_moderated_responses')}: {moderated_responses}")
                    # Ajout en gras
                    st.write(f"**Responses published: {published_responses}**")

    def log_selected_videos(self, prefix: str, videos: List[Dict[str, Any]], keyword: str = "N/A"):
        """Affiche un log des vidéos retenues avec leurs statistiques."""
        with st.expander(f"{t('automarket_selected_videos')} ({len(videos)})"):
            for video in videos:
                st.markdown(f"Video: [**{video['title']}**]({video['url']})")
                st.markdown(
                    f"Channel: [**{video['channel_title']}**](https://www.youtube.com/channel/{video['channel_id']})")
                st.write(
                    f"Keyword: {keyword if keyword != 'N/A' else video.get('keyword', 'N/A')}")
                st.write(f"Stats: Subscribers: {self.youtube_api.format_count(video['subscriber_count'])}, "
                         f"Views: {self.youtube_api.format_count(video['view_count'])}, "
                         f"Likes: {self.youtube_api.format_count(video['like_count'])}, "
                         f"Comments: {video['comment_count']}, "
                         f"Age: {video['days_old']} days")
                st.write("---")

    def log_rejected_videos(self, prefix: str, videos: List[Dict[str, Any]]):
        """Affiche un log des vidéos rejetées avec leurs statistiques."""
        with st.expander(f"{t('automarket_rejected_videos')} ({len(st.session_state.rejected_videos)})"):
            for rejected in videos:
                st.markdown(
                    f"Video: [**{rejected['title']}**]({rejected['url']})")
                channel_url = rejected.get(
                    'channel_url', f"https://www.youtube.com/channel/{rejected.get('channel_id', '')}")
                st.markdown(
                    f"Channel: [**{rejected['channel_title']}**]({channel_url})")
                st.write(f"{t('automarket_keyword')}: {rejected['keyword']}")
                st.write(
                    f"{t('automarket_criterion')}: {rejected['criterion']}")
                st.write(t("automarket_rejected_reason").format(
                    rejected['reason']))
                st.write(f"Stats: {rejected['stats']}")
                st.write("---")

    def log_accepted_videos(self, prefix: str, accepted_videos: List[Dict[str, Any]]):
        """Affiche un log des vidéos conservées avec leurs statistiques."""
        with st.expander(f"Accepted Videos ({len(accepted_videos)})"):
            for video in accepted_videos:
                st.markdown(f"Video: [**{video['title']}**]({video['url']})")
                st.markdown(
                    f"Channel: [**{video['channel_title']}**](https://www.youtube.com/channel/{video['channel_id']})")
                st.write(f"Keyword: {video.get('keyword', 'N/A')}")
                st.write(f"Criterion: {video.get('criterion', 'N/A')}")
                st.write(f"Stats: Subscribers: {self.youtube_api.format_count(video['subscriber_count'])}, "
                         f"Views: {self.youtube_api.format_count(video['view_count'])}, "
                         f"Likes: {self.youtube_api.format_count(video['like_count'])}, "
                         f"Comments: {video['comment_count']}, Age: {video['days_old']} days")
                st.write("---")

    def log_rejected_comments(self, prefix: str):
        if st.session_state.excluded_comments:
            with st.expander(f"{t('automarket_excluded_comments')} ({len(st.session_state.excluded_comments)})"):
                for excluded in st.session_state.excluded_comments:
                    st.write(f"Comment: {excluded['text']}")
                    st.write(f"Author: {excluded['author']}")
                    st.write(
                        f"Video: [{excluded['video_title']}](https://www.youtube.com/watch?v={excluded['video_id']})")
                    st.write(f"Published: {excluded['published_at']}")
                    st.write("---")

    def display_quota(self):
        quota_info = self.youtube_api.get_quota_usage()
        st.markdown(
            f"Quota restant : {quota_info['remaining_percentage']:.2f}% restant ({quota_info['quota_usage']} unités sur {quota_info['quota_limit']}) [Check](https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas?hl=fr&inv=1&invt=AbrUGw&pageState=(%22allQuotasTable%22%253A(%22c%22%253A%5B%22displayDimensions%22%5D)))")

    def fetch_monitor_videos(self, selected_keyword: str, max_videos_per_keyword: int, expiry_days: int, min_subscribers: int, view_threshold: int, search_time: bool, search_relevant: bool, search_trusted: bool) -> List[Dict[str, Any]]:
        """Récupère les vidéos pour l'onglet Surveiller les tendances avec filtrage de langue."""
        target_videos = []
        if search_time:
            videos = self.youtube_api.search_videos(
                selected_keyword, max_videos_per_keyword, order="date",
                language=st.session_state.lang)
            target_videos.extend([v for v in videos if v['days_old'] <=
                                 expiry_days and v['language'] == st.session_state.lang])

        if search_relevant:
            videos = self.youtube_api.search_videos(
                selected_keyword, max_videos_per_keyword, order="relevance",
                language=st.session_state.lang)
            target_videos.extend([v for v in videos if v['days_old'] <=
                                 expiry_days and v['language'] == st.session_state.lang])

        if search_trusted:
            trusted_channels = [ch for ch in get_target_channels(
            ) if selected_keyword in ch['keywords']]
            for channel in trusted_channels:
                videos = self.youtube_api.get_channel_recent_videos(
                    channel['channel_id'], max_results=max_videos_per_keyword)
                target_videos.extend([v for v in videos if v['days_old'] <=
                                     expiry_days and v['language'] == st.session_state.lang])

        # Filtrer selon les critères
        filtered_videos = []
        for video in target_videos:
            last_comment = self.youtube_api.get_comments(
                video['video_id'], max_results=1, order="time")
            last_comment_days = (datetime.now(pytz.UTC) - datetime.strptime(
                last_comment[0]['published_at'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)).days if last_comment else expiry_days + 1

            reject_reason = []
            if video['subscriber_count'] < min_subscribers:
                reject_reason.append(
                    f"Subscribers ({video['subscriber_count']} < {min_subscribers})")
            if last_comment_days > expiry_days:
                reject_reason.append(
                    f"Last comment ({last_comment_days} days > {expiry_days})")
            if video['view_count'] < view_threshold:
                reject_reason.append(
                    f"Views ({video['view_count']} < {view_threshold})")

            if not reject_reason:
                filtered_videos.append(video)
            else:
                st.session_state.rejected_videos.append({
                    'title': video['title'],
                    'url': video['url'],
                    'channel_title': video['channel_title'],
                    'reason': ", ".join(reject_reason),
                    'stats': {
                        'subscribers': video['subscriber_count'],
                        'views': video['view_count'],
                        'days_old': video['days_old'],
                        'last_comment_days': last_comment_days
                    },
                    'keyword': selected_keyword,
                    'criterion': 'monitor'
                })
        return filtered_videos

    def fetch_monitor_comments(self, videos: List[Dict[str, Any]], comments_per_video: int, selected_keyword: str) -> List[Dict[str, Any]]:
        """Récupère les commentaires pour les vidéos sélectionnées."""
        comments = []
        for video in videos:
            video_comments = self.youtube_api.get_comments(
                video['video_id'], max_results=comments_per_video, order="relevance")
            for comment in video_comments:
                comment['video_title'] = video['title']
                comment['channel_title'] = video['channel_title']
                comment['channel_id'] = video['channel_id']
                comment['view_count'] = video['view_count']
                comment['like_count'] = video['like_count']
                comment['comment_count'] = video['comment_count']
                comment['days_old'] = video['days_old']
                comment['subscriber_count'] = video['subscriber_count']
                comment['keyword'] = selected_keyword
                comment['criterion'] = 'monitor'
            comments.extend(video_comments)
        return comments

    def fetch_campaign_videos(self, campaign_video: Dict[str, Any], max_videos_per_keyword: int, min_subscribers: int, expiry_days: int, view_threshold: int, combine_keywords: bool, search_keywords: bool, search_trusted: bool, trusted_channel_videos: int) -> List[Dict[str, Any]]:
        """Récupère les vidéos pour une campagne dans tab1 avec filtrage de langue."""
        target_videos = []
        if search_keywords:
            if combine_keywords:
                combined_query = " ".join(campaign_video['keywords'])
                videos = self.fetch_videos_for_keyword(
                    combined_query, max_videos_per_keyword, min_subscribers, expiry_days, view_threshold, combine_keywords=True)
                target_videos.extend(videos)
            else:
                for keyword in campaign_video['keywords']:
                    videos = self.fetch_videos_for_keyword(
                        keyword, max_videos_per_keyword, min_subscribers, expiry_days, view_threshold, combine_keywords=False)
                    target_videos.extend(videos)

        if search_trusted:
            for keyword in campaign_video['keywords']:
                trusted_videos = self.fetch_videos_from_trusted_channels(
                    keyword, trusted_channel_videos, min_subscribers, expiry_days, view_threshold)
                target_videos.extend(trusted_videos)

        return target_videos

    def fetch_campaign_comments(self, videos: List[Dict[str, Any]], comments_per_video: int, campaign_video: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Récupère les commentaires pour les vidéos d'une campagne dans tab1."""
        comments = []
        for video in videos:
            video_comments = self.youtube_api.get_comments(
                video['video_id'], max_results=comments_per_video, order="relevance")
            for comment in video_comments:
                comment['video_title'] = video['title']
                comment['channel_title'] = video['channel_title']
                comment['channel_id'] = video['channel_id']
                comment['view_count'] = video['view_count']
                comment['like_count'] = video['like_count']
                comment['comment_count'] = video['comment_count']
                comment['days_old'] = video['days_old']
                comment['subscriber_count'] = video['subscriber_count']
                if 'description' not in video:
                    st.warning(
                        f"Debug: Video {video['title']} (ID: {video['video_id']}) lacks 'description'. Keys available: {list(video.keys())}")
                    video['description'] = ''
                try:
                    comment['keyword'] = next(
                        (kw for kw in campaign_video['keywords'] if kw in video['title'].lower(
                        ) or kw in video.get('description', '').lower()),
                        'unknown'
                    )
                except Exception as e:
                    st.error(
                        f"Debug: Error assigning keyword for video {video['title']} (ID: {video['video_id']}): {str(e)}")
                    comment['keyword'] = 'unknown'
                comment['criterion'] = 'trust' if 'trust' in video.get('criterion', '') else (
                    'relevance' if video in videos else 'date')
            comments.extend(video_comments)
        return comments

    def run(self, config):
        tab1, tab2, tab3 = st.tabs(
            ["Lancer une campagne", "Réponses existantes", t("monitor_trends_tab")])
        self._initialize_session_state()
        if 'campaign_timestamp' not in st.session_state:
            st.session_state.campaign_timestamp = datetime.now(
                pytz.UTC).isoformat()
        with tab1:
            st.header(t("automarket_header"))

            # 1. Sélection de la vidéo à promouvoir
            videos = [v for v in get_videos() if v['keywords']]
            if not videos:
                st.warning(t("automarket_no_videos_with_keywords"))
                return

            video_options = {
                f"{v['title']} ({', '.join(v['keywords'])})": v for v in videos}
            selected_video_title = st.selectbox(
                t("automarket_select_video"),
                options=list(video_options.keys()),
                key="automarket_select_video"
            )
            campaign_video = video_options[selected_video_title]

            # 2. Configuration de la campagne
            comments_per_video = st.number_input(
                t("automarket_comments_per_video"),
                min_value=1,
                value=int(config['automarket']['comments_per_video']),
                key="automarket_comments_per_video"
            )
            min_subscribers = st.number_input(
                t("automarket_min_subscribers"),
                min_value=0,
                value=int(config['automarket']['min_subscribers']),
                key="automarket_min_subscribers"
            )
            max_videos_per_keyword = st.number_input(
                t("automarket_max_videos_per_keyword"),
                min_value=1,
                value=int(config['automarket']['max_videos_per_keyword']),
                key="automarket_max_videos_per_keyword"
            )
            expiry_days = st.number_input(
                t("automarket_expiry_days"),
                min_value=1,
                value=int(config['automarket']['expiry_days']),
                key="automarket_expiry_days"
            )
            view_threshold = st.number_input(
                t("automarket_view_threshold"),
                min_value=0,
                value=int(config['automarket']['view_threshold']),
                key="automarket_view_threshold"
            )
            trusted_channel_videos = st.number_input(
                t("automarket_trusted_channel_videos"),
                min_value=1,
                value=int(config['automarket']['trusted_channel_videos']),
                key="automarket_trusted_channel_videos"
            )

            debug_mode = st.checkbox(
                t("automarket_debug_mode"), value=False, key="automarket_debug_mode")
            max_comments_debug = None
            if debug_mode:
                max_comments_debug = st.number_input(
                    t("automarket_max_comments_debug"),
                    min_value=1,
                    value=5,
                    key="automarket_max_comments_debug"
                )

            combine_keywords = st.checkbox(
                "Combine Keywords", value=False, key="automarket_combine_keywords")
            search_keywords = st.checkbox(
                t("automarket_search_keywords"), value=True, key="automarket_search_keywords")
            search_trusted = st.checkbox(
                t("automarket_search_trusted"), value=True, key="automarket_search_trusted")

            self.display_quota()

            # Ajout des trois boutons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                fetch_videos_btn = st.button(
                    "Récupérer les vidéos", key="campaign_fetch_videos")
            with col2:
                fetch_comments_btn = st.button(
                    "Récupérer les commentaires", key="campaign_fetch_comments")
            with col3:
                generate_responses_btn = st.button(
                    "Générer les réponses", key="campaign_generate_responses")
            with col4:
                start_campaign_btn = st.button(
                    t("automarket_start_campaign"), key="campaign_start_campaign")

            if fetch_videos_btn or start_campaign_btn:
                with st.spinner(t("automarket_processing")):
                    st.session_state.campaign_timestamp = datetime.now(
                        pytz.UTC).isoformat()
                    initial_quota = self.youtube_api.quota_usage
                    st.session_state.rejected_videos = []  # Réinitialiser uniquement ici
                    st.session_state.campaign_target_videos = self.fetch_campaign_videos(
                        campaign_video, max_videos_per_keyword, min_subscribers, expiry_days,
                        view_threshold, combine_keywords, search_keywords, search_trusted,
                        trusted_channel_videos)
                    quota_used = self.youtube_api.quota_usage - initial_quota
                    st.info(t("automarket_quota_consumed").format(
                        units=quota_used))

            # Afficher les vidéos stockées même si on ne vient pas de les récupérer
            if st.session_state.campaign_target_videos:
                self.log_selected_videos(
                    "campaign", st.session_state.campaign_target_videos)

            if (fetch_comments_btn or start_campaign_btn) and st.session_state.campaign_target_videos:
                with st.spinner(t("automarket_processing")):
                    initial_quota = self.youtube_api.quota_usage
                    st.session_state.campaign_current_comments = self.fetch_campaign_comments(
                        st.session_state.campaign_target_videos, comments_per_video, campaign_video)
                    quota_used = self.youtube_api.quota_usage - initial_quota
                    st.info(t("automarket_quota_consumed").format(
                        units=quota_used))

            if (generate_responses_btn or start_campaign_btn) and st.session_state.campaign_current_comments:
                with st.spinner(t("automarket_generating_responses")):
                    initial_quota = self.youtube_api.quota_usage
                    st.session_state.campaign_responses = self.generate_responses(
                        config, campaign_video, st.session_state.campaign_current_comments,
                        max_comments_debug if debug_mode else None)
                    st.session_state.selected_responses = {
                        i: not debug_mode for i in range(len(st.session_state.campaign_responses))}
                    quota_used = self.youtube_api.quota_usage - initial_quota
                    st.info(t("automarket_quota_consumed").format(
                        units=quota_used))

            self.log_rejected_videos(
                "campaign", st.session_state.rejected_videos)
            self.log_rejected_comments("campaign")

            if st.session_state.campaign_responses:
                self.display_responses("campaign", config, campaign_video,
                                       st.session_state.campaign_responses, st.session_state.campaign_timestamp)

        with tab2:
            st.write("tab2")
            self.response_db_widget.display()

        with tab3:
            st.header(t("monitor_trends_header"))

            all_keywords = set()
            for channel in get_target_channels():
                all_keywords.update(channel['keywords'])
            for video in get_videos():
                all_keywords.update(video['keywords'])

            if not all_keywords:
                st.warning("Aucun mot-clé trouvé dans la base de données.")
                return

            selected_keyword = st.selectbox(
                t("monitor_trends_select_keyword"),
                options=sorted(list(all_keywords)),
                key="monitor_keyword"
            )

            st.subheader(t("monitor_trends_search_criteria"))
            search_time = st.checkbox(
                t("monitor_trends_time"), value=True, key="monitor_time")
            search_relevant = st.checkbox(
                t("monitor_trends_relevant"), value=True, key="monitor_relevant")
            search_trusted = st.checkbox(
                t("monitor_trends_trusted"), value=True, key="monitor_trusted")

            comments_per_video = st.number_input(
                t("automarket_comments_per_video"),
                min_value=1,
                value=int(config['automarket']['comments_per_video']),
                key="monitor_comments_per_video"
            )
            min_subscribers = st.number_input(
                t("automarket_min_subscribers"),
                min_value=0,
                value=int(config['automarket']['min_subscribers']),
                key="monitor_min_subscribers"
            )
            max_videos_per_keyword = st.number_input(
                t("automarket_max_videos_per_keyword"),
                min_value=1,
                value=int(config['automarket']['max_videos_per_keyword']),
                key="monitor_max_videos_per_keyword"
            )
            expiry_days = st.number_input(
                t("automarket_expiry_days"),
                min_value=1,
                value=int(config['automarket']['expiry_days']),
                key="monitor_expiry_days"
            )
            view_threshold = st.number_input(
                t("automarket_view_threshold"),
                min_value=0,
                value=int(config['automarket']['view_threshold']),
                key="monitor_view_threshold"
            )

            self.display_quota()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                fetch_videos_btn = st.button(
                    "Récupérer les vidéos", key="monitor_fetch_videos")
            with col2:
                fetch_comments_btn = st.button(
                    "Récupérer les commentaires", key="monitor_fetch_comments")
            with col3:
                generate_responses_btn = st.button(
                    "Générer les réponses", key="monitor_generate_responses")
            with col4:
                start_campaign_btn = st.button(
                    t("automarket_start_campaign"), key="monitor_start_campaign")

            campaign_video = next(
                (v for v in get_videos() if selected_keyword in v['keywords']), None)
            if not campaign_video:
                st.error(
                    f"Aucune vidéo personnelle trouvée pour le mot-clé {selected_keyword}")
                return

            if fetch_videos_btn or start_campaign_btn:
                with st.spinner(t("automarket_processing")):
                    st.session_state.campaign_timestamp = datetime.now(
                        pytz.UTC).isoformat()
                    initial_quota = self.youtube_api.quota_usage
                    st.session_state.rejected_videos = []  # Réinitialiser uniquement ici
                    st.session_state.target_videos = self.fetch_monitor_videos(
                        selected_keyword, max_videos_per_keyword, expiry_days, min_subscribers,
                        view_threshold, search_time, search_relevant, search_trusted)
                    self.log_selected_videos(
                        "monitor", st.session_state.target_videos, selected_keyword)
                    quota_used = self.youtube_api.quota_usage - initial_quota
                    st.info(t("automarket_quota_consumed").format(
                        units=quota_used))

            # Afficher les vidéos stockées même si on ne vient pas de les récupérer
            if st.session_state.target_videos:
                self.log_selected_videos(
                    "monitor", st.session_state.target_videos, selected_keyword)

            if (fetch_comments_btn or start_campaign_btn) and st.session_state.target_videos:
                with st.spinner(t("automarket_processing")):
                    initial_quota = self.youtube_api.quota_usage
                    st.session_state.current_comments = self.fetch_monitor_comments(
                        st.session_state.target_videos, comments_per_video, selected_keyword)
                    quota_used = self.youtube_api.quota_usage - initial_quota
                    st.info(t("automarket_quota_consumed").format(
                        units=quota_used))

            if (generate_responses_btn or start_campaign_btn) and st.session_state.current_comments:
                with st.spinner(t("automarket_generating_responses")):
                    initial_quota = self.youtube_api.quota_usage
                    st.session_state.campaign_responses = self.generate_responses(
                        config, campaign_video, st.session_state.current_comments)
                    st.session_state.selected_responses = {
                        i: True for i in range(len(st.session_state.campaign_responses))}
                    quota_used = self.youtube_api.quota_usage - initial_quota
                    st.info(t("automarket_quota_consumed").format(
                        units=quota_used))

            self.log_rejected_videos(
                "monitor", st.session_state.rejected_videos)
            self.log_rejected_comments("monitor")

            if st.session_state.campaign_responses:
                self.display_responses("monitor", config, campaign_video,
                                       st.session_state.campaign_responses, st.session_state.campaign_timestamp)
