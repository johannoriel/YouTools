from global_vars import translations, t
from app import Plugin
import streamlit as st
from youtube_api import YoutubeAPI
from youtube_db import *
from plugins.ragllm import RagllmPlugin
from typing import List, Dict, Any, Optional
from datetime import datetime
import pytz

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
})


class AutomarketPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.youtube_api = YoutubeAPI(self.plugin_manager.config)
        self.ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'campaign_responses' not in st.session_state:
            st.session_state.campaign_responses = []
        if 'selected_responses' not in st.session_state:
            st.session_state.selected_responses = {}
        if 'rejected_videos' not in st.session_state:
            st.session_state.rejected_videos = []

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
            }
        }

    def get_tabs(self):
        return [{"name": t("automarket_tab"), "plugin": "automarket"}]

    def fetch_videos_for_keyword(self, keyword: str, max_videos: int, min_subscribers: int, expiry_days: int, view_threshold: int) -> List[Dict[str, Any]]:
        """Récupère les vidéos pour un mot-clé avec filtres."""
        videos = []
        for order in ["relevance", "date"]:
            search_results = self.youtube_api.search_videos(
                keyword, max_videos * 2, order=order)
            for video in search_results:
                published_at = datetime.strptime(
                    video['published_at'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
                days_old = (datetime.now(pytz.UTC) - published_at).days
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
                if days_old > 1 and video['view_count'] < view_threshold:
                    reject_reason.append(
                        f"Views ({video['view_count']} < {view_threshold})")

                if not reject_reason:
                    videos.append(video)
                else:
                    st.session_state.rejected_videos.append({
                        'url': video['url'],
                        'channel_url': f"https://www.youtube.com/channel/{video['channel_id']}",
                        'reason': ", ".join(reject_reason),
                        'stats': {
                            'subscribers': video['subscriber_count'],
                            'views': video['view_count'],
                            'days_old': days_old,
                            'last_comment_days': last_comment_days
                        }
                    })

                if len(videos) >= max_videos:
                    break
            if len(videos) >= max_videos:
                break
        return videos[:max_videos]

    def fetch_videos_from_trusted_channels(self, keyword: str, max_videos: int, min_subscribers: int, expiry_days: int, view_threshold: int) -> List[Dict[str, Any]]:
        """Récupère les vidéos récentes des chaînes de confiance pour un mot-clé."""
        trusted_channels = [
            ch for ch in get_target_channels() if keyword in ch['keywords']]
        videos = []
        for channel in trusted_channels:
            channel_videos = self.youtube_api.get_channel_recent_videos(
                channel['channel_id'], max_results=max_videos * 2)
            for video in channel_videos:
                published_at = datetime.strptime(
                    video['published_at'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.UTC)
                days_old = (datetime.now(pytz.UTC) - published_at).days
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
                if days_old > 1 and video['view_count'] < view_threshold:
                    reject_reason.append(
                        f"Views ({video['view_count']} < {view_threshold})")

                if not reject_reason:
                    videos.append(video)
                else:
                    st.session_state.rejected_videos.append({
                        'url': video['url'],
                        'channel_url': f"https://www.youtube.com/channel/{video['channel_id']}",
                        'reason': ", ".join(reject_reason),
                        'stats': {
                            'subscribers': video['subscriber_count'],
                            'views': video['view_count'],
                            'days_old': days_old,
                            'last_comment_days': last_comment_days
                        }
                    })

                if len(videos) >= max_videos:
                    break
            if len(videos) >= max_videos:
                break
        return videos[:max_videos]

    def generate_responses(self, config, campaign_video: Dict[str, Any], comments: List[Dict[str, Any]]):
        """Génère les réponses pour les commentaires (réutilisation de Marketyoutube)."""
        responses = []
        total_comments = len(comments)
        progress_bar = st.progress(0)
        progress_text = st.empty()

        prompt = config['automarket']['response_prompt'].format(
            url=campaign_video['url'],
            transcript=campaign_video.get('transcript', '')
        )

        for idx, comment in enumerate(comments):
            progress = (idx + 1) / total_comments
            progress_bar.progress(progress)
            progress_text.text(
                t("automarket_progress").format(idx + 1, total_comments))

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
                    'comment_text': comment['text'],
                    'channel_id': comment['channel_id'],
                    'channel_title': comment['channel_title'],
                    'video_title': comment['video_title'],
                    'view_count': comment.get('view_count', 0),
                    'like_count': comment.get('like_count', 0),
                    'comment_count': comment.get('comment_count', 0),
                    'days_old': comment.get('days_old', 0),
                    'subscriber_count': comment.get('subscriber_count', 0)
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
                    'subscriber_count': comment.get('subscriber_count', 0)
                })

        progress_bar.empty()
        progress_text.empty()
        return responses

    def post_responses(self, selected_responses):
        """Poste les réponses sélectionnées."""
        for response in selected_responses:
            try:
                self.youtube_api.post_comment_reply(
                    response['comment_id'], response['response'])
            except Exception as e:
                st.error(
                    f"Error posting response to {response['comment_id']}: {str(e)}")

    def run(self, config):
        st.header(t("automarket_header"))

        # 1. Sélection de la vidéo à promouvoir (uniquement celles avec mots-clés)
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

        # 3. Lancement de la campagne
        if st.button(t("automarket_start_campaign")):
            with st.spinner(t("automarket_processing")):
                st.session_state.rejected_videos = []
                target_videos = []

                # Recherche par mot-clé
                for keyword in campaign_video['keywords']:
                    videos = self.fetch_videos_for_keyword(
                        keyword, max_videos_per_keyword, min_subscribers, expiry_days, view_threshold)
                    target_videos.extend(videos)

                    # Recherche dans les chaînes de confiance
                    trusted_videos = self.fetch_videos_from_trusted_channels(
                        keyword, trusted_channel_videos, min_subscribers, expiry_days, view_threshold)
                    target_videos.extend(trusted_videos)

                # Récupération des commentaires
                comments = []
                for video in target_videos:
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
                    comments.extend(video_comments)

                # Génération des réponses
                st.session_state.campaign_responses = self.generate_responses(
                    config, campaign_video, comments)
                st.session_state.selected_responses = {
                    i: True for i in range(len(st.session_state.campaign_responses))}

        # 4. Affichage des résultats
        if st.session_state.rejected_videos:
            with st.expander(t("automarket_rejected_videos")):
                for rejected in st.session_state.rejected_videos:
                    st.write(f"Video: [{rejected['url']}]({rejected['url']})")
                    st.write(
                        f"Channel: [{rejected['channel_url']}]({rejected['channel_url']})")
                    st.write(t("automarket_rejected_reason").format(
                        rejected['reason']))
                    st.write(f"Stats: {rejected['stats']}")
                    st.write("---")

        if st.session_state.campaign_responses:
            st.subheader(t("automarket_responses"))
            for i, response in enumerate(st.session_state.campaign_responses):
                with st.expander(f"Response {i+1}"):
                    st.write(f"{t('automarket_channel')}: {response['channel_title']} "
                             f"({self.youtube_api.format_count(response['subscriber_count'])} subscribers)")
                    if st.button(t("automarket_add_to_trusted"), key=f"add_trusted_{i}"):
                        channel_url = f"https://www.youtube.com/channel/{response['channel_id']}"
                        add_target_channel(
                            channel_id=response['channel_id'],
                            channel_title=response['channel_title'],
                            channel_url=channel_url,
                            keywords=[keyword for keyword in campaign_video['keywords'] if keyword in response['video_title'].lower(
                            ) or keyword in response['comment_text'].lower()],
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
                    st.write(
                        f"{t('automarket_comment')}: {response['comment_text']}")
                    st.write(
                        f"{t('automarket_response')}: {response['response']}")
                    st.session_state.selected_responses[i] = st.checkbox(
                        t("automarket_exclude_response"),
                        value=not st.session_state.selected_responses[i],
                        key=f"exclude_{i}"
                    )

            # 5. Bouton pour poster les réponses
            if st.button(t("automarket_post_responses")):
                with st.spinner(t("automarket_posting")):
                    selected_responses = [r for i, r in enumerate(st.session_state.campaign_responses)
                                          if not st.session_state.selected_responses[i]]
                    self.post_responses(selected_responses)
                    st.success(t("automarket_campaign_complete"))
