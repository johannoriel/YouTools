from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from lib.youtube_api import YoutubeAPI
from typing import List, Dict, Any
from datetime import datetime
import pytz
from lib.youtube_db import add_target_channel, get_response_moderation_status

class CommonAutomarketWidget(Widget):
    def __init__(self, name, prefix, plugin_manager, youtube_api: YoutubeAPI):
        super().__init__(name, prefix, plugin_manager)
        self.youtube_api = youtube_api

    def log_selected_videos(self, videos: List[Dict[str, Any]], keyword: str = "N/A"):
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

    def log_rejected_videos(self, rejected_videos: List[Dict[str, Any]]):
        with st.expander(f"{t('automarket_rejected_videos')} ({len(rejected_videos)})"):
            for rejected in rejected_videos:
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

    def log_rejected_comments(self, excluded_comments: List[Dict[str, Any]]):
        if excluded_comments:
            with st.expander(f"{t('automarket_excluded_comments')} ({len(excluded_comments)})"):
                for excluded in excluded_comments:
                    st.write(f"Comment: {excluded['text']}")
                    st.write(f"Author: {excluded['author']}")
                    st.write(
                        f"Video: [{excluded['video_title']}](https://www.youtube.com/watch?v={excluded['video_id']})")
                    st.write(f"Published: {excluded['published_at']}")
                    st.write("---")

    def display_responses(self, config: dict, campaign_video: Dict[str, Any], responses: List[Dict[str, Any]], campaign_timestamp: str, selected_responses: dict, post_responses_callback):
        st.subheader(t("automarket_responses"))
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("automarket_expand_all"), key=f"{self.prefix}_expand_all"):
                st.session_state.expand_all = True
        with col2:
            if st.button(t("automarket_collapse_all"), key=f"{self.prefix}_collapse_all"):
                st.session_state.expand_all = False

        default_prompt = config['response_prompt']
        new_prompt = st.text_area(
            "Nouveau prompt pour regénérer les réponses",
            value=default_prompt,
            height=150,
            key=f"{self.prefix}_regen_prompt"
        )

        if st.button("Regénérer les réponses", key=f"{self.prefix}_regen_button"):
            if 'current_comments' in st.session_state:
                with st.spinner("Regénération des réponses..."):
                    original_prompt = config['response_prompt']
                    config['response_prompt'] = new_prompt
                    st.session_state.campaign_responses = self.generate_responses(
                        config, campaign_video, st.session_state.current_comments)
                    st.session_state.selected_responses = {
                        i: True for i in range(len(st.session_state.campaign_responses))}
                    config['response_prompt'] = original_prompt
                    st.success("Réponses regénérées avec succès !")
            else:
                st.warning(
                    "Aucune campagne précédente trouvée pour regénération.")

        col3, col4 = st.columns(2)
        with col3:
            if st.button(t("automarket_select_all"), key=f"{self.prefix}_select_all"):
                for i in range(len(st.session_state.campaign_responses)):
                    st.session_state.selected_responses[i] = True
                st.rerun()
        with col4:
            if st.button(t("automarket_deselect_all"), key=f"{self.prefix}_deselect_all"):
                for i in range(len(st.session_state.campaign_responses)):
                    st.session_state.selected_responses[i] = False
                st.rerun()

        for i, response in enumerate(responses):
            title = f"{t('automarket_reponse_to_comment').format(i+1)} : {response['video_title']}"
            with st.expander(title, expanded=st.session_state.expand_all):
                st.write(f"{t('automarket_channel')}: {response['channel_title']} "
                         f"({self.youtube_api.format_count(response['subscriber_count'])} subscribers)")
                if st.button(t("automarket_add_to_trusted"), key=f"{self.prefix}_add_trusted_{i}"):
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

                current_value = selected_responses.get(i, False)
                new_value = st.checkbox(
                    t("automarket_exclude_response"),
                    value=current_value,
                    key=f"{self.prefix}_exclude_{i}"
                )
                selected_responses[i] = new_value

        if st.button(t("automarket_post_responses"), key=f"{self.prefix}_post_responses"):
            with st.spinner(t("automarket_posting")):
                selected = [r for i, r in enumerate(responses)
                            if not selected_responses.get(i, False)]
                post_responses_callback(selected, campaign_timestamp)

    def generate_responses(self, config: dict, campaign_video: Dict[str, Any], comments: List[Dict[str, Any]], max_comments_debug: int = None):
        responses = []
        total_comments = min(
            len(comments), max_comments_debug) if max_comments_debug else len(comments)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        exclusion_keyword = config['exclusion_keyword'].strip()

        prompt = config['response_prompt'].format(
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
