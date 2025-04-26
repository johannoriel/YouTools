from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from lib.youtube_api import YoutubeAPI
from lib.youtube_db import get_target_channels, get_videos
from typing import List, Dict, Any
from datetime import datetime
import pytz
from .common_automarket import CommonAutomarketWidget

class YoutubeTrendWatcherWidget(Widget):
    def __init__(self, name, prefix, plugin_manager, youtube_api: YoutubeAPI, config_params: dict, post_responses_callback):
        super().__init__(name, prefix, plugin_manager)
        self.youtube_api = youtube_api
        self.config_params = config_params
        self.post_responses_callback = post_responses_callback
        self.common_widget = CommonAutomarketWidget(
            "common_automarket", f"{prefix}_common", plugin_manager, youtube_api)

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
        if 'target_videos' not in st.session_state:
            st.session_state.target_videos = []
        if 'current_comments' not in st.session_state:
            st.session_state.current_comments = []
        if 'campaign_timestamp' not in st.session_state:
            st.session_state.campaign_timestamp = datetime.now(
                pytz.UTC).isoformat()

    def fetch_monitor_videos(self, selected_keyword: str, max_videos_per_keyword: int, expiry_days: int, min_subscribers: int, view_threshold: int, search_time: bool, search_relevant: bool, search_trusted: bool) -> List[Dict[str, Any]]:
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

    def display_quota(self):
        quota_info = self.youtube_api.get_quota_usage()
        st.markdown(
            f"Quota restant : {quota_info['remaining_percentage']:.2f}% restant ({quota_info['quota_usage']} unités sur {quota_info['quota_limit']}) [Check](https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas?hl=fr&inv=1&invt=AbrUGw&pageState=(%22allQuotasTable%22%253A(%22c%22%253A%5B%22displayDimensions%22%5D)))")

    def display(self):
        self._initialize_session_state()
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
            key=f"{self.prefix}_keyword"
        )

        st.subheader(t("monitor_trends_search_criteria"))
        search_time = st.checkbox(
            t("monitor_trends_time"), value=True, key=f"{self.prefix}_time")
        search_relevant = st.checkbox(
            t("monitor_trends_relevant"), value=True, key=f"{self.prefix}_relevant")
        search_trusted = st.checkbox(
            t("monitor_trends_trusted"), value=True, key=f"{self.prefix}_trusted")

        comments_per_video = st.number_input(
            t("automarket_comments_per_video"),
            min_value=1,
            value=int(self.config_params['comments_per_video']),
            key=f"{self.prefix}_comments_per_video"
        )
        min_subscribers = st.number_input(
            t("automarket_min_subscribers"),
            min_value=0,
            value=int(self.config_params['min_subscribers']),
            key=f"{self.prefix}_min_subscribers"
        )
        max_videos_per_keyword = st.number_input(
            t("automarket_max_videos_per_keyword"),
            min_value=1,
            value=int(self.config_params['max_videos_per_keyword']),
            key=f"{self.prefix}_max_videos_per_keyword"
        )
        expiry_days = st.number_input(
            t("automarket_expiry_days"),
            min_value=1,
            value=int(self.config_params['expiry_days']),
            key=f"{self.prefix}_expiry_days"
        )
        view_threshold = st.number_input(
            t("automarket_view_threshold"),
            min_value=0,
            value=int(self.config_params['view_threshold']),
            key=f"{self.prefix}_view_threshold"
        )

        self.display_quota()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fetch_videos_btn = st.button(
                "Récupérer les vidéos", key=f"{self.prefix}_fetch_videos")
        with col2:
            fetch_comments_btn = st.button(
                "Récupérer les commentaires", key=f"{self.prefix}_fetch_comments")
        with col3:
            generate_responses_btn = st.button(
                "Générer les réponses", key=f"{self.prefix}_generate_responses")
        with col4:
            start_campaign_btn = st.button(
                t("automarket_start_campaign"), key=f"{self.prefix}_start_campaign")

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
                st.session_state.rejected_videos = []
                st.session_state.target_videos = self.fetch_monitor_videos(
                    selected_keyword, max_videos_per_keyword, expiry_days, min_subscribers,
                    view_threshold, search_time, search_relevant, search_trusted)
                self.common_widget.log_selected_videos(
                    st.session_state.target_videos, selected_keyword)
                quota_used = self.youtube_api.quota_usage - initial_quota
                st.info(t("automarket_quota_consumed").format(
                    units=quota_used))

        if st.session_state.target_videos:
            self.common_widget.log_selected_videos(
                st.session_state.target_videos, selected_keyword)

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
                st.session_state.campaign_responses = self.common_widget.generate_responses(
                    self.config_params, campaign_video, st.session_state.current_comments)
                st.session_state.selected_responses = {
                    i: True for i in range(len(st.session_state.campaign_responses))}
                quota_used = self.youtube_api.quota_usage - initial_quota
                st.info(t("automarket_quota_consumed").format(
                    units=quota_used))

        self.common_widget.log_rejected_videos(
            st.session_state.rejected_videos)
        self.common_widget.log_rejected_comments(
            st.session_state.excluded_comments)

        if st.session_state.campaign_responses:
            self.common_widget.display_responses(
                self.config_params, campaign_video,
                st.session_state.campaign_responses,
                st.session_state.campaign_timestamp,
                st.session_state.selected_responses,
                self.post_responses_callback)
