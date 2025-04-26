from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from lib.youtube_api import YoutubeAPI
from lib.youtube_db import get_videos
from typing import List, Dict, Any
from datetime import datetime
import pytz
from .common_automarket import CommonAutomarketWidget
from langdetect import detect

class AutoCampaignWidget(Widget):
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
        if 'campaign_target_videos' not in st.session_state:
            st.session_state.campaign_target_videos = []
        if 'campaign_current_comments' not in st.session_state:
            st.session_state.campaign_current_comments = []
        if 'campaign_timestamp' not in st.session_state:
            st.session_state.campaign_timestamp = datetime.now(
                pytz.UTC).isoformat()

    def fetch_videos_for_keyword(self, keyword: str, max_videos: int, min_subscribers: int, expiry_days: int, view_threshold: int, combine_keywords: bool = False) -> List[Dict[str, Any]]:
        videos = []
        for order in ["relevance", "date"]:
            search_results = self.youtube_api.search_videos(
                keyword, max_videos * 2, order=order, language=st.session_state.lang, combine_keywords=combine_keywords)
            st.info(
                f"Nombre de vidéos trouvées par l'API pour '{keyword}' (ordre: {order}) : {len(search_results)}")
            rejected = 0

            for video in search_results:
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
        from lib.youtube_db import get_target_channels
        trusted_channels = [
            ch for ch in get_target_channels() if keyword in ch['keywords']]
        videos = []
        for channel in trusted_channels:
            channel_videos = self.youtube_api.get_channel_recent_videos(
                channel['channel_id'], max_results=max_videos * 2)
            for video in channel_videos:
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

    def fetch_campaign_videos(self, campaign_video: Dict[str, Any], max_videos_per_keyword: int, min_subscribers: int, expiry_days: int, view_threshold: int, combine_keywords: bool, search_keywords: bool, search_trusted: bool, trusted_channel_videos: int) -> List[Dict[str, Any]]:
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

    def display_quota(self):
        quota_info = self.youtube_api.get_quota_usage()
        st.markdown(
            f"Quota restant : {quota_info['remaining_percentage']:.2f}% restant ({quota_info['quota_usage']} unités sur {quota_info['quota_limit']}) [Check](https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas?hl=fr&inv=1&invt=AbrUGw&pageState=(%22allQuotasTable%22%253A(%22c%22%253A%5B%22displayDimensions%22%5D)))")

    def display(self):
        self._initialize_session_state()
        st.header(t("automarket_header"))

        videos = [v for v in get_videos() if v['keywords']]
        if not videos:
            st.warning(t("automarket_no_videos_with_keywords"))
            return

        video_options = {
            f"{v['title']} ({', '.join(v['keywords'])})": v for v in videos}
        selected_video_title = st.selectbox(
            t("automarket_select_video"),
            options=list(video_options.keys()),
            key=f"{self.prefix}_select_video"
        )
        campaign_video = video_options[selected_video_title]

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
        trusted_channel_videos = st.number_input(
            t("automarket_trusted_channel_videos"),
            min_value=1,
            value=int(self.config_params['trusted_channel_videos']),
            key=f"{self.prefix}_trusted_channel_videos"
        )

        debug_mode = st.checkbox(
            t("automarket_debug_mode"), value=False, key=f"{self.prefix}_debug_mode")
        max_comments_debug = None
        if debug_mode:
            max_comments_debug = st.number_input(
                t("automarket_max_comments_debug"),
                min_value=1,
                value=5,
                key=f"{self.prefix}_max_comments_debug"
            )

        combine_keywords = st.checkbox(
            "Combine Keywords", value=False, key=f"{self.prefix}_combine_keywords")
        search_keywords = st.checkbox(
            t("automarket_search_keywords"), value=True, key=f"{self.prefix}_search_keywords")
        search_trusted = st.checkbox(
            t("automarket_search_trusted"), value=True, key=f"{self.prefix}_search_trusted")

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

        if fetch_videos_btn or start_campaign_btn:
            with st.spinner(t("automarket_processing")):
                st.session_state.campaign_timestamp = datetime.now(
                    pytz.UTC).isoformat()
                initial_quota = self.youtube_api.quota_usage
                st.session_state.rejected_videos = []
                st.session_state.campaign_target_videos = self.fetch_campaign_videos(
                    campaign_video, max_videos_per_keyword, min_subscribers, expiry_days,
                    view_threshold, combine_keywords, search_keywords, search_trusted,
                    trusted_channel_videos)
                quota_used = self.youtube_api.quota_usage - initial_quota
                st.info(t("automarket_quota_consumed").format(
                    units=quota_used))

        if st.session_state.campaign_target_videos:
            self.common_widget.log_selected_videos(
                st.session_state.campaign_target_videos)

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
                st.session_state.campaign_responses = self.common_widget.generate_responses(
                    self.config_params, campaign_video, st.session_state.campaign_current_comments,
                    max_comments_debug if debug_mode else None)
                st.session_state.selected_responses = {
                    i: not debug_mode for i in range(len(st.session_state.campaign_responses))}
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
