from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from lib.youtube_api import YoutubeAPI
from lib.youtube_db import get_videos, get_target_channels
from plugins.promoteyoutube import PromoteyoutubePlugin
import pandas as pd
import os
from datetime import datetime

class MarketOneVideoWidget(Widget):
    def __init__(self, name, prefix, plugin_manager, campaign_keywords, max_campaign_videos, max_campaign_comments, response_prompt):
        super().__init__(name, prefix, plugin_manager)
        self.youtube_api = YoutubeAPI(self.plugin_manager.config)
        self.campaign_keywords = campaign_keywords
        self.max_campaign_videos = max_campaign_videos
        self.max_campaign_comments = max_campaign_comments
        self.response_prompt = response_prompt

    def display(self):
        st.header(t("marketyoutube_header_campaigns"))

        # Récupérer toutes les vidéos
        videos = get_videos()

        col1, col2 = st.columns([1,4])

        # Ajouter un champ de recherche pour filtrer les vidéos à promouvoir
        search_keyword = col1.text_input(
            "Search video by keyword",
            value="",
            key=f"{self.prefix}_campaign_video_search_keyword",
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
            col2.warning("No videos match your search keyword.")
            selected_video_title = None
            campaign_video = None
        else:
            selected_video_title = col2.selectbox(
                t("marketyoutube_select_video"),
                options=list(video_options.keys()),
                key=f"{self.prefix}_campaign_select_video"
            )
            campaign_video = video_options.get(selected_video_title)

        if "campaign_target_videos" not in st.session_state:
            st.session_state["campaign_target_videos"] = []

        target_source = st.radio(
            "Target Source",
            options=["Search by Keywords", "Target Channels", t("marketyoutube_target_source_csv")],
            index=0,
            key=f"{self.prefix}_campaign_target_source"
        )

        if target_source == "Search by Keywords":
            keywords = st.text_input(
                t("marketyoutube_keywords"),
                value=self.campaign_keywords,
                key=f"{self.prefix}_campaign_keywords_search"
            )
            max_videos = st.number_input(
                t("marketyoutube_max_videos"),
                min_value=1,
                max_value=50,
                value=self.max_campaign_videos,
                key=f"{self.prefix}_campaign_max_videos_search"
            )
        elif target_source == "Target Channels":
            target_channels = get_target_channels()
            if not target_channels:
                st.warning("No target channels available. Please add some in the Channel Manager tab.")
                return

            all_keywords = set()
            for channel in target_channels:
                all_keywords.update(channel['keywords'])
            all_keywords = sorted(list(all_keywords))

            selected_keywords = st.multiselect(
                "Select Keywords to Filter Channels",
                options=all_keywords,
                key=f"{self.prefix}_campaign_keywords_filter"
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
                key=f"{self.prefix}_campaign_select_channels"
            )

            max_videos_per_channel = st.number_input(
                "Max Videos per Channel",
                min_value=1,
                max_value=50,
                value=5,
                key=f"{self.prefix}_campaign_max_videos_per_channel"
            )
            keywords = f"trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif target_source == t("marketyoutube_target_source_csv"):
            csv_file_path = os.path.join(
                self.plugin_manager.config['common']['work_directory'], 'video_list.csv')
            if os.path.exists(csv_file_path):
                try:
                    df = pd.read_csv(csv_file_path)
                    st.write(f"Found {len(df)} videos in CSV file")

                    csv_videos = []
                    for _, row in df.iterrows():
                        try:
                            video_id = row['url'].split('v=')[-1].split('&')[0]
                            # Ensure date is parsed correctly to ISO format
                            published_at = parse_date(row['published_at'])

                            video = {
                                'video_id': video_id,
                                'id': video_id,
                                'title': row['title'],
                                'url': row['url'],
                                'views': int(row['view_count']) if pd.notna(row['view_count']) and str(row['view_count']).isdigit() else 0,
                                'language': row['language'],
                                'published_at': published_at,  # Already in ISO format
                                'keyword': row['keyword']
                            }
                            video_complete = self.youtube_api.get_video_infos(video) | video
                            csv_videos.append(video_complete)
                        except Exception as e:
                            st.warning(f"Error processing video {row['url']}: {str(e)}")
                            continue

                    # Filtrer par mot-clé si nécessaire
                    unique_keywords = df['keyword'].unique().tolist()
                    selected_csv_keywords = st.multiselect(
                        "Filter by Keyword",
                        options=unique_keywords,
                        default=unique_keywords,
                        key=f"{self.prefix}_csv_keyword_filter"
                    )

                    filtered_csv_videos = [
                        v for v in csv_videos if v['keyword'] in selected_csv_keywords]
                    keywords = unique_keywords[0]

                    # Afficher les vidéos disponibles avec leurs mots-clés
                    st.subheader("Videos from CSV")
                    st.dataframe(df)

                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    filtered_csv_videos = []
            else:
                st.warning(f"CSV file not found at: {csv_file_path}")
                filtered_csv_videos = []

        max_comments = st.number_input(
            t("marketyoutube_max_comments"),
            min_value=1,
            max_value=10,
            value=self.max_campaign_comments,
            key=f"{self.prefix}_campaign_max_comments"
        )

        if st.button(t("marketyoutube_start_campaign"), key=f"{self.prefix}_campaign_start_button"):
            if not campaign_video:
                st.error("Please select a video to promote.")
            else:
                with st.spinner(t("marketyoutube_searching")):
                    if target_source == "Search by Keywords":
                        target_videos = self.youtube_api.search_videos(keywords, max_videos)
                    elif target_source == "Target Channels":
                        target_videos = []
                        for channel in filtered_channels:
                            if f"{channel['channel_title']} ({', '.join(channel['keywords']) if channel['keywords'] else '--'}) ({channel['subscriber_count']} subscribers)" in selected_channels:
                                channel_videos = self.youtube_api.get_channel_recent_videos(
                                    channel['channel_id'],
                                    max_results=max_videos_per_channel
                                )
                                target_videos.extend(channel_videos)
                    else:  # Video list (CSV)
                        target_videos = filtered_csv_videos

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
            promoteyoutube = PromoteyoutubePlugin("promoteyoutube", self.plugin_manager)
            promoteyoutube.run_campaign(
                config=self.plugin_manager.config,
                target_videos=st.session_state["campaign_target_videos"],
                campaign_video=campaign_video,
                max_comments=max_comments,
                prefix="campaign_",
                keywords=keywords
            )

def parse_date(date_str):
    formats = [
        "%Y-%m-%d %H:%M:%S",  # Format in your CSV
        "%Y-%m-%dT%H:%M:%SZ",  # ISO format expected by the app
        "%Y-%m-%d",           # Just date
        "%d/%m/%Y %H:%M:%S",  # European format
    ]

    for fmt in formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            # Always return ISO format
            return parsed_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue

    # Fallback: Use the first part of the string as a date and assume midnight
    try:
        return f"{date_str.split(' ')[0]}T00:00:00Z"
    except Exception as e:
        raise ValueError(f"Could not parse date: {date_str}, error: {str(e)}")
