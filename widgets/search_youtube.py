# File: widgets/search_youtube.py
from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
import os
from lib.youtube_api import YoutubeAPI

# Add translations
translations["en"].update({
    "promoteyoutube_keywords": "Keywords to Search",
    "promoteyoutube_search": "Search Videos",
    "promoteyoutube_searching": "Searching videos...",
    "promoteyoutube_max_videos_label": "Number of videos to search",
    "promoteyoutube_keywords_warning": "Please enter keywords to search for videos.",
    "promoteyoutube_export_success": "Exported successfully to {}",
    "promoteyoutube_export_error": "Error during export: {}",
    "promoteyoutube_overwrite_checkbox": "Overwrite existing file",
    "promoteyoutube_save_videos": "Save Video List",
    "promoteyoutube_videos_found": "Videos Found",
})

translations["fr"].update({
    "promoteyoutube_keywords": "Mots-clés à rechercher",
    "promoteyoutube_search": "Rechercher des vidéos",
    "promoteyoutube_searching": "Recherche des vidéos...",
    "promoteyoutube_max_videos_label": "Nombre de vidéos à rechercher",
    "promoteyoutube_keywords_warning": "Veuillez entrer des mots-clés pour la recherche.",
    "promoteyoutube_export_success": "Exporté avec succès vers {}",
    "promoteyoutube_export_error": "Erreur lors de l'export : {}",
    "promoteyoutube_overwrite_checkbox": "Écraser le fichier existant",
    "promoteyoutube_save_videos": "Enregistrer la liste des vidéos",
    "promoteyoutube_videos_found": "Vidéos trouvées",
})

class SearchYoutubeWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def search_videos(self, keywords: str, max_videos: int, video_order: str):
        youtube_api = YoutubeAPI(self.plugin_manager.config)
        videos = youtube_api.search_videos(
            keywords,
            max_videos,
            order=video_order,
            language=st.session_state.lang
        )
        return videos

    def export_videos(self, videos, work_dir, overwrite):
        try:
            base_filename = "video_list.csv"
            output_path = os.path.join(work_dir, base_filename)

            if not overwrite and os.path.exists(output_path):
                i = 1
                while True:
                    new_filename = f"video_list_{i:03d}.csv"
                    new_output_path = os.path.join(work_dir, new_filename)
                    if not os.path.exists(new_output_path):
                        output_path = new_output_path
                        break
                    i += 1

            videos_data = [
                {
                    'video_id': video.get('video_id', ''),
                    'title': video.get('title', ''),
                    'url': video.get('url', ''),
                    'channel_id': video.get('channel_id', ''),
                    'channel_title': video.get('channel_title', ''),
                    'view_count': video.get('view_count', 0),
                    'comment_count': video.get('comment_count', 0),
                    'published_at': video.get('published_at', ''),
                    'language': video.get('language', ''),
                    'relevance_score': video.get('relevance_score', 0),
                    'subscriber_count': video.get('subscriber_count', 0),
                    'keyword': st.session_state.get('keyword', '')
                }
                for video in videos
            ]
            df = pd.DataFrame(videos_data)
            df.to_csv(output_path, index=False)
            st.success(t("promoteyoutube_export_success").format(output_path))
        except Exception as e:
            st.error(t("promoteyoutube_export_error").format(str(e)))

    def display(self, config):
        work_dir = config['common']['work_directory']
        max_videos = config['promoteyoutube']['max_videos']

        keywords = st.text_input(
            t("promoteyoutube_keywords"), key=f"{self.prefix}_keywords")

        max_videos_input = st.number_input(
            t("promoteyoutube_max_videos_label"),
            min_value=1,
            max_value=50,
            value=int(max_videos),
            key=f"{self.prefix}_max_videos"
        )

        if st.button(t("promoteyoutube_search"), key=f"{self.prefix}_search"):
            if keywords:
                with st.spinner(t("promoteyoutube_searching")):
                    st.session_state['keyword'] = keywords
                    videos = self.search_videos(keywords, max_videos_input, "relevance")
                    st.session_state[f'{self.prefix}_found_videos'] = videos
            else:
                st.warning(t("promoteyoutube_keywords_warning"))

        if f'{self.prefix}_found_videos' in st.session_state and st.session_state[f'{self.prefix}_found_videos']:
            st.subheader(t("promoteyoutube_videos_found"))
            videos_df = pd.DataFrame([
                {
                    'title': video.get('title', ''),
                    'url': video.get('url', ''),
                    'channel_title': video.get('channel_title', ''),
                    'view_count': video.get('view_count', 0),
                    'comment_count': video.get('comment_count', 0),
                    'language': video.get('language', ''),
                    'relevance_score': video.get('relevance_score', 0),
                    'subscriber_count': video.get('subscriber_count', 0),
                }
                for video in st.session_state[f'{self.prefix}_found_videos']
            ])

            column_config = {
                "title": st.column_config.TextColumn("Title", width="large"),
                "url": st.column_config.LinkColumn(
                    "Video URL",
                    help="Click to visit the video",
                    display_text="Visit",
                    width="small"
                ),
                "channel_title": st.column_config.TextColumn("Channel", width="medium"),
                "view_count": st.column_config.NumberColumn("Views", width="small"),
                "comment_count": st.column_config.NumberColumn("Comments", width="small"),
                "language": st.column_config.TextColumn("Language", width="small"),
                "relevance_score": st.column_config.NumberColumn("Relevance", width="small"),
                "subscriber_count": st.column_config.NumberColumn("Subscribers", width="small"),
            }

            st.dataframe(
                videos_df,
                column_config=column_config,
                width='stretch',
                height=400,
                key=f"{self.prefix}_videos_dataframe"
            )

            overwrite = st.checkbox(
                t("promoteyoutube_overwrite_checkbox"), key=f"{self.prefix}_overwrite")

            if st.button(t("promoteyoutube_save_videos"), key=f"{self.prefix}_save_videos"):
                self.export_videos(
                    st.session_state[f'{self.prefix}_found_videos'], work_dir, overwrite)

        return st.session_state.get(f'{self.prefix}_found_videos', [])
