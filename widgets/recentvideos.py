from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript
from widgets.yt_transcript import YoutubeTranscriptWidget
from lib.youtube_api import YoutubeAPI
from typing import Dict, Optional

translations["en"].update({
    "recent_videos_header": "10 Latest Videos",
    "recent_videos_transcript_button": "Transcript",
    "recent_videos_transcript_header": "Transcript",
    "recent_videos_transcript_language": "Transcript Language:",
    "recent_videos_transcript_content": "Transcript Content",
    "recent_videos_save_transcript": "Save transcript",
    "recent_videos_save_success": "Transcript saved successfully!",
    "recent_videos_configure_channel_id": "Please configure the channel ID in the Configuration tab.",
    "recent_videos_transcript_error": "An error occurred while retrieving the transcript: ",
})

translations["fr"].update({
    "recent_videos_header": "10 dernières vidéos",
    "recent_videos_transcript_button": "Transcript",
    "recent_videos_transcript_header": "Transcript",
    "recent_videos_transcript_language": "Langue de la transcription :",
    "recent_videos_transcript_content": "Contenu du transcript",
    "recent_videos_save_transcript": "Sauver le transcript",
    "recent_videos_save_success": "Transcript sauvée avec succès!",
    "recent_videos_configure_channel_id": "Veuillez configurer l'ID de la chaîne dans l'onglet Configuration.",
    "recent_videos_transcript_error": "Une erreur s'est produite lors de la récupération du transcript : ",
})

class RecentVideosWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def get_transcript(self, video_id, language):
        ytt = self.yt_transcript_widget = YoutubeTranscriptWidget(f"{self.prefix}transcript", f"{self.prefix}transcript", self.plugin_manager)
        return ytt.fetch_transcript(f"https://www.youtube.com/watch?v={video_id}"), ""
        try:
            st.write(f"Fetching {video_id} transcript in {language}...")
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=[language])
            st.success(f"Transcript fetched successfully!")
        except CouldNotRetrieveTranscript:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                st.success(f"Transcript fetched successfully after retry!")
            except Exception as e:
                return f"{t('recent_videos_transcript_error')}{str(e)}", "N/A"
        full_transcript = " ".join([entry['text'] for entry in transcript])
        return full_transcript, language

    def save_transcript(self, transcript, video_id, work_directory):
        with open(os.path.join(work_directory, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)
        with open(os.path.join(work_directory, "url.txt"), "w", encoding="utf-8") as f:
            f.write(f"https://www.youtube.com/watch?v={video_id}")
        return True

    def display(self, config):
        st.header(t("recent_videos_header"))
        youtube_api = YoutubeAPI(config)
        work_directory = config['common']['work_directory']
        language = config['common']['language']

        if 'channel_id' not in config['common'] or not config['common']['channel_id']:
            st.info(t("recent_videos_configure_channel_id"))
            return

        if 'all_videos' not in st.session_state:
            st.session_state.all_videos = youtube_api.get_channel_videos(
                config['common']['channel_id'])

        filter_keywords = st.text_input(
            "Filtrer les vidéos par mots-clés", key=f"{self.prefix}_filter")
        filtered_videos = [video for video in st.session_state.all_videos
                           if not filter_keywords or filter_keywords.lower() in video['title'].lower()]

        page_size = 10
        page_number = st.session_state.get(f"{self.prefix}_page_number", 0)
        total_pages = (len(filtered_videos) + page_size - 1) // page_size
        start_index = page_number * page_size
        videos_to_display = filtered_videos[start_index:start_index + page_size]

        for video in videos_to_display:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.image(video['thumbnail'])
            with col2:
                st.subheader(video['title'])
                st.markdown(
                    f"[Voir la vidéo](https://www.youtube.com/watch?v={video['video_id']})")
                st.write(f"Statut : {video['status']}")
                if video['is_short']:
                    st.write("**Short** 🎥")
            with col3:
                if st.button(t("recent_videos_transcript_button"), key=f"{self.prefix}_transcript_{video['video_id']}"):
                    transcript, lang = self.get_transcript(
                        video['video_id'], language)
                    if not transcript.startswith(t('recent_videos_transcript_error')):
                        self.save_transcript(
                            transcript, video['video_id'], work_directory)
                        st.session_state[f"{self.prefix}_transcript"] = transcript
                        st.session_state[f"{self.prefix}_title"] = video['title']
                        st.session_state[f"{self.prefix}_transcript_lang"] = lang
                        st.session_state[f"{self.prefix}_show_transcript"] = True
                        st.session_state[f"{self.prefix}_current_video_id"] = video['video_id']

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if page_number > 0 and st.button("Page Précédente", key=f"{self.prefix}_prev"):
                st.session_state[f"{self.prefix}_page_number"] = page_number - 1
                st.rerun()
        with col2:
            st.write(f"Page {page_number + 1} / {total_pages}")
        with col3:
            if page_number < total_pages - 1 and st.button("Page Suivante", key=f"{self.prefix}_next"):
                st.session_state[f"{self.prefix}_page_number"] = page_number + 1
                st.rerun()

        if st.session_state.get(f"{self.prefix}_show_transcript", False):
            st.header(t("recent_videos_transcript_header"))
            st.write(
                f"{t('recent_videos_transcript_language')} {st.session_state[f'{self.prefix}_transcript_lang']}")
            st.text_area(t("recent_videos_transcript_content"),
                         st.session_state[f"{self.prefix}_transcript"],
                         height=300,
                         key=f"{self.prefix}_transcript_area")

    def display_simple(self, config):
        st.header(t("recent_videos_header"))
        youtube_api = YoutubeAPI(config)
        work_directory = config['common']['work_directory']
        language = config['common']['language']

        if 'channel_id' not in config['common'] or not config['common']['channel_id']:
            st.info(t("recent_videos_configure_channel_id"))
            return

        if 'all_videos' not in st.session_state:
            st.session_state.all_videos = youtube_api.get_channel_videos(
                config['common']['channel_id'])

        video_options = {video['title']: video['video_id']
                         for video in st.session_state.all_videos[:10]}
        selected_video = st.selectbox("Sélectionner une vidéo",
                                      list(video_options.keys()),
                                      key=f"{self.prefix}_select")

        if st.button(t("recent_videos_save_transcript"), key=f"{self.prefix}_save"):
            video_id = video_options[selected_video]
            transcript, _ = self.get_transcript(video_id, language)
            if not transcript.startswith(t('recent_videos_transcript_error')):
                self.save_transcript(transcript, video_id, work_directory)
                st.success(t("recent_videos_save_success"))

    def choose_simple(self, config) -> Optional[Dict[str, str]]:
        youtube_api = YoutubeAPI(config)
        language = config['common']['language']

        if 'channel_id' not in config['common'] or not config['common']['channel_id']:
            st.info(t("recent_videos_configure_channel_id"))
            return None

        if 'all_videos' not in st.session_state:
            st.session_state.all_videos = youtube_api.get_channel_videos(
                config['common']['channel_id'])

        video_options = {video['title']: video for video in st.session_state.all_videos[:10]}
        selected_video = st.selectbox(
            t("recent_videos_header"),
            list(video_options.keys()),
            key=f"{self.prefix}_video_select"
        )

        selected_video_data = video_options[selected_video]
        transcript, _ = self.get_transcript(selected_video_data['video_id'], language)
        if transcript.startswith(t('recent_videos_transcript_error')):
            st.error(transcript)
            return None

        return {
            "title": selected_video_data['title'],
            "url": selected_video_data['url'],
            "keywords": selected_video_data['keywords'],
            "content": transcript
        }
