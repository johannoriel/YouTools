# yt_transcript.py

from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from lib.youtube_api import YoutubeAPI
from lib.youtube_db import save_transcript, get_video_transcript
from lib.video_utils import transcribe_video_whisper_cli
import os

translations["en"].update({
    "transcript_widget_title": "YouTube Video Transcript",
    "transcript_url_input": "Enter YouTube Video URL",
    "transcript_fetch_button": "Get Transcript",
    "transcript_fetching": "Fetching transcript...",
    "transcript_error": "Error: {error}",
    "transcript_not_available": "Transcript not available",
})

translations["fr"].update({
    "transcript_widget_title": "Transcription de vidéo YouTube",
    "transcript_url_input": "Entrez l'URL de la vidéo YouTube",
    "transcript_fetch_button": "Obtenir la transcription",
    "transcript_fetching": "Récupération de la transcription...",
    "transcript_error": "Erreur : {error}",
    "transcript_not_available": "Transcription non disponible",
})

class YoutubeTranscriptWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.youtube_api = YoutubeAPI(self.plugin_manager.config)

    def get_video_id_from_url(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        if "youtube.com/watch?v=" in url:
            return url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        return ""

    def fetch_transcript(self, video_url: str) -> str:
        """Fetch transcript from DB, YouTube API, or Whisper transcription in that order."""
        video_id = self.get_video_id_from_url(video_url)
        if not video_id:
            st.error(t("transcript_error").format(error="Invalid YouTube URL"))
            return ""

        # Check database first
        transcript = get_video_transcript(video_id)
        if transcript:
            return transcript
        st.info("Not found in database...")

        # Try YouTube API transcript
        try:
            transcript, _ = self.youtube_api.get_transcript(video_id, self.plugin_manager.config["common"]["language"])
            if transcript and not transcript.startswith("get_transcript error"):
                save_transcript(video_id, transcript)
                return transcript
        except Exception as e:
            st.error(t("transcript_error").format(error=str(e)))

        # Fallback to Whisper transcription
        st.info("Not found on youtube transcript... trying to download")
        return ""
        video_path = None
        try:
            from yt_dlp import YoutubeDL
            work_directory = self.plugin_manager.config['common']['work_directory']
            ydl_opts_audio = {
                'format': 'bestaudio',
                'outtmpl': os.path.join(work_directory, f'{video_id}.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }],
            }
            with YoutubeDL(ydl_opts_audio) as ydl:
                ydl.download([video_url])
                video_path = os.path.join(work_directory, f'{video_id}.mp3')

            transcript = transcribe_video_whisper_cli(
                video_path=video_path,
                output_format="txt",
                whisper_path=os.path.expanduser(self.plugin_manager.config['transcript']["whisper_path"]),
                ffmpeg_path=os.path.expanduser(self.plugin_manager.config['transcript']["ffmpeg_path"]),
                whisper_model=self.plugin_manager.config['transcript']["whisper_model"],
                lang=self.plugin_manager.config["common"]["language"]
            )

            if transcript:
                save_transcript(video_id, transcript)
                return transcript
            return ""

        except Exception as e:
            st.error(t("transcript_error").format(error=str(e)))
            return ""
        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception as e:
                    st.error(t("transcript_error").format(error=str(e)))

    def display(self):
        st.title(t("transcript_widget_title"))

        # Input for YouTube URL
        video_url = st.text_input(t("transcript_url_input"), key=f"{self.prefix}_video_url")
        video_id = self.get_video_id_from_url(video_url)

        # Button to fetch transcript
        if st.button(t("transcript_fetch_button"), key=f"{self.prefix}_fetch_transcript"):
            if video_id:
                with st.spinner(t("transcript_fetching")):
                    transcript = self.fetch_transcript(video_url)
                    if transcript:
                        st.session_state[f"{self.prefix}_transcript"] = transcript
                    else:
                        st.warning(t("transcript_not_available"))
            else:
                st.error(t("transcript_error").format(error="Invalid YouTube URL"))

        # Display transcript if available
        if f"{self.prefix}_transcript" in st.session_state:
            transcript = st.session_state[f"{self.prefix}_transcript"]
            st.subheader("Transcript")
            st.text_area("Transcript", transcript, height=300, key=f"{self.prefix}_transcript_display")
