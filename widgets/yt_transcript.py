from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import os
from lib.youtube_api import YoutubeAPI
from lib.youtube_db import save_transcript, get_video_transcript
from lib.video_utils import transcribe_video_whisper_cli

translations["en"].update({
    "transcript_widget_title": "YouTube Video Transcript",
    "transcript_url_input": "Enter YouTube Video URL",
    "transcript_fetch_button": "Get Transcript",
    "transcript_fetching": "Fetching transcript...",
    "transcript_summary_button": "Summarize Transcript",
    "transcript_question_button": "Ask Question",
    "transcript_question_input": "Enter your question about the transcript",
    "transcript_error": "Error: {error}",
    "transcript_not_available": "Transcript not available",
    "transcript_summary_title": "Transcript Summary",
    "transcript_answer_title": "Answer to Your Question"
})

translations["fr"].update({
    "transcript_widget_title": "Transcription de vidéo YouTube",
    "transcript_url_input": "Entrez l'URL de la vidéo YouTube",
    "transcript_fetch_button": "Obtenir la transcription",
    "transcript_fetching": "Récupération de la transcription...",
    "transcript_summary_button": "Résumer la transcription",
    "transcript_question_button": "Poser une question",
    "transcript_question_input": "Entrez votre question sur la transcription",
    "transcript_error": "Erreur : {error}",
    "transcript_not_available": "Transcription non disponible",
    "transcript_summary_title": "Résumé de la transcription",
    "transcript_answer_title": "Réponse à votre question"
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

    def transcribe_video(self, video_path, output_format, whisper_path=None, whisper_model=None, ffmpeg_path=None, lang=None):
        """Wrapper autour de transcribe_video_whisper_cli qui gère les paramètres par défaut de la classe."""
        if whisper_path is None:
            whisper_path = os.path.expanduser(
                self.plugin_manager.config['transcript']["whisper_path"])
        if ffmpeg_path is None:
            ffmpeg_path = os.path.expanduser(
                self.plugin_manager.config['transcript']["ffmpeg_path"])
        if whisper_model is None:
            whisper_model = self.plugin_manager.config['transcript']["whisper_model"]
        if lang is None:
            lang = self.plugin_manager.config["common"]["language"]

        return transcribe_video_whisper_cli(
            video_path=video_path,
            output_format=output_format,
            whisper_path=whisper_path,
            whisper_model=whisper_model,
            ffmpeg_path=ffmpeg_path,
            lang=lang
        )

    def generate_transcript(self, video_id: str, video_url: str) -> str:
        """Generate transcript by downloading video and processing it with transcript plugin."""
        video_path = None
        try:
            from yt_dlp import YoutubeDL
            ydl_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': [self.plugin_manager.config['common']['language']],
                'subtitlesformat': 'txt',
                'outtmpl': os.path.join(self.plugin_manager.config['common']['work_directory'], '%(id)s.%(ext)s'),
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                if 'subtitles' in info and self.plugin_manager.config['common']['language'] in info['subtitles']:
                    subtitle_file = os.path.join(
                        self.plugin_manager.config['common']['work_directory'],
                        f"{video_id}.{self.plugin_manager.config['common']['language']}.txt"
                    )
                    ydl.download([video_url])
                    if os.path.exists(subtitle_file):
                        with open(subtitle_file, 'r', encoding='utf-8') as f:
                            transcript = f.read()
                        if transcript:
                            save_transcript(video_id, transcript)
                            os.remove(subtitle_file)
                            return transcript

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
                video_path = os.path.join(
                    work_directory, f'{video_id}.mp3')

            # Transcribe video

            transcript = self.transcribe_video(video_path, "txt")

            if transcript:
                save_transcript(video_id, transcript)
                return transcript
            return ""

        except Exception as e:
            st.error(t("transcript_error").format(error=str(e)))
            raise e
            return ""
        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception as e:
                    st.error(t("transcript_error").format(error=str(e)))

    def summarize_transcript(self, transcript: str) -> str:
        """Generate summary of transcript using LLM."""
        prompt = """
        Summarize the following transcript in 3-5 sentences, capturing the main points:
        {transcript}
        """
        return self.process_with_llm(prompt.format(transcript=transcript), "", transcript)

    def answer_question(self, transcript: str, question: str) -> str:
        """Answer a question about the transcript using LLM."""
        prompt = """
        Based on the following transcript, answer the question: {question}
        Transcript: {transcript}
        """
        return self.process_with_llm(
            prompt.format(question=question, transcript=transcript),
            "",
            transcript
        )

    def display(self):
        st.title(t("transcript_widget_title"))

        # Input for YouTube URL
        video_url = st.text_input(
            t("transcript_url_input"), key=f"{self.prefix}_video_url")
        video_id = self.get_video_id_from_url(video_url)

        # Button to fetch transcript
        if st.button(t("transcript_fetch_button"), key=f"{self.prefix}_fetch_transcript"):
            if video_id:
                with st.spinner(t("transcript_fetching")):
                    transcript = get_video_transcript(video_id)
                    if not transcript:
                        transcript = self.generate_transcript(
                            video_id, video_url)
                    if transcript:
                        st.session_state[f"{self.prefix}_transcript"] = transcript
                    else:
                        st.warning(t("transcript_not_available"))
            else:
                st.error(t("transcript_error").format(
                    error="Invalid YouTube URL"))

        # Display transcript if available
        if f"{self.prefix}_transcript" in st.session_state:
            transcript = st.session_state[f"{self.prefix}_transcript"]
            st.subheader("Transcript")
            st.text_area("Transcript", transcript, height=300,
                         key=f"{self.prefix}_transcript_display")

            col1, col2 = st.columns(2)

            # Summary button
            with col1:
                if st.button(t("transcript_summary_button"), key=f"{self.prefix}_summary"):
                    with st.spinner("Generating summary..."):
                        summary = self.summarize_transcript(transcript)
                        st.session_state[f"{self.prefix}_summary_result"] = summary

            # Question button and input
            with col2:
                question = st.text_input(
                    t("transcript_question_input"),
                    key=f"{self.prefix}_question_input"
                )
                if st.button(t("transcript_question_button"), key=f"{self.prefix}_ask_question"):
                    if question:
                        with st.spinner("Processing question..."):
                            answer = self.answer_question(transcript, question)
                            st.session_state[f"{self.prefix}_answer"] = answer
                    else:
                        st.warning("Please enter a question")

            # Display summary if available
            if f"{self.prefix}_summary_result" in st.session_state:
                st.subheader(t("transcript_summary_title"))
                st.write(st.session_state[f"{self.prefix}_summary_result"])

            # Display answer if available
            if f"{self.prefix}_answer" in st.session_state:
                st.subheader(t("transcript_answer_title"))
                st.write(st.session_state[f"{self.prefix}_answer"])
