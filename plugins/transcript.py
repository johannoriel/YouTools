from lib.global_vars import translations, t
from app import Plugin
from plugins.common import list_all_video_files
import streamlit as st
import os
from lib.video_utils import transcribe_video_whisper_cli
from widgets.yt_transcript import YoutubeTranscriptWidget
from widgets.prompt_manager import PromptsManagerWidget
from pytubefix import Playlist
import re

translations["en"].update({
    "transcript_tab": "Transcription tools",
    "transcript_header": "Local Video Transcription",
    "transcript_no_videos": "No videos found in the directory",
    "transcript_select_video": "Select a video to transcribe",
    "transcript_output_format": "Output format",
    "transcript_transcribe_button": "Transcribe",
    "transcript_transcribing": "Transcribing... (this may take several minutes)",
    "transcript_transcription_done": "Transcription completed!",
    "transcript_content": "Transcription Content",
    "transcript_copy_button": "Copy Transcription",
    "transcript_copy_success": "Transcription copied! Use Ctrl+C (or Cmd+C on Mac) to copy it from the code block above.",
    "transcript_download_button": "Download Transcription",
    "transcript_summary_with_llm": "Summarize with LLM",
    "transcript_custom_prompt": "Custom prompt (optional)",
    "transcript_summary_button": "Summarize with LLM",
    "transcript_llm_summary": "LLM Summary",
    "transcript_llm_copy_button": "Copy LLM Summary",
    "transcript_llm_copy_success": "LLM Summary copied! Use Ctrl+C (or Cmd+C on Mac) to copy it from the code block above.",
    "transcript_llm_download_button": "Download LLM Summary",
    "transcript_error_transcribing": "Error during transcription: ",
    "prompt_management": "Prompt Management",
    "select_prompt": "Select a prompt",
    "custom_prompt": "Custom prompt (optional)",
    "apply_prompt": "Apply Prompt",
    "new_prompt_name": "New prompt name",
    "new_prompt_content": "New prompt content",
    "add_prompt": "Add Prompt",
    "save_prompt": "Save Prompt",
    "edit_prompt": "Edit Prompt",
    "delete_prompt": "Delete Prompt",
    "prompt_result": "Prompt Result",
    "copy_result": "Copy Result",
    "download_result": "Download Result",
    "result_copied": "Result copied! Use Ctrl+C (or Cmd+C on Mac) to copy it from the code block above.",
    "promt_result_display": "Result",
    "transcript_summary_title": "Transcript Summary",
    "transcript_question_button": "Ask Question",
    "transcript_question_input": "Enter your question about the transcript",
    "transcript_answer_title": "Answer to Your Question",
    "transcript_free_prompt_input": "Enter a custom prompt for the transcript",
    "transcript_free_prompt_button": "Apply Custom Prompt",
    "playlist_tab": "Playlist Processing",
    "playlist_url": "Enter YouTube Playlist URL",
    "playlist_process_button": "Process Playlist",
    "playlist_processing": "Processing playlist ({current}/{total})...",
    "playlist_results": "Playlist Processing Results",
    "playlist_video_title": "Video Title",
    "playlist_response": "Response",
    "playlist_no_results": "No results to display",
    "playlist_invalid_url": "Invalid YouTube Playlist URL. Please provide a valid playlist URL (e.g., https://www.youtube.com/playlist?list=... or a video URL with a playlist parameter).",
    "playlist_download_button": "Download Playlist Results",
    "combined_results_prompt": "Enter a question or prompt about the combined results",
    "combined_results_title": "Combined Results Response",
})

translations["fr"].update({
    "transcript_tab": "Outils de transcription",
    "transcript_header": "Transcription locale de vidéos",
    "transcript_no_videos": "Aucune vidéo trouvée dans le répertoire",
    "transcript_select_video": "Sélectionnez une vidéo à transcrire",
    "transcript_output_format": "Format de sortie",
    "transcript_transcribe_button": "Transcrire",
    "transcript_transcribing": "Transcription en cours... (ça peut prendre plusieurs minutes)",
    "transcript_transcription_done": "Transcription terminée!",
    "transcript_content": "Contenu de la transcription",
    "transcript_copy_button": "Copier la transcription",
    "transcript_copy_success": "Transcription copiée ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour la copier depuis le bloc de code ci-dessus.",
    "transcript_download_button": "Télécharger la transcription",
    "transcript_summary_with_llm": "Résumé avec LLM",
    "transcript_custom_prompt": "Prompt personnalisé (optionnel)",
    "transcript_summary_button": "Résumer avec LLM",
    "transcript_llm_summary": "Résumé LLM",
    "transcript_llm_copy_button": "Copier le résumé LLM",
    "transcript_llm_copy_success": "Résumé LLM copié ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour le copier depuis le bloc de code ci-dessus.",
    "transcript_llm_download_button": "Télécharger le résumé LLM",
    "transcript_error_transcribing": "Erreur lors de la transcription : ",
    "prompt_management": "Gestion des prompts",
    "select_prompt": "Sélectionner un prompt",
    "custom_prompt": "Prompt personnalisé (optionnel)",
    "apply_prompt": "Appliquer le Prompt",
    "new_prompt_name": "Nom du nouveau prompt",
    "new_prompt_content": "Contenu du nouveau prompt",
    "add_prompt": "Ajouter un Prompt",
    "save_prompt": "Sauver le Prompt",
    "edit_prompt": "Modifier le Prompt",
    "delete_prompt": "Supprimer le Prompt",
    "prompt_result": "Résultat du Prompt",
    "copy_result": "Copier le Résultat",
    "download_result": "Télécharger le Résultat",
    "result_copied": "Résultat copié ! Utilisez Ctrl+C (ou Cmd+C on Mac) pour le copier depuis le bloc de code ci-dessus.",
    "promt_result_display": "Resultat",
    "transcript_summary_title": "Résumé de la transcription",
    "transcript_question_button": "Poser une question",
    "transcript_question_input": "Entrez votre question sur la transcription",
    "transcript_answer_title": "Réponse à votre question",
    "transcript_free_prompt_input": "Entrez un prompt personnalisé pour la transcription",
    "transcript_free_prompt_button": "Appliquer le Prompt Personnalisé",
    "playlist_tab": "Traitement de la playlist",
    "playlist_url": "Entrez l'URL de la playlist YouTube",
    "playlist_process_button": "Traiter la playlist",
    "playlist_processing": "Traitement de la playlist ({current}/{total})...",
    "playlist_results": "Résultats du traitement de la playlist",
    "playlist_video_title": "Titre de la vidéo",
    "playlist_response": "Réponse",
    "playlist_no_results": "Aucun résultat à afficher",
    "playlist_invalid_url": "URL de playlist YouTube invalide. Veuillez fournir une URL de playlist valide (par ex., https://www.youtube.com/playlist?list=... ou une URL de vidéo avec un paramètre de playlist).",
    "playlist_download_button": "Télécharger les résultats de la playlist",
    "combined_results_prompt": "Entrez une question ou un prompt sur l'ensemble des résultats",
    "combined_results_title": "Réponse sur l'ensemble des résultats",
})

class TranscriptPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.yt_transcript_widget = YoutubeTranscriptWidget("transcript", "transcript", plugin_manager)
        if 'prompts' not in st.session_state:
            st.session_state.prompts = {}

    def get_config_fields(self):
        fields = {
            "whisper_path": {
                "type": "text",
                "label": "transcript_whisper_path",
                "default": "~/Evaluation/whisper.cpp/main"
            },
            "whisper_model": {
                "type": "select",
                "label": "transcript_whisper_model",
                "options": [("tiny", "Tiny"), ("base", "Base"), ("small", "Small"), ("medium", "Medium"), ("large", "Large")],
                "default": "medium"
            },
            "ffmpeg_path": {
                "type": "text",
                "label": "transcript_ffmpeg_path",
                "default": "ffmpeg"
            },
            "prompts": {
                "type": "json",
                "label": "Saved Prompts",
                "default": "{}"
            }
        }
        return fields

    def get_tabs(self):
        return [
            {"name": t("transcript_tab"), "plugin": "transcript"},
            {"name": t("playlist_tab"), "plugin": "transcript"}
        ]

    def transcribe_video(self, video_path, output_format, whisper_path=None, whisper_model=None, ffmpeg_path=None, lang=None):
        if whisper_path is None:
            whisper_path = os.path.expanduser(self.get_config("whisper_path"))
        if ffmpeg_path is None:
            ffmpeg_path = os.path.expanduser(self.get_config("ffmpeg_path"))
        if whisper_model is None:
            whisper_model = self.get_config("whisper_model")
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

    def apply_prompt(self, transcript, prompt, llm_config):
        response = self.process_with_llm(prompt, llm_config.get('llm_sys_prompt', ''), transcript)
        return response

    def summarize_transcript(self, transcript: str) -> str:
        prompt = """
        Summarize the following transcript in 3-5 sentences, capturing the main points:
        {transcript}
        """
        return self.process_with_llm(prompt.format(transcript=transcript), "", transcript)

    def answer_question(self, transcript: str, question: str) -> str:
        prompt = """
        Based on the following transcript, answer the question: {question}
        Transcript: {transcript}
        """
        return self.process_with_llm(prompt.format(question=question, transcript=transcript), "", transcript)

    def process_transcript(self, transcript, question=None, selected_prompt=None, llm_config=None):
        if question:
            return self.answer_question(transcript, question)
        elif selected_prompt:
            return self.apply_prompt(transcript, st.session_state.prompts[selected_prompt], llm_config)
        return None

    def process_combined_results(self, results, question=None, selected_prompt=None, llm_config=None):
        combined_text = "\n\n".join([f"Video: {r['title']}\nResponse: {r['response']}" for r in results])
        if question:
            return self.answer_question(combined_text, question)
        elif selected_prompt:
            return self.apply_prompt(combined_text, st.session_state.prompts[selected_prompt], llm_config)
        return None

    def display_transcript_results(self, transcript_key, answer_key, prompt_result_key, config, mode="remote"):
        if transcript_key in st.session_state:
            transcript = st.session_state[transcript_key]

            col1, col2 = st.columns(2)

            with col1:
                question = st.text_input(t("transcript_question_input"), key=f"{mode}_question_input")
                if st.button(t("transcript_question_button"), key=f"{mode}_ask_question"):
                    if question:
                        with st.spinner("Processing question..."):
                            answer = self.answer_question(transcript, question)
                            st.session_state[answer_key] = answer
                    else:
                        st.warning("Please enter a question")

            with col2:
                prompt_options = list(st.session_state.prompts.keys())
                if prompt_options:
                    selected_prompt = st.selectbox(t("select_prompt"), options=prompt_options, key=f"{mode}_prompt_select")
                    if st.button(t("apply_prompt"), key=f"{mode}_apply_prompt"):
                        with st.spinner("Processing prompt..."):
                            llm_config = config.get('llm', {})
                            result = self.apply_prompt(transcript, st.session_state.prompts[selected_prompt], llm_config)
                            st.session_state[prompt_result_key] = result

            if answer_key in st.session_state:
                st.subheader(t("transcript_answer_title"))
                st.write(st.session_state[answer_key])

            if prompt_result_key in st.session_state:
                st.subheader(t("prompt_result"))
                st.write(st.session_state[prompt_result_key])

    def extract_playlist_id(self, url):
        """Extract playlist ID from YouTube URL."""
        pattern = r"(?:list=)([0-9A-Za-z_-]+)|(?:youtube\.com/playlist\?list=)([0-9A-Za-z_-]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1) or match.group(2)
        return None

    def run_local(self, config):
        st.header(t("transcript_header"))

        work_directory = os.path.expanduser(config['common']['work_directory'])
        videos = list_all_video_files(work_directory)

        if not videos:
            st.info(f"{t('transcript_no_videos')} {work_directory}")
            return

        selected_video = st.selectbox(t("transcript_select_video"), options=[v[0] for v in videos])
        selected_video_path = next(v[1] for v in videos if v[0] == selected_video)

        output_format = st.radio(t("transcript_output_format"), ["txt", "srt"])

        if st.button(t("transcript_transcribe_button")):
            with st.spinner(t("transcript_transcribing")):
                transcript = self.transcribe_video(selected_video_path, output_format)
                st.session_state.transcript = transcript
                st.session_state.show_transcript = True
                with open(os.path.join(work_directory, "transcript.txt"), "w", encoding="utf-8") as f:
                    f.write(transcript)

        if st.session_state.get('show_transcript', False):
            st.success(t("transcript_transcription_done"))
            st.text_area(t("transcript_content"), st.session_state.transcript, height=300)

            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("transcript_copy_button")):
                    st.code(st.session_state.transcript)
                    st.success(t("transcript_copy_success"))
            with col2:
                st.download_button(
                    label=t("transcript_download_button"),
                    data=st.session_state.transcript,
                    file_name=f"transcript_{os.path.splitext(selected_video)[0]}.{output_format}",
                    mime="text/plain"
                )

            self.display_transcript_results("transcript", "transcript_answer", "transcript_prompt_result", config, mode="local")

    def run_remote(self, config):
        self.yt_transcript_widget.display()
        self.display_transcript_results("transcript_transcript", "transcript_answer", "transcript_prompt_result", config, mode="remote")

    def run_playlist(self, config):
        st.header(t("playlist_tab"))
        playlist_url = st.text_input(t("playlist_url"))
        prompt_options = list(st.session_state.prompts.keys())
        question = st.text_input(t("transcript_question_input"), key="playlist_question_input")
        selected_prompt = None
        if prompt_options:
            selected_prompt = st.selectbox(t("select_prompt"), options=prompt_options, key="playlist_prompt_select")

        if st.button(t("playlist_process_button")) and playlist_url:
            playlist_id = self.extract_playlist_id(playlist_url)
            if not playlist_id:
                st.error(t("playlist_invalid_url"))
                return

            normalized_playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
            with st.spinner(t("playlist_processing").format(current=0, total=0)):
                try:
                    playlist = Playlist(normalized_playlist_url)
                    videos = list(playlist.videos)  # Convert to list to get total count
                    total_videos = len(videos)
                    progress_bar = st.progress(0)
                    results = []
                    llm_config = config.get('llm', {})

                    for i, video in enumerate(videos, 1):
                        progress_bar.progress(i / total_videos, text=t("playlist_processing").format(current=i, total=total_videos))
                        try:
                            # Step 1: Get transcript
                            transcript = self.yt_transcript_widget.fetch_transcript(video.watch_url)
                            if transcript:
                                # Step 2: Apply prompt or question
                                response = self.process_transcript(transcript, question, selected_prompt, llm_config)
                                if response:
                                    results.append({"title": video.title, "url": video.watch_url, "response": response})
                        except Exception as e:
                            st.error(f"Error processing video {video.title}: {str(e)}")

                    st.session_state.playlist_results = results
                    progress_bar.empty()
                except Exception as e:
                    st.error(f"{t('playlist_invalid_url')} ({normalized_playlist_url}): {str(e)}")

        if "playlist_results" in st.session_state and st.session_state.playlist_results:
            st.subheader(t("playlist_results"))
            data = [{"Video Title": r["title"], "URL": r["url"], "Response": r["response"]} for r in st.session_state.playlist_results]
            st.table(data)

            # Export results to a file
            results_text = "\n\n".join([f"Video: {r['title']}\nURL: {r['url']}\nResponse: {r['response']}" for r in st.session_state.playlist_results])
            st.download_button(
                label=t("playlist_download_button"),
                data=results_text,
                file_name="playlist_request.txt",
                mime="text/plain"
            )

            # Process combined results
            st.subheader(t("combined_results_title"))
            new_question = st.text_input(t("combined_results_prompt"), key="playlist_new_question_input")
            new_prompt = None
            if prompt_options:
                new_prompt = st.selectbox(t("select_prompt"), options=prompt_options, key="playlist_new_prompt_select")

            if st.button(t("apply_prompt"), key="playlist_new_apply_prompt"):
                with st.spinner(t("playlist_processing").format(current=0, total=0)):
                    llm_config = config.get('llm', {})
                    response = self.process_combined_results(st.session_state.playlist_results, new_question, new_prompt, llm_config)
                    if response:
                        st.session_state.combined_results_response = response

            if "combined_results_response" in st.session_state:
                st.write(st.session_state.combined_results_response)
        else:
            st.info(t("playlist_no_results"))

    def run(self, config):
        tab1, tab2, tab3, tab4 = st.tabs(["Local", "Remote", t("prompt_management"), t("playlist_tab")])
        with tab1:
            self.run_local(config)
        with tab2:
            self.run_remote(config)
        with tab3:
            PromptsManagerWidget("prompt_manager", "prompt_manager", self.plugin_manager).display(config)
        with tab4:
            self.run_playlist(config)
