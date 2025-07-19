# transcript.py

from lib.global_vars import translations, t
from app import Plugin
from plugins.common import list_all_video_files
import streamlit as st
import os
from lib.video_utils import transcribe_video_whisper_cli
from widgets.yt_transcript import YoutubeTranscriptWidget
from widgets.prompt_manager import PromptsManagerWidget
from widgets.file_selector import FileSelectorWidget
from pytubefix import Playlist
import re

# Ajout des traductions
translations["en"].update({
    "transcript_header": "Video Transcription",
    "transcript_tab": "Local Transcription",
    "playlist_tab": "YouTube Playlist",
    "transcript_select_video": "Select a video",
    "transcript_no_videos": "No video files found in",
    "transcript_output_format": "Output format",
    "transcript_transcribe_button": "Transcribe",
    "transcript_transcribing": "Transcribing video...",
    "transcript_transcription_done": "Transcription completed!",
    "transcript_content": "Transcript",
    "transcript_copy_button": "Copy Transcript",
    "transcript_copy_success": "Transcript copied to clipboard!",
    "transcript_download_button": "Download Transcript",
    "transcript_question_input": "Ask a question about the transcript",
    "transcript_question_button": "Ask Question",
    "transcript_answer_title": "Answer",
    "prompt_result": "Prompt Result",
    "select_prompt": "Select a Prompt",
    "apply_prompt": "Apply Prompt",
    "playlist_url": "Enter YouTube Playlist URL",
    "playlist_process_button": "Process Playlist",
    "playlist_invalid_url": "Invalid playlist URL",
    "playlist_processing": "Processing video {current} of {total}",
    "playlist_results": "Playlist Results",
    "playlist_download_button": "Download Playlist Results",
    "combined_results_title": "Combined Results",
    "combined_results_prompt": "Ask a question about combined results",
    "prompt_management": "Prompt Management",
    "batch_tab": "Batch Processing",
    "batch_transcribe_button": "Transcribe Selected Videos",
    "batch_no_files_selected": "No files selected. Please select at least one video file.",
    "batch_transcribing": "Transcribing {current} of {total} videos...",
    "batch_transcription_done": "Transcription completed for {file}",
    "batch_transcription_failed": "Failed to transcribe {file}: {error}",
    "batch_results_title": "Batch Transcription Results",
    "batch_download_all_button": "Download All Transcripts",
    "batch_no_results": "No batch transcription results available.",
})

translations["fr"].update({
    "transcript_header": "Transcription de vidéo",
    "transcript_tab": "Transcription locale",
    "playlist_tab": "Playlist YouTube",
    "transcript_select_video": "Sélectionner une vidéo",
    "transcript_no_videos": "Aucun fichier vidéo trouvé dans",
    "transcript_output_format": "Format de sortie",
    "transcript_transcribe_button": "Transcrire",
    "transcript_transcribing": "Transcription de la vidéo en cours...",
    "transcript_transcription_done": "Transcription terminée !",
    "transcript_content": "Transcription",
    "transcript_copy_button": "Copier la transcription",
    "transcript_copy_success": "Transcription copiée dans le presse-papiers !",
    "transcript_download_button": "Télécharger la transcription",
    "transcript_question_input": "Poser une question sur la transcription",
    "transcript_question_button": "Poser la question",
    "transcript_answer_title": "Réponse",
    "prompt_result": "Résultat du prompt",
    "select_prompt": "Sélectionner un prompt",
    "apply_prompt": "Appliquer le prompt",
    "playlist_url": "Entrez l'URL de la playlist YouTube",
    "playlist_process_button": "Traiter la playlist",
    "playlist_invalid_url": "URL de playlist invalide",
    "playlist_processing": "Traitement de la vidéo {current} sur {total}",
    "playlist_results": "Résultats de la playlist",
    "playlist_download_button": "Télécharger les résultats de la playlist",
    "combined_results_title": "Résultats combinés",
    "combined_results_prompt": "Poser une question sur les résultats combinés",
    "prompt_management": "Gestion des prompts",
    "batch_tab": "Traitement par lot",
    "batch_transcribe_button": "Transcrire les vidéos sélectionnées",
    "batch_no_files_selected": "Aucun fichier sélectionné. Veuillez sélectionner au moins un fichier vidéo.",
    "batch_transcribing": "Transcription de {current} sur {total} vidéos...",
    "batch_transcription_done": "Transcription terminée pour {file}",
    "batch_transcription_failed": "Échec de la transcription de {file} : {error}",
    "batch_results_title": "Résultats de la transcription par lot",
    "batch_download_all_button": "Télécharger toutes les transcriptions",
    "batch_no_results": "Aucun résultat de transcription par lot disponible.",
})

class TranscriptPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.yt_transcript_widget = YoutubeTranscriptWidget("transcript", "transcript", plugin_manager)
        self.file_selector = FileSelectorWidget("transcript", "batch", plugin_manager)
        if 'prompts' not in st.session_state:
            st.session_state.prompts = {}
        # Déclarer les prompts auprès du plugin llm
        llm_plugin = plugin_manager.get_plugin("llm")
        if llm_plugin:
            llm_plugin.declare_json_prompt("transcript_prompts", self.name, "prompts")

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
                "options": [("tiny", "Tiny"), ("base", "Base"), ("small", "Duel"), ("medium", "Medium"), ("large", "Large")],
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
            {"name": t("playlist_tab"), "plugin": "transcript"},
            {"name": t("batch_tab"), "plugin": "transcript"}
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

        selected_video = st.selectbox(t("transcript_select_video"), options=[v[0] for v in videos], key="local_select_video")
        selected_video_path = next(v[1] for v in videos if v[0] == selected_video)

        output_format = st.radio(t("transcript_output_format"), ["txt", "srt"], key="local_output_format")

        if st.button(t("transcript_transcribe_button"), key="local_transcribe_button"):
            with st.spinner(t("transcript_transcribing")):
                transcript = self.transcribe_video(selected_video_path, output_format)
                st.session_state.transcript = transcript
                st.session_state.show_transcript = True
                with open(os.path.join(work_directory, "transcript.txt"), "w", encoding="utf-8") as f:
                    f.write(transcript)

        if st.session_state.get('show_transcript', False):
            st.success(t("transcript_transcription_done"))
            st.text_area(t("transcript_content"), st.session_state.transcript, height=300, key="local_transcript_area")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("transcript_copy_button"), key="local_copy_button"):
                    st.code(st.session_state.transcript)
                    st.success(t("transcript_copy_success"))
            with col2:
                st.download_button(
                    label=t("transcript_download_button"),
                    data=st.session_state.transcript,
                    file_name=f"transcript_{os.path.splitext(selected_video)[0]}.{output_format}",
                    mime="text/plain",
                    key="local_download_button"
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

    def run_batch(self, config):
        st.header(t("batch_tab"))

        # Initialiser le sélecteur de fichiers
        selected_files = self.file_selector.display(
            mode="video",
            allowed_extensions=['.mp4', '.mkv', '.mov', '.avi']
        )

        # Sélection du format de sortie
        output_format = st.radio(t("transcript_output_format"), ["txt", "srt"], key="batch_output_format")

        # Bouton pour lancer la transcription par lot
        if st.button(t("batch_transcribe_button"), key="batch_transcribe_button"):
            if not selected_files:
                st.warning(t("batch_no_files_selected"))
                return

            work_directory = os.path.expanduser(config['common']['work_directory'])
            total_files = len(selected_files)
            progress_bar = st.progress(0)
            results = []

            for i, file_path in enumerate(selected_files, 1):
                file_name = os.path.basename(file_path)
                with st.spinner(t("batch_transcribing").format(current=i, total=total_files)):
                    try:
                        transcript = self.transcribe_video(file_path, output_format)
                        output_file = os.path.join(work_directory, f"transcript_{os.path.splitext(file_name)[0]}.{output_format}")
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(transcript)
                        results.append({
                            "file": file_name,
                            "transcript": transcript,
                            "output_file": output_file
                        })
                        st.success(t("batch_transcription_done").format(file=file_name))
                    except Exception as e:
                        st.error(t("batch_transcription_failed").format(file=file_name, error=str(e)))

                progress_bar.progress(i / total_files)

            st.session_state.batch_results = results
            progress_bar.empty()

        # Afficher les résultats
        if "batch_results" in st.session_state and st.session_state.batch_results:
            st.subheader(t("batch_results_title"))
            data = [{"File": r["file"], "Output File": r["output_file"]} for r in st.session_state.batch_results]
            st.table(data)

            # Télécharger toutes les transcriptions
            all_transcripts = "\n\n".join([f"File: {r['file']}\nTranscript:\n{r['transcript']}" for r in st.session_state.batch_results])
            st.download_button(
                label=t("batch_download_all_button"),
                data=all_transcripts,
                file_name="batch_transcripts.txt",
                mime="text/plain",
                key="batch_download_all_button"
            )

            # Afficher les transcriptions individuelles
            for result in st.session_state.batch_results:
                with st.expander(f"Transcript for {result['file']}"):
                    st.text_area(t("transcript_content"), result["transcript"], height=200, key=f"batch_transcript_area_{result['file']}")
                    st.download_button(
                        label=t("transcript_download_button"),
                        data=result["transcript"],
                        file_name=f"transcript_{result['file']}.{output_format}",
                        mime="text/plain",
                        key=f"batch_download_button_{result['file']}"
                    )
        else:
            st.info(t("batch_no_results"))

    def run(self, config):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Local", "Remote", t("prompt_management"), t("playlist_tab"), t("batch_tab")])
        with tab1:
            self.run_local(config)
        with tab2:
            self.run_remote(config)
        with tab3:
            PromptsManagerWidget("transcript", "prompt_manager", self.plugin_manager).display("prompts")
        with tab4:
            self.run_playlist(config)
        with tab5:
            self.run_batch(config)
