from lib.global_vars import translations, t
from app import Plugin
from plugins.common import list_all_video_files
import streamlit as st
import os
from lib.video_utils import transcribe_video_whisper_cli
from widgets.yt_transcript import YoutubeTranscriptWidget
from widgets.prompt_manager import PromptsManagerWidget

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
        return [{"name": t("transcript_tab"), "plugin": "transcript"}]

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

            col1, col2 = st.columns(2)
            with col1:
                question = st.text_input(t("transcript_question_input"), key="local_question_input")
                if st.button(t("transcript_question_button"), key="local_ask_question"):
                    if question:
                        with st.spinner("Processing question..."):
                            answer = self.answer_question(st.session_state.transcript, question)
                            st.session_state[f"transcript_answer"] = answer
                    else:
                        st.warning("Please enter a question")

            with col2:
                prompt_options = list(st.session_state.prompts.keys())
                if prompt_options:
                    selected_prompt = st.selectbox(t("select_prompt"), options=prompt_options, key="local_prompt_select")
                    if st.button(t("apply_prompt"), key="local_apply_prompt"):
                        with st.spinner("Processing prompt..."):
                            llm_config = config.get('llm', {})
                            result = self.apply_prompt(st.session_state.transcript, st.session_state.prompts[selected_prompt], llm_config)
                            st.session_state[f"transcript_prompt_result"] = result

            if f"transcript_answer" in st.session_state:
                st.subheader(t("transcript_answer_title"))
                st.write(st.session_state[f"transcript_answer"])

            if f"transcript_prompt_result" in st.session_state:
                st.subheader(t("prompt_result"))
                st.write(st.session_state[f"transcript_prompt_result"])

    def run_remote(self, config):
        self.yt_transcript_widget.display()

        if f"transcript_transcript" in st.session_state:
            transcript = st.session_state[f"transcript_transcript"]

            col1, col2 = st.columns(2)

            with col1:
                question = st.text_input(t("transcript_question_input"), key="remote_question_input")
                if st.button(t("transcript_question_button"), key="remote_ask_question"):
                    if question:
                        with st.spinner("Processing question..."):
                            answer = self.answer_question(transcript, question)
                            st.session_state[f"transcript_answer"] = answer
                    else:
                        st.warning("Please enter a question")

            with col2:
                prompt_options = list(st.session_state.prompts.keys())
                if prompt_options:
                    selected_prompt = st.selectbox(t("select_prompt"), options=prompt_options, key="remote_prompt_select")
                    if st.button(t("apply_prompt"), key="remote_apply_prompt"):
                        with st.spinner("Processing prompt..."):
                            llm_config = config.get('llm', {})
                            result = self.apply_prompt(transcript, st.session_state.prompts[selected_prompt], llm_config)
                            st.session_state[f"transcript_prompt_result"] = result

            if f"transcript_answer" in st.session_state:
                st.subheader(t("transcript_answer_title"))
                st.write(st.session_state[f"transcript_answer"])

            if f"transcript_prompt_result" in st.session_state:
                st.subheader(t("prompt_result"))
                st.write(st.session_state[f"transcript_prompt_result"])

    def run(self, config):
        tab1, tab2, tab3 = st.tabs(["Local", "Remote", t("prompt_management")])
        with tab1:
            self.run_local(config)
        with tab2:
            self.run_remote(config)
        with tab3:
            PromptsManagerWidget("prompt_manager", "prompt_manager", self.plugin_manager).display(config)
