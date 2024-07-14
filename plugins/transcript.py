from global_vars import translations, t
from app import Plugin
from plugins.common import list_all_video_files
import streamlit as st
import os
import subprocess
import json
import tempfile
import getpass

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "transcript_tab": "Local Transcription",
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
    "transcript_error_transcribing": "Error during transcription: "
})

translations["fr"].update({
    "transcript_tab": "Transcription locale",
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
    "transcript_error_transcribing": "Erreur lors de la transcription : "
})

class TranscriptPlugin(Plugin):
    def get_config_fields(self):
        return {
            "whisper_path": {
                "type": "text",
                "label": t("transcript_whisper_path"),
                "default": "~/Evaluation/whisper.cpp/main"
            },
            "whisper_model": {
                "type": "select",
                "label": t("transcript_whisper_model"),
                "options": [("tiny", "Tiny"), ("base", "Base"), ("small", "Small"), ("medium", "Medium"), ("large", "Large")],
                "default": "medium"
            },
            "ffmpeg_path": {
                "type": "text",
                "label": t("transcript_ffmpeg_path"),
                "default": "ffmpeg"
            }
        }

    def get_tabs(self):
        return [{"name": t("transcript_tab"), "plugin": "transcript"}]

    def transcribe_video(self, video_path, output_format, whisper_path, whisper_model, ffmpeg_path, lang):
        print("Exécuté par l'utilisateur :", getpass.getuser())
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        try:
            # Conversion de la vidéo en audio WAV 16kHz
            print(f"Conversion to {temp_audio_path} 16bits...")
            ffmpeg_command = [
                ffmpeg_path, '-y',
                '-i', video_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                temp_audio_path
            ]
            print("Commande ffmpeg:", " ".join(ffmpeg_command))
            try:
                result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                print("Sortie STDOUT:", result.stdout)
                #print("Sortie STDERR:", result.stderr)
            except subprocess.CalledProcessError as e:
                print("Erreur lors de l'exécution de ffmpeg:")
                print(e.stderr)  # Affiche le message d'erreur de ffmpeg


            # Transcription avec whisper.cpp
            print(f"Transcription with whisper {whisper_model}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as temp_output:
                output_file = temp_output.name
            file_without_extension, file_extension = os.path.splitext(output_file)
            whisper_command = [
                whisper_path,
                "-m", f"{os.path.dirname(whisper_path)}/models/ggml-{whisper_model}.bin",
                "-f", temp_audio_path,
                "-l", lang,
                "-of", file_without_extension,
                "-otxt" if output_format == "txt" else "-osrt"
            ]
            print("Commande whisper:", " ".join(whisper_command))
            try:
                result = subprocess.run(whisper_command, check=True, capture_output=True, text=True)
                print("Sortie STDOUT:", result.stdout)
                #print("Sortie STDERR:", result.stderr)
            except subprocess.CalledProcessError as e:
                print("Erreur lors de l'exécution de whisper:")
                print(e.stderr)  # Affiche le message d'erreur de whisper
            print('done')

            with open(output_file, 'r') as f:
                transcript = f.read()

            os.remove(output_file)
            #os.remove(temp_audio_path)

            return transcript

        except subprocess.CalledProcessError as e:
            st.error(f"{t('transcript_error_transcribing')}{e.stderr}")
            return None

        finally:
            # Nettoyage des fichiers temporaires
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(f"transcript.{output_format}"):
                os.remove(f"transcript.{output_format}")

    def run(self, config):
        st.header(t("transcript_header"))

        work_directory = os.path.expanduser(config['common']['work_directory'])
        whisper_path = os.path.expanduser(config['transcript']['whisper_path'])
        whisper_model = config['transcript']['whisper_model']
        ffmpeg_path = config['transcript']['ffmpeg_path']

        videos = list_all_video_files(work_directory)

        if not videos:
            st.info(f"{t('transcript_no_videos')} {work_directory}")
            return

        selected_video = st.selectbox(t("transcript_select_video"), options=[v[0] for v in videos])
        selected_video_path = next(v[1] for v in videos if v[0] == selected_video)

        output_format = st.radio(t("transcript_output_format"), ["txt", "srt"])

        if st.button(t("transcript_transcribe_button")):
            with st.spinner(t("transcript_transcribing")):
                transcript = self.transcribe_video(selected_video_path, output_format, whisper_path, whisper_model, ffmpeg_path, config['common']['language'])
                st.session_state.transcript = transcript
                st.session_state.show_transcript = True
                st.session_state.show_llm_response = False

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

            # Résumé avec LLM
            llm_plugin = self.plugin_manager.get_plugin('llm')
            llm_config = config.get('llm', {})
            prompt = llm_config.get('llm_prompt', '')

            with st.expander(t("transcript_summary_with_llm")):
                with st.form("llm_form"):
                    st.markdown(t("transcript_summary_with_llm"))
                    custom_prompt = st.text_input(t("transcript_custom_prompt"), prompt)
                    submit_button = st.form_submit_button(t("transcript_summary_button"))

                if submit_button:
                    video_content = f"# {selected_video} \n {st.session_state.transcript}"
                    llm_response = llm_plugin.process_with_llm(
                        custom_prompt or prompt,
                        llm_config.get('llm_sys_prompt', ''),
                        video_content,
                        llm_config.get('llm_model', '')
                    )
                    st.text_area(t("transcript_llm_summary"), llm_response, height=200)
                    st.session_state.llm_response = llm_response
                    st.session_state.show_llm_response = True

                if st.session_state.get('show_llm_response', False):
                    if st.button(t("transcript_llm_copy_button")):
                        st.code(st.session_state.llm_response)
                        st.success(t("transcript_llm_copy_success"))

                    st.download_button(
                        label=t("transcript_llm_download_button"),
                        data=st.session_state.llm_response,
                        file_name=f"llm_summary_{os.path.splitext(selected_video)[0]}.txt",
                        mime="text/plain"
                    )
