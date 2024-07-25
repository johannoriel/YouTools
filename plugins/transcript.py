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
    "transcript_error_transcribing": "Error during transcription: ",
    "prompt_management": "Prompt Management",
    "select_prompt": "Select a prompt",
    "custom_prompt": "Custom prompt (optional)",
    "apply_prompt": "Apply Prompt",
    "new_prompt_name": "New prompt name",
    "new_prompt_content": "New prompt content",
    "add_prompt": "Add Prompt",
    "edit_prompt": "Edit Prompt",
    "delete_prompt": "Delete Prompt",
    "prompt_result": "Prompt Result",
    "copy_result": "Copy Result",
    "download_result": "Download Result",
    "result_copied": "Result copied! Use Ctrl+C (or Cmd+C on Mac) to copy it from the code block above.",
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
    "transcript_error_transcribing": "Erreur lors de la transcription : ",
    "prompt_management": "Gestion des prompts",
    "select_prompt": "Sélectionner un prompt",
    "custom_prompt": "Prompt personnalisé (optionnel)",
    "apply_prompt": "Appliquer le Prompt",
    "new_prompt_name": "Nom du nouveau prompt",
    "new_prompt_content": "Contenu du nouveau prompt",
    "add_prompt": "Ajouter un Prompt",
    "edit_prompt": "Modifier le Prompt",
    "delete_prompt": "Supprimer le Prompt",
    "prompt_result": "Résultat du Prompt",
    "copy_result": "Copier le Résultat",
    "download_result": "Télécharger le Résultat",
    "result_copied": "Résultat copié ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour le copier depuis le bloc de code ci-dessus.",
})

class TranscriptPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        if 'prompts' not in st.session_state:
            st.session_state.prompts = {}

    def get_config_fields(self):
        fields = {
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
            expanded_whisper_path = os.path.expanduser(whisper_path)
            whisper_command = [
                expanded_whisper_path,
                "-m", f"{os.path.dirname(expanded_whisper_path)}/models/ggml-{whisper_model}.bin",
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
            print('Transcription done')

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

    def manage_prompts(self, config):
        st.subheader(t("prompt_management"))

        # Charger les prompts depuis la configuration
        if 'prompts' not in config['transcript']:
            config['transcript']['prompts'] = {}
        st.session_state.prompts = config['transcript']['prompts']

        # Afficher les prompts existants
        prompt_options = list(st.session_state.prompts.keys()) + ['Custom']
        selected_prompt = st.selectbox(t("select_prompt"), options=prompt_options, key="prompt_select")

        if selected_prompt == 'Custom':
            prompt_content = st.text_area(t("custom_prompt"), "", key="custom_prompt")
        else:
            prompt_content = st.text_area(t("edit_prompt"), st.session_state.prompts.get(selected_prompt, ""), key="edit_prompt")

        # Ajouter ou modifier un prompt
        new_prompt_name = st.text_input(t("new_prompt_name"), key="new_prompt_name")
        if st.button(t("add_prompt"), key="add_prompt"):
            if new_prompt_name:
                st.session_state.prompts[new_prompt_name] = prompt_content
                config['transcript']['prompts'] = st.session_state.prompts
                self.plugin_manager.save_config(config)
                st.success(f"Prompt '{new_prompt_name}' added/updated.")
                st.rerun()

        # Supprimer un prompt
        if selected_prompt != 'Custom' and st.button(t("delete_prompt"), key="delete_prompt"):
            del st.session_state.prompts[selected_prompt]
            config['transcript']['prompts'] = st.session_state.prompts
            self.plugin_manager.save_config(config)
            st.success(f"Prompt '{selected_prompt}' deleted.")
            st.rerun()

        return selected_prompt, prompt_content

    def apply_prompt(self, transcript, prompt, llm_plugin, llm_config):
        response = llm_plugin.process_with_llm(
            prompt,
            llm_config.get('llm_sys_prompt', ''),
            transcript,
            llm_config.get('llm_model', '')
        )
        return response

    def run(self, config):
        st.header(t("transcript_header"))

        # Gestion des prompts (indépendante du transcript)
        with st.expander(t("prompt_management"), expanded=True):
            selected_prompt, prompt_content = self.manage_prompts(config)

        # Le reste du code pour la transcription
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

            # Application du prompt
            if st.button(t("apply_prompt")):
                llm_plugin = self.plugin_manager.get_plugin('llm')
                llm_config = config.get('llm', {})

                final_prompt = prompt_content
                if selected_prompt != 'Custom':
                    final_prompt = st.session_state.prompts[selected_prompt] + "\n" + prompt_content

                result = self.apply_prompt(st.session_state.transcript, final_prompt, llm_plugin, llm_config)
                st.session_state.prompt_result = result

            # Affichage du résultat
            if 'prompt_result' in st.session_state:
                st.subheader(t("prompt_result"))
                st.text_area("", st.session_state.prompt_result, height=300)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(t("copy_result")):
                        st.code(st.session_state.prompt_result)
                        st.success(t("result_copied"))
                with col2:
                    st.download_button(
                        label=t("download_result"),
                        data=st.session_state.prompt_result,
                        file_name="prompt_result.txt",
                        mime="text/plain"
                    )
