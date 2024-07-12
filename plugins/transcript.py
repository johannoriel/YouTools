from app import Plugin
import streamlit as st
import os
import subprocess
import json
import tempfile
import getpass

class TranscriptPlugin(Plugin):
    def get_config_fields(self):
        return {
            "whisper_path": {
                "type": "text",
                "label": "Chemin vers l'exécutable whisper.cpp",
                "default": "~/Evaluation/whisper.cpp/main"
            },
            "whisper_model": {
                "type": "select",
                "label": "Modèle Whisper à utiliser",
                "options": [("tiny", "Tiny"), ("base", "Base"), ("small", "Small"), ("medium", "Medium"), ("large", "Large")],
                "default": "medium"
            },
            "ffmpeg_path": {
                "type": "text",
                "label": "Chemin vers l'exécutable ffmpeg",
                "default": "ffmpeg"
            }
        }

    def get_tabs(self):
        return [{"name": "Transcription locale", "plugin": "transcript"}]

    def list_video_files(self, directory):
        video_files = []
        for file in os.listdir(directory):
            if file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                full_path = os.path.join(directory, file)
                mod_time = os.path.getmtime(full_path)
                video_files.append((file, full_path, mod_time))
        video_files.sort(key=lambda x: x[2], reverse=True)
        return video_files

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
                print("Sortie STDERR:", result.stderr)
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
                print("Sortie STDERR:", result.stderr)
            except subprocess.CalledProcessError as e:
                print("Erreur lors de l'exécution de ffmpeg:")
                print(e.stderr)  # Affiche le message d'erreur de ffmpeg
            print('done')
            
            with open(output_file, 'r') as f:
                transcript = f.read()
                
            os.remove(output_file)
            #os.remove(temp_audio_path)

            return transcript

        except subprocess.CalledProcessError as e:
            st.error(f"Erreur lors de la transcription : {e.stderr}")
            return None

        finally:
            # Nettoyage des fichiers temporaires
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(f"transcript.{output_format}"):
                os.remove(f"transcript.{output_format}")

    def run(self, config):
        st.header("Transcription locale de vidéos")
        
        work_directory = os.path.expanduser(config['common']['work_directory'])
        whisper_path = os.path.expanduser(config['transcript']['whisper_path'])
        whisper_model = config['transcript']['whisper_model']
        ffmpeg_path = config['transcript']['ffmpeg_path']
        
        videos = self.list_video_files(work_directory)
        
        if not videos:
            st.info(f"Aucune vidéo trouvée dans le répertoire {work_directory}")
            return
        
        selected_video = st.selectbox("Sélectionnez une vidéo à transcrire", options=[v[0] for v in videos])
        selected_video_path = next(v[1] for v in videos if v[0] == selected_video)
        
        output_format = st.radio("Format de sortie", ["txt", "srt"])
        
        if st.button("Transcrire"):
            with st.spinner("Transcription en cours... (ça peut prendre plusieurs min)"):
                transcript = self.transcribe_video(selected_video_path, output_format, whisper_path, whisper_model, ffmpeg_path, config['common']['language'])
                st.session_state.transcript = transcript
                st.session_state.show_transcript = True
                st.session_state.show_llm_response = False
            
        if st.session_state.get('show_transcript', False):
            st.success("Transcription terminée !")
            st.text_area("Contenu de la transcription", st.session_state.transcript, height=300)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Copier la transcription"):
                    st.code(st.session_state.transcript)
                    st.success("Transcription copiée ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour la copier depuis le bloc de code ci-dessus.")
            with col2:
                st.download_button(
                    label="Télécharger la transcription",
                    data=st.session_state.transcript,
                    file_name=f"transcript_{os.path.splitext(selected_video)[0]}.{output_format}",
                    mime="text/plain"
                )
            
            # Résumé avec LLM
            llm_plugin = self.plugin_manager.get_plugin('llm')
            llm_config = config.get('llm', {})
            prompt = llm_config.get('llm_prompt', '')
            
            with st.expander("Résumé avec LLM"):
                with st.form("llm_form"):
                    st.markdown("Voulez-vous résumer la transcription avec le LLM ?")
                    custom_prompt = st.text_input("Prompt personnalisé (optionnel)", prompt)
                    submit_button = st.form_submit_button("Résumer avec LLM")
                
                if submit_button:
                    video_content = f"# {selected_video} \n {st.session_state.transcript}"
                    llm_response = llm_plugin.process_with_llm(
                        custom_prompt or prompt, 
                        llm_config.get('llm_sys_prompt', ''), 
                        video_content, 
                        llm_config.get('llm_model', '')
                    )
                    st.text_area("Résumé LLM", llm_response, height=200)
                    st.session_state.llm_response = llm_response
                    st.session_state.show_llm_response = True
                    
                if st.session_state.get('show_llm_response', False):
                    if st.button("Copier le résumé LLM"):
                        st.code(st.session_state.llm_response)
                        st.success("Résumé LLM copié ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour le copier depuis le bloc de code ci-dessus.")
                    
                    st.download_button(
                        label="Télécharger le résumé LLM",
                        data=st.session_state.llm_response,
                        file_name=f"llm_summary_{os.path.splitext(selected_video)[0]}.txt",
                        mime="text/plain"
                    )
                        
