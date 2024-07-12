import os
import subprocess
import streamlit as st
from app import Plugin, list_video_files

class TrimsilencesPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        
    def get_config_fields(self):
        return {
            "silence_threshold": {
                "type": "number",
                "label": "Seuil de silence (dB)",
                "default": -35
            },
            "silence_duration": {
                "type": "number",
                "label": "Durée minimale du silence (secondes)",
                "default": 0.5
            }
        }

    def get_config_ui(self, config):
        updated_config = {}
        #updated_config['separator_trimsilences'] = st.header('Retirer les silences')
        updated_config["silence_threshold"] = st.slider(
            "Seuil de silence (dB)",
            min_value=-60,
            max_value=0,
            value=config.get("silence_threshold", -35)
        )
        updated_config["silence_duration"] = st.slider(
            "Durée minimale du silence (secondes)",
            min_value=0.1,
            max_value=2.0,
            value=config.get("silence_duration", 0.5),
            step=0.1
        )
        return updated_config

    def get_tabs(self):
        return [{"name": "Retrait des silences", "plugin": "trimsilences"}]

    def remove_silence(self, input_file, threshold, duration, videos_dir):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        remsi_path = os.path.join(current_dir, '..', 'remsi.py')
        output_sh_path = os.path.join(current_dir, '..', 'output.sh')
        input_filename = os.path.basename(input_file)
        output_filename = f"outfile_{input_filename}"
        output_file = os.path.join(videos_dir, output_filename)
        ffmpeg_command = f"ffmpeg -i '{input_file}' -hide_banner -af silencedetect=n={threshold}dB:d={duration} -f null - 2>&1 | python '{remsi_path}' > '{output_sh_path}'"
        try:
            subprocess.run(ffmpeg_command, shell=True, check=True, cwd=current_dir)
            
            if not os.path.exists(output_sh_path) or os.path.getsize(output_sh_path) == 0:
                raise subprocess.CalledProcessError(1, ffmpeg_command, "output.sh n'a pas été créé ou est vide")
            
            subprocess.run(f"chmod +x '{output_sh_path}'", shell=True, check=True)
            subprocess.run(output_sh_path, shell=True, check=True)
            os.remove(output_sh_path)
            
            return output_file
        except subprocess.CalledProcessError as e:
            return f"Erreur lors de l'exécution de la commande : {str(e)}"
        except Exception as e:
            return f"Une erreur inattendue s'est produite : {str(e)}"

    def run(self, config):
        st.header("Retrait des silences")

        all_videos = list_video_files(config['common']['work_directory'])

        video_files, outfile_videos, _ = all_videos
        st.session_state['list_video_files'] = all_videos
        
        st.subheader("Vidéos originales")
        for file, full_path, _ in video_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button("Retirer les silences", key=f"remove_silence_{file}"):
                    with st.spinner(f"Traitement de {file} en cours..."):
                        result = self.remove_silence(
                            full_path,
                            config['trimsilences']['silence_threshold'],
                            config['trimsilences']['silence_duration'],
                            config['common']['work_directory']
                        )
                    if result.startswith("Erreur") or result.startswith("Une erreur"):
                        st.error(result)
                    else:
                        st.success(f"Traitement terminé. Fichier de sortie : {result}")
                        st.experimental_rerun()
