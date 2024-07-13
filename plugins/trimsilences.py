from global_vars import translations, t
import os
import subprocess
import streamlit as st
from app import Plugin
from plugins.common import list_video_files

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "trim_silences_tab": "Silence Removal",
    "trim_silences_header": "Remove Silences from Videos",
    "trim_silences_threshold_label": "Silence threshold (dB)",
    "trim_silences_duration_label": "Minimum silence duration (seconds)",
    "trim_silences_original_videos": "Original Videos",
    "trim_silences_button": "Remove Silences",
    "trim_silences_processing": "Processing {file}...",
    "trim_silences_success": "Processing completed. Output file: {result}",
    "trim_silences_error": "Error during command execution: {error}",
    "trim_silences_unexpected_error": "An unexpected error occurred: {error}",
    "trim_silences_output_sh_error": "output.sh was not created or is empty"
})

translations["fr"].update({
    "trim_silences_tab": "Retrait des silences",
    "trim_silences_header": "Retirer les silences des vidéos",
    "trim_silences_threshold_label": "Seuil de silence (dB)",
    "trim_silences_duration_label": "Durée minimale du silence (secondes)",
    "trim_silences_original_videos": "Vidéos originales",
    "trim_silences_button": "Retirer les silences",
    "trim_silences_processing": "Traitement de {file} en cours...",
    "trim_silences_success": "Traitement terminé. Fichier de sortie : {result}",
    "trim_silences_error": "Erreur lors de l'exécution de la commande : {error}",
    "trim_silences_unexpected_error": "Une erreur inattendue s'est produite : {error}",
    "trim_silences_output_sh_error": "output.sh n'a pas été créé ou est vide"
})

class TrimsilencesPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        
    def get_config_fields(self):
        return {
            "silence_threshold": {
                "type": "number",
                "label": t("trim_silences_threshold_label"),
                "default": -35
            },
            "silence_duration": {
                "type": "number",
                "label": t("trim_silences_duration_label"),
                "default": 0.5
            }
        }

    def get_config_ui(self, config):
        updated_config = {}
        updated_config["silence_threshold"] = st.slider(
            t("trim_silences_threshold_label"),
            min_value=-60,
            max_value=0,
            value=config.get("silence_threshold", -35)
        )
        updated_config["silence_duration"] = st.slider(
            t("trim_silences_duration_label"),
            min_value=0.1,
            max_value=2.0,
            value=config.get("silence_duration", 0.5),
            step=0.1
        )
        return updated_config

    def get_tabs(self):
        return [{"name": t("trim_silences_tab"), "plugin": "trimsilences"}]

    def remove_silence(self, input_file, threshold, duration, videos_dir):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        remsi_path = os.path.join(current_dir, '..', 'remsi.py')
        output_sh_path = os.path.join(current_dir, '..', 'output.sh')
        input_filename = os.path.basename(input_file)
        output_filename = f"outfile_{input_filename}"
        output_file = os.path.join(videos_dir, output_filename)
        ffmpeg_command = f"ffmpeg -i '{input_file}' -hide_banner -y -af silencedetect=n={threshold}dB:d={duration} -f null - 2>&1 | python '{remsi_path}' > '{output_sh_path}'"
        print(ffmpeg_command)
        try:
            subprocess.run(ffmpeg_command, shell=True, check=True, cwd=current_dir)
            
            if not os.path.exists(output_sh_path) or os.path.getsize(output_sh_path) == 0:
                raise subprocess.CalledProcessError(1, ffmpeg_command, t("trim_silences_output_sh_error"))
            
            subprocess.run(f"chmod +x '{output_sh_path}'", shell=True, check=True)
            subprocess.run(output_sh_path, shell=True, check=True)
            os.remove(output_sh_path)
            
            return output_file
        except subprocess.CalledProcessError as e:
            return t("trim_silences_error").format(error=str(e))
        except Exception as e:
            return t("trim_silences_unexpected_error").format(error=str(e))

    def run(self, config):
        st.header(t("trim_silences_header"))

        all_videos = list_video_files(config['common']['work_directory'])

        video_files, outfile_videos, _ = all_videos
        st.session_state['list_video_files'] = all_videos
        
        st.subheader(t("trim_silences_original_videos"))
        for file, full_path, _ in video_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button(t("trim_silences_button"), key=f"remove_silence_{file}"):
                    with st.spinner(t("trim_silences_processing").format(file=file)):
                        result = self.remove_silence(
                            full_path,
                            config['trimsilences']['silence_threshold'],
                            config['trimsilences']['silence_duration'],
                            config['common']['work_directory']
                        )
                    if result.startswith("Erreur") or result.startswith("Une erreur"):
                        st.error(result)
                    else:
                        st.success(t("trim_silences_success").format(result=result))
                        st.experimental_rerun()

