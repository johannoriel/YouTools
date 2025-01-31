from global_vars import translations, t
import os
import numpy as np
from typing import List, Tuple
import streamlit as st
from app import Plugin
from plugins.common import list_video_files
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Mise à jour des traductions existantes
translations["en"].update({
    "trim_silences_tab": "Silence Removal",
    "trim_silences_header": "Remove Silences from Videos",
    "trim_silences_threshold_label": "Silence threshold (dB)",
    "trim_silences_duration_label": "Minimum silence duration (seconds)",
    "trim_silences_original_videos": "Original Videos",
    "trim_silences_button": "Remove Silences",
    "trim_silences_processing": "Processing {file}...",
    "trim_silences_success": "Processing completed. Output file: {result}",
    "trim_silences_error": "Error during processing: {error}",
    "trim_silences_progress": "Processing: {progress}%"
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
    "trim_silences_error": "Erreur lors du traitement : {error}",
    "trim_silences_progress": "Progression : {progress}%"
})

def detect_silence_segments(audio_array: np.ndarray, sample_rate: int,
                          threshold_db: float, min_duration: float) -> List[Tuple[float, float]]:
    """
    Détecte les segments de silence dans un signal audio.

    Args:
        audio_array: Signal audio (numpy array)
        sample_rate: Taux d'échantillonnage
        threshold_db: Seuil de silence en dB
        min_duration: Durée minimale du silence en secondes

    Returns:
        Liste de tuples (début, fin) des segments non-silencieux en secondes
    """
    # Convertir le seuil dB en amplitude linéaire
    threshold_amp = 10 ** (threshold_db / 20)

    # Calculer l'amplitude RMS sur des fenêtres courtes
    window_size = int(sample_rate * 0.02)  # fenêtre de 20ms
    rms = np.array([np.sqrt(np.mean(window**2))
                   for window in np.array_split(audio_array, len(audio_array) // window_size)])

    # Détecter les segments silencieux
    is_silence = rms < threshold_amp

    # Convertir les indices en temps
    time_per_window = window_size / sample_rate
    changes = np.where(np.diff(is_silence))[0]

    # Construire les segments non-silencieux
    non_silence_segments = []
    start_time = 0

    for i in range(0, len(changes), 2):
        if i + 1 >= len(changes):
            break

        silence_duration = (changes[i] - changes[i-1]) * time_per_window if i > 0 else 0

        # Si le silence est assez long, créer un nouveau segment
        if silence_duration >= min_duration:
            end_time = changes[i-1] * time_per_window if i > 0 else 0
            if end_time > start_time:
                non_silence_segments.append((start_time, end_time))
            start_time = changes[i] * time_per_window

    # Ajouter le dernier segment si nécessaire
    end_time = len(audio_array) / sample_rate
    if end_time > start_time:
        non_silence_segments.append((start_time, end_time))

    return non_silence_segments

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

    def remove_silence(self, input_file: str, threshold: float, duration: float,
                      videos_dir: str, progress_callback=None) -> str:
        """
        Supprime les silences d'une vidéo en utilisant moviepy.

        Args:
            input_file: Chemin du fichier vidéo d'entrée
            threshold: Seuil de silence en dB
            duration: Durée minimale du silence en secondes
            videos_dir: Répertoire de sortie
            progress_callback: Fonction de callback pour la progression

        Returns:
            Chemin du fichier de sortie
        """
        try:
            if progress_callback:
                progress_callback(0)

            # Charger la vidéo
            video = VideoFileClip(input_file)

            # Extraire l'audio et le convertir en array numpy
            audio_array = video.audio.to_soundarray()
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)  # Convertir en mono si stéréo

            # Détecter les segments non-silencieux
            non_silence_segments = detect_silence_segments(
                audio_array,
                video.audio.fps,
                threshold,
                duration
            )

            if progress_callback:
                progress_callback(33)

            # Découper la vidéo selon les segments
            clips = []
            for i, (start, end) in enumerate(non_silence_segments):
                clip = video.subclip(start, end)
                clips.append(clip)
                if progress_callback:
                    progress = 33 + (i / len(non_silence_segments) * 33)
                    progress_callback(int(progress))

            # Concaténer les segments
            final_video = concatenate_videoclips(clips)

            if progress_callback:
                progress_callback(66)

            # Générer le nom du fichier de sortie
            output_filename = f"outfile_{os.path.basename(input_file)}"
            output_file = os.path.join(videos_dir, output_filename)

            # Écrire le fichier final
            final_video.write_videofile(output_file,
                                      codec='libx264',
                                      audio_codec='aac',
                                      temp_audiofile='temp-audio.m4a',
                                      remove_temp=True,
                                      audio_bitrate="192k",
                                      preset='medium')

            # Nettoyer
            video.close()
            final_video.close()
            for clip in clips:
                clip.close()

            if progress_callback:
                progress_callback(100)

            return output_file

        except Exception as e:
            return t("trim_silences_error").format(error=str(e))

    def run(self, config):
        st.header(t("trim_silences_header"))

        all_videos = list_video_files(config['common']['work_directory'])
        video_files, outfile_videos, _, _ = all_videos
        st.session_state['list_video_files'] = all_videos

        st.subheader(t("trim_silences_original_videos"))
        for file, full_path, _ in video_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button(t("trim_silences_button"), key=f"remove_silence_{file}"):
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def update_progress(progress):
                        progress_bar.progress(progress)
                        progress_text.text(t("trim_silences_progress").format(progress=progress))

                    with st.spinner(t("trim_silences_processing").format(file=file)):
                        result = self.remove_silence(
                            full_path,
                            config['trimsilences']['silence_threshold'],
                            config['trimsilences']['silence_duration'],
                            config['common']['work_directory'],
                            update_progress
                        )

                    progress_bar.empty()
                    progress_text.empty()

                    if result.startswith(t("trim_silences_error").format(error="")):
                        st.error(result)
                    else:
                        st.success(t("trim_silences_success").format(result=result))
                        st.rerun()
