from lib.global_vars import translations, t
import os
import numpy as np
from typing import List, Tuple
import streamlit as st
from app import Plugin
from plugins.common import list_video_files
from moviepy import VideoFileClip, concatenate_videoclips

# Ajout des nouvelles traductions
translations["en"].update({
    "trim_silences_tab": "Silence Removal",
    "trim_silences_header": "Remove Silences from Videos",
    "trim_silences_threshold_label": "Silence threshold (dB)",
    "trim_silences_duration_label": "Minimum silence duration to detect (seconds)",
    "trim_silences_keep_duration_label": "Duration to keep for each silence (seconds)",
    "trim_silences_original_videos": "Original Videos",
    "trim_silences_button": "Remove Silences",
    "trim_silences_processing": "Processing {file}...",
    "trim_silences_success": "Processing completed. Output file: {result}",
    "trim_silences_error": "Error during processing: {error}",
    "trim_silences_progress": "Processing: {progress}%",
    "trim_silences_params": "Silence Detection Parameters",
    "trim_silences_apply": "Apply Parameters",
})

translations["fr"].update({
    "trim_silences_tab": "Retrait des silences",
    "trim_silences_header": "Retirer les silences des vidéos",
    "trim_silences_threshold_label": "Seuil de silence (dB)",
    "trim_silences_duration_label": "Durée minimale de silence à détecter (secondes)",
    "trim_silences_keep_duration_label": "Durée à conserver pour chaque silence (secondes)",
    "trim_silences_original_videos": "Vidéos originales",
    "trim_silences_button": "Retirer les silences",
    "trim_silences_processing": "Traitement de {file} en cours...",
    "trim_silences_success": "Traitement terminé. Fichier de sortie : {result}",
    "trim_silences_error": "Erreur lors du traitement : {error}",
    "trim_silences_progress": "Progression : {progress}%",
    "trim_silences_params": "Paramètres de détection des silences",
    "trim_silences_apply": "Appliquer les paramètres",
})


def detect_silence_segments(audio_array: np.ndarray, sample_rate: int,
                            threshold_db: float, min_duration: float,
                            keep_duration: float) -> List[Tuple[float, float, float]]:
    print(
        f"Longueur audio array: {len(audio_array)} échantillons, Sample rate: {sample_rate} Hz")

    # Convertir le seuil dB en amplitude linéaire
    threshold_amp = 10 ** (threshold_db / 20)

    # Calculer l'amplitude RMS sur des fenêtres courtes
    window_size = int(sample_rate * 0.02)  # fenêtre de 20ms
    print(f"Taille fenêtre: {window_size} échantillons")
    if window_size == 0:
        print("Erreur: window_size est 0")
        return []
    num_windows = len(audio_array) // window_size
    print(f"Nombre de fenêtres: {num_windows}")
    if num_windows == 0:
        print("Erreur: num_windows est 0")
        return []
    audio_array = audio_array[:num_windows * window_size]
    rms = np.array([np.sqrt(np.mean(window**2))
                   for window in np.array_split(audio_array, num_windows)])
    print(f"Longueur RMS: {len(rms)}")

    # Détecter les segments silencieux
    is_silence = rms < threshold_amp
    if len(is_silence) == 0:
        print("Erreur: is_silence est vide")
        return []

    # Trouver les indices de changement d'état
    changes = np.where(np.diff(is_silence))[0] + 1
    changes = changes.tolist()
    print(f"Indices de changement: {changes}")

    # Ajouter les bords si nécessaire pour gérer les silences initiaux/finaux
    if is_silence[0]:
        changes.insert(0, 0)
    if is_silence[-1]:
        changes.append(len(is_silence))
    print(f"Indices ajustés: {changes}")

    segments = []
    start_time = 0.0
    time_per_window = window_size / sample_rate
    print(f"Temps par fenêtre: {time_per_window} secondes")

    # Traiter chaque intervalle de silence
    for i in range(0, len(changes), 2):
        if i + 1 >= len(changes):
            break

        silence_start = changes[i]
        silence_end = changes[i + 1]

        silence_duration = (silence_end - silence_start) * time_per_window
        # print(f"Silence détecté: début={silence_start}, fin={silence_end}, durée={silence_duration}s")

        if silence_duration >= min_duration:
            non_silent_end = silence_start * time_per_window
            if non_silent_end > start_time:
                segments.append((start_time, non_silent_end, None))
                # print(f"Segment non-silencieux: {start_time} -> {non_silent_end}")

            silence_middle = (silence_start + silence_end) / \
                2 * time_per_window
            keep_start = max(start_time, silence_middle - keep_duration / 2)
            keep_end = min(silence_end * time_per_window,
                           silence_middle + keep_duration / 2)
            segments.append((keep_start, keep_end, silence_middle))
            # print(f"Segment conservé: {keep_start} -> {keep_end}, milieu={silence_middle}")

            start_time = silence_end * time_per_window

    final_end_time = len(audio_array) / sample_rate
    if start_time < final_end_time:
        segments.append((start_time, final_end_time, None))
        # print(f"Dernier segment: {start_time} -> {final_end_time}")

    # Fusionner les segments adjacents sans pause
    merged_segments = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            last_seg = merged_segments[-1]
            if last_seg[2] is None and seg[2] is None:
                merged_segments[-1] = (last_seg[0], seg[1], None)
            else:
                merged_segments.append(seg)

    # print("Segments finaux:")
    # for i, (start, end, middle) in enumerate(merged_segments):
    #    print(f"Segment {i}: {start} -> {end} (durée={(end-start)}s)")
    return merged_segments


class TrimsilencesPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        if 'temp_silence_params' not in st.session_state:
            st.session_state.temp_silence_params = None

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
            },
            "keep_duration": {
                "type": "number",
                "label": t("trim_silences_keep_duration_label"),
                "default": 0.1
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
        updated_config["keep_duration"] = st.slider(
            t("trim_silences_keep_duration_label"),
            min_value=0.0,
            max_value=0.5,
            value=config.get("keep_duration", 0.1),
            step=0.05
        )
        return updated_config

    def get_tabs(self):
        return [{"name": t("trim_silences_tab"), "plugin": "trimsilences"}]

    def remove_silence(self, input_file: str, threshold: float, duration: float,
                       keep_duration: float, videos_dir: str,
                       progress_callback=None) -> tuple[str, str, float, float]:
        """
        Supprime les silences d'une vidéo en conservant une durée minimale.

        Args:
            input_file: Chemin du fichier vidéo d'entrée
            threshold: Seuil de silence en dB
            duration: Durée minimale du silence en secondes
            keep_duration: Durée à conserver pour chaque silence
            videos_dir: Répertoire de sortie
            progress_callback: Fonction de callback pour la progression

        Returns:
            Tuple contenant (output_file, reduction_str, original_duration, final_duration)
        """
        try:
            if progress_callback:
                progress_callback(0)

            # Charger la vidéo
            video = VideoFileClip(input_file)
            original_duration = video.duration

            # Extraire l'audio et le convertir en array numpy
            audio_array = video.audio.to_soundarray(fps=video.audio.fps)
            print(f"Type audio_array: {type(audio_array)}")
            print(f"Shape audio_array: {audio_array.shape}")
            print(f"Dtype audio_array: {audio_array.dtype}")
            print(
                f"Valeurs min/max audio_array: {audio_array.min()}, {audio_array.max()}")
            print(f"Sample rate: {video.audio.fps}")

            # Convertir en mono si stéréo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1).astype(np.float32)
                print(f"Après conversion mono - Shape: {audio_array.shape}")

            # Détecter les segments avec leurs points de transition
            segments = detect_silence_segments(
                audio_array,
                video.audio.fps,
                threshold,
                duration,
                keep_duration
            )
            print(f"Nombre de segments détectés: {len(segments)}")

            if progress_callback:
                progress_callback(33)

            print("Découpage des segments silencieux")
            # Découper la vidéo selon les segments
            clips = []
            for i, (start, end, silence_middle) in enumerate(segments):
                # Ajouter le segment non-silencieux
                clip = video.subclipped(start_time=start, end_time=end)
                clips.append(clip)

                if progress_callback:
                    progress = 33 + (i / len(segments) * 33)
                    progress_callback(int(progress))

            # Concaténer les segments
            print("Concaténation des segments")
            final_video = concatenate_videoclips(clips)

            if progress_callback:
                progress_callback(66)

            # Générer le nom du fichier de sortie
            output_filename = f"outfile_{os.path.basename(input_file)}"
            output_file = os.path.join(videos_dir, output_filename)

            # Écrire le fichier final
            print("Écriture du fichier final")
            final_video.write_videofile(output_file,
                                        codec='libx264',
                                        audio_codec='aac',
                                        temp_audiofile='temp-audio.m4a',
                                        remove_temp=True,
                                        audio_bitrate="192k",
                                        preset='medium')
            final_duration = final_video.duration

            # Calcul du pourcentage de réduction
            if final_duration >= original_duration:
                error_msg = t("trim_silences_error").format(
                    error="Output video is not shorter than input video"
                )
                # Nettoyer
                video.close()
                final_video.close()
                for clip in clips:
                    clip.close()
                if os.path.exists(output_file):
                    os.remove(output_file)  # Supprimer le fichier invalide
                return error_msg, "0%", original_duration, 0.0

            reduction_percentage = ((original_duration - final_duration) /
                                    original_duration * 100)
            reduction_str = f"{reduction_percentage:.1f}%"

            # Nettoyer
            video.close()
            final_video.close()
            for clip in clips:
                clip.close()

            if progress_callback:
                progress_callback(100)

            return output_file, reduction_str, original_duration, final_duration

        except Exception as e:
            raise e
            return t("trim_silences_error").format(error=str(e)), "0%", 0.0, 0.0

    def run(self, config):
        st.header(t("trim_silences_header"))

        # Section pour les paramètres temporaires
        st.subheader(t("trim_silences_params"))

        # Initialiser les paramètres temporaires si nécessaire
        if st.session_state.temp_silence_params is None:
            st.session_state.temp_silence_params = {
                "silence_threshold": config['trimsilences']['silence_threshold'],
                "silence_duration": config['trimsilences']['silence_duration'],
                "keep_duration": config['trimsilences']['keep_duration']
            }

        # Interface pour modifier les paramètres
        col1, col2, col3 = st.columns(3)
        with col1:
            temp_threshold = st.slider(
                t("trim_silences_threshold_label"),
                min_value=-60,
                max_value=0,
                value=st.session_state.temp_silence_params["silence_threshold"]
            )
        with col2:
            temp_duration = st.slider(
                t("trim_silences_duration_label"),
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.temp_silence_params["silence_duration"],
                step=0.1
            )
        with col3:
            temp_keep_duration = st.slider(
                t("trim_silences_keep_duration_label"),
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.temp_silence_params["keep_duration"],
                step=0.05
            )

        # Mettre à jour les paramètres temporaires
        st.session_state.temp_silence_params = {
            "silence_threshold": temp_threshold,
            "silence_duration": temp_duration,
            "keep_duration": temp_keep_duration
        }

        # Liste des vidéos
        all_videos = list_video_files(config['common']['work_directory'])
        video_files, outfile_videos, _, _ = all_videos
        st.session_state['list_video_files'] = all_videos

        st.subheader(t("trim_silences_original_videos"))
        for file, full_path, _ in video_files:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button(t("trim_silences_button"), key=f"remove_silence_{file}"):
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def update_progress(progress):
                        progress_bar.progress(progress)
                        progress_text.text(
                            t("trim_silences_progress").format(progress=progress))

                    with st.spinner(t("trim_silences_processing").format(file=file)):
                        # Utiliser les paramètres temporaires au lieu des paramètres de configuration
                        result, reduction, _, _ = self.remove_silence(
                            full_path,
                            st.session_state.temp_silence_params["silence_threshold"],
                            st.session_state.temp_silence_params["silence_duration"],
                            st.session_state.temp_silence_params["keep_duration"],
                            config['common']['work_directory'],
                            update_progress
                        )

                    progress_bar.empty()
                    progress_text.empty()

                    if result.startswith(t("trim_silences_error").format(error="")):
                        st.error(result)
                    else:
                        st.success(
                            t("trim_silences_success").format(result=result) +
                            f" - Reduction: {reduction}"
                        )
                        st.rerun()
