from lib.global_vars import translations, t
import os
import numpy as np
from typing import List, Tuple
import streamlit as st
from app import Plugin
from plugins.common import list_video_files2  # Updated import
from moviepy import VideoFileClip, concatenate_videoclips
from lib.video_utils import normalize_full_audio
import matplotlib.pyplot as plt

# Ajout des nouvelles traductions
translations["en"].update({
    "trim_silences_tab": "Silence Removal",
    "trim_silences_header": "Remove Silences from Videos",
    "trim_silences_threshold_label": "Silence threshold (dB)",
    "trim_silences_duration_label": "Minimum silence duration to detect (seconds)",
    "trim_silences_keep_duration_label": "Duration to keep for each silence (seconds)",
    "trim_silences_original_videos": "Original Videos",
    "trim_silences_button": "Remove Silences",
    "trim_silences_simple_button": "Remove Silences (Simple)",
    "trim_silences_processing": "Processing {file}...",
    "trim_silences_success": "Processing completed. Output file: {result}",
    "trim_silences_error": "Error during processing: {error}",
    "trim_silences_progress": "Processing: {progress}%",
    "trim_silences_params": "Silence Detection Parameters",
    "trim_silences_apply": "Apply Parameters",
    "analyze_button": "Analyze Audio",
    "analyze_max_level": "Maximum audio level: {max_level} dB",
    "analyze_min_level": "Minimum audio level: {min_level} dB",
    "analyze_volume_plot": "Volume Level per Second",
    "analyze_silence_threshold_label": "Silence detection threshold (dB)",
    "analyze_granularity_label": "Analysis granularity",
    "analyze_granularity_seconds": "Per second",
    "analyze_granularity_frames": "Per frame",
    "analyze_silence_removed": "Silence removed: {seconds} seconds ({percentage}%)",
    "select_videos_label": "Select videos to process",
    "process_selected_videos": "Process Selected Videos",
})

translations["fr"].update({
    "trim_silences_tab": "Retrait des silences",
    "trim_silences_header": "Retirer les silences des vidéos",
    "trim_silences_threshold_label": "Seuil de silence (dB)",
    "trim_silences_duration_label": "Durée minimale de silence à détecter (secondes)",
    "trim_silences_keep_duration_label": "Durée à conserver pour chaque silence (secondes)",
    "trim_silences_original_videos": "Vidéos originales",
    "trim_silences_button": "Retirer les silences",
    "trim_silences_simple_button": "Retirer les silences (Simple)",
    "trim_silences_processing": "Traitement de {file} en cours...",
    "trim_silences_success": "Traitement terminé. Fichier de sortie : {result}",
    "trim_silences_error": "Erreur lors du traitement : {error}",
    "trim_silences_progress": "Progression : {progress}%",
    "trim_silences_params": "Paramètres de détection des silences",
    "trim_silences_apply": "Appliquer les paramètres",
    "analyze_button": "Analyser l'audio",
    "analyze_max_level": "Niveau audio maximal : {max_level} dB",
    "analyze_min_level": "Niveau audio minimal : {min_level} dB",
    "analyze_volume_plot": "Niveau de volume par seconde",
    "analyze_silence_threshold_label": "Seuil de détection des silences (dB)",
    "analyze_granularity_label": "Granularité de l'analyse",
    "analyze_granularity_seconds": "Par seconde",
    "analyze_granularity_frames": "Par image",
    "analyze_silence_removed": "Silence supprimé : {seconds} secondes ({percentage}%)",
    "select_videos_label": "Sélectionner les vidéos à traiter",
    "process_selected_videos": "Traiter les vidéos sélectionnées",
})

def detect_silence_segments(audio_array: np.ndarray, sample_rate: int,
                            threshold_db: float, min_duration: float,
                            keep_duration: float, padding: float = 0.2) -> List[Tuple[float, float, float]]:
    print(f"Longueur audio array: {len(audio_array)} échantillons, Sample rate: {sample_rate} Hz")
    threshold_amp = 10 ** (threshold_db / 20)
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
    is_silence = rms < threshold_amp
    if len(is_silence) == 0:
        print("Erreur: is_silence est vide")
        return []
    changes = np.where(np.diff(is_silence))[0] + 1
    changes = changes.tolist()
    print(f"Indices de changement: {changes}")
    if is_silence[0]:
        changes.insert(0, 0)
    if is_silence[-1]:
        changes.append(len(is_silence))
    print(f"Indices ajustés: {changes}")
    segments = []
    start_time = 0.0
    time_per_window = window_size / sample_rate
    print(f"Temps par fenêtre: {time_per_window} secondes")
    for i in range(0, len(changes), 2):
        if i + 1 >= len(changes):
            break
        silence_start = changes[i]
        silence_end = changes[i + 1]
        silence_duration = (silence_end - silence_start) * time_per_window
        if silence_duration >= min_duration:
            non_silent_end = silence_start * time_per_window
            if non_silent_end > start_time:
                segments.append((start_time, non_silent_end, None))
            silence_middle = (silence_start + silence_end) / 2 * time_per_window
            keep_start = max(start_time, silence_middle - keep_duration / 2)
            keep_end = min(silence_end * time_per_window, silence_middle + keep_duration / 2)
            segments.append((keep_start, keep_end, silence_middle))
            start_time = silence_end * time_per_window
    final_end_time = len(audio_array) / sample_rate
    if start_time < final_end_time:
        segments.append((start_time, final_end_time, None))
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
    return merged_segments

def detect_silence_segments_simple(audio_array: np.ndarray, sample_rate: int,
                                  threshold_db: float) -> List[Tuple[float, float]]:
    threshold_amp = 10 ** (threshold_db / 20)
    window_size = int(sample_rate / 30)  # Granularité au niveau des frames (env. 33ms)
    if window_size == 0:
        return []
    num_windows = len(audio_array) // window_size
    if num_windows == 0:
        return []
    audio_array = audio_array[:num_windows * window_size]
    rms = np.array([np.sqrt(np.mean(window**2))
                   for window in np.array_split(audio_array, num_windows)])
    is_non_silent = rms >= threshold_amp
    time_per_window = window_size / sample_rate
    segments = []
    start_idx = None
    for i in range(len(is_non_silent)):
        if is_non_silent[i] and start_idx is None:
            start_idx = i
        elif not is_non_silent[i] and start_idx is not None:
            segments.append((start_idx * time_per_window, i * time_per_window))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx * time_per_window, len(is_non_silent) * time_per_window))
    return segments

def analyze_audio(audio_array: np.ndarray, sample_rate: int, granularity: str = "seconds") -> tuple[float, float, np.ndarray, np.ndarray]:
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1).astype(np.float32)
    if granularity == "seconds":
        window_size = sample_rate  # 1 seconde
    else:  # frames
        window_size = int(sample_rate / 30)
    num_windows = len(audio_array) // window_size
    if num_windows == 0:
        return 0.0, 0.0, np.array([]), np.array([])
    audio_array = audio_array[:num_windows * window_size]
    volume_levels = np.array([np.sqrt(np.mean(window**2))
                             for window in np.array_split(audio_array, num_windows)])
    volume_levels_db = 20 * np.log10(volume_levels + 1e-10)
    times = np.arange(num_windows) * (window_size / sample_rate)
    max_level_db = np.max(volume_levels_db)
    min_level_db = np.min(volume_levels_db[volume_levels_db > -float('inf')])
    return max_level_db, min_level_db, times, volume_levels_db

def calculate_silence_duration(audio_array: np.ndarray, sample_rate: int, threshold_db: float, min_duration: float) -> tuple[float, float]:
    threshold_amp = 10 ** (threshold_db / 20)
    window_size = int(sample_rate * 0.02)  # fenêtre de 20ms
    if window_size == 0:
        return 0.0, 0.0
    num_windows = len(audio_array) // window_size
    if num_windows == 0:
        return 0.0, 0.0
    audio_array = audio_array[:num_windows * window_size]
    rms = np.array([np.sqrt(np.mean(window**2))
                   for window in np.array_split(audio_array, num_windows)])
    is_silence = rms < threshold_amp
    time_per_window = window_size / sample_rate
    changes = np.where(np.diff(is_silence))[0] + 1
    changes = changes.tolist()
    if is_silence[0]:
        changes.insert(0, 0)
    if is_silence[-1]:
        changes.append(len(is_silence))
    total_silence_duration = 0.0
    for i in range(0, len(changes), 2):
        if i + 1 >= len(changes):
            break
        silence_start = changes[i]
        silence_end = changes[i + 1]
        silence_duration = (silence_end - silence_start) * time_per_window
        if silence_duration >= min_duration:
            total_silence_duration += silence_duration
    total_duration = len(audio_array) / sample_rate
    silence_percentage = (total_silence_duration / total_duration * 100) if total_duration > 0 else 0.0
    return total_silence_duration, silence_percentage

class TrimsilencesPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        if 'temp_silence_params' not in st.session_state:
            st.session_state.temp_silence_params = None
        if 'analyzed_audio' not in st.session_state:
            st.session_state.analyzed_audio = {}
        if 'current_analyzed_file' not in st.session_state:
            st.session_state.current_analyzed_file = None

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
            min_value=-90,
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

    def remove_silence(self, input_file: str, directory: str, progress_callback=None):
        threshold = self.plugin_manager.config['trimsilences']['silence_threshold']
        return self.remove_silence_simple(input_file, threshold, directory, progress_callback)

    def remove_silence_legacy(self, input_file: str, threshold: float, duration: float,
                       keep_duration: float, videos_dir: str,
                       progress_callback=None) -> tuple[str, str, float, float]:
        try:
            if progress_callback:
                progress_callback(0)
            video = VideoFileClip(input_file)
            original_duration = video.duration
            audio_array = video.audio.to_soundarray(fps=video.audio.fps)
            print(f"Type audio_array: {type(audio_array)}")
            print(f"Shape audio_array: {audio_array.shape}")
            print(f"Dtype audio_array: {audio_array.dtype}")
            print(f"Valeurs min/max audio_array: {audio_array.min()}, {audio_array.max()}")
            print(f"Sample rate: {video.audio.fps}")
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1).astype(np.float32)
                print(f"Après conversion mono - Shape: {audio_array.shape}")
            segments = detect_silence_segments(
                audio_array,
                video.audio.fps,
                threshold,
                duration,
                keep_duration,
                padding=0.2
            )
            print(f"Nombre de segments détectés: {len(segments)}")
            if progress_callback:
                progress_callback(33)
            print("Découpage des segments silencieux")
            clips = []
            for i, (start, end, silence_middle) in enumerate(segments):
                clip = video.subclip(start, end)
                clips.append(clip)
                if progress_callback:
                    progress = 33 + (i / len(segments) * 33)
                    progress_callback(int(progress))
            print("Concaténation des segments")
            final_video = concatenate_videoclips(clips)
            if progress_callback:
                progress_callback(66)
            output_filename = f"outfile_{os.path.basename(input_file)}"
            output_file = os.path.join(videos_dir, output_filename)
            print("Écriture du fichier final")
            final_video.write_videofile(output_file,
                                        codec='libx264',
                                        audio_codec='aac',
                                        temp_audiofile='temp-audio.m4a',
                                        remove_temp=True,
                                        audio_bitrate="192k",
                                        preset='medium')
            final_duration = final_video.duration
            if final_duration >= original_duration:
                error_msg = t("trim_silences_error").format(
                    error="Output video is not shorter than input video"
                )
                video.close()
                final_video.close()
                for clip in clips:
                    clip.close()
                if os.path.exists(output_file):
                    os.remove(output_file)
                return error_msg, "0%", original_duration, 0.0
            reduction_percentage = ((original_duration - final_duration) /
                                    original_duration * 100)
            reduction_str = f"{reduction_percentage:.1f}%"
            video.close()
            final_video.close()
            for clip in clips:
                clip.close()
            if progress_callback:
                progress_callback(100)
            return output_file, reduction_str, original_duration, final_duration
        except Exception as e:
            return t("trim_silences_error").format(error=str(e)), "0%", 0.0, 0.0

    def remove_silence_simple(self, input_file: str, threshold: float, videos_dir: str,
                             progress_callback=None) -> tuple[str, str, float, float]:
        try:
            st.info(f"Removing silence : {input_file}")
            if progress_callback:
                progress_callback(0)
            video = VideoFileClip(input_file)
            original_duration = video.duration
            audio_array = video.audio.to_soundarray(fps=video.audio.fps)
            print(f"Type audio_array: {type(audio_array)}")
            print(f"Shape audio_array: {audio_array.shape}")
            print(f"Dtype audio_array: {audio_array.dtype}")
            print(f"Valeurs min/max audio_array: {audio_array.min()}, {audio_array.max()}")
            print(f"Sample rate: {video.audio.fps}")
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1).astype(np.float32)
                print(f"Après conversion mono - Shape: {audio_array.shape}")
            segments = detect_silence_segments_simple(
                audio_array,
                video.audio.fps,
                threshold
            )
            print(f"Nombre de segments détectés (simple): {len(segments)}")
            if progress_callback:
                progress_callback(33)
            print("Découpage des segments non-silencieux")
            clips = []
            for i, (start, end) in enumerate(segments):
                clip = video.subclip(start, end)
                clips.append(clip)
                if progress_callback:
                    progress = 33 + (i / len(segments) * 33)
                    progress_callback(int(progress))
            print("Concaténation des segments")
            final_video = concatenate_videoclips(clips)
            if progress_callback:
                progress_callback(66)
            output_filename = f"outfile_simple_{os.path.basename(input_file)}"
            output_file = os.path.join(videos_dir, output_filename)
            print("Écriture du fichier final")
            final_video.write_videofile(output_file,
                                        codec='libx264',
                                        audio_codec='aac',
                                        temp_audiofile='temp-audio.m4a',
                                        remove_temp=True,
                                        audio_bitrate="192k",
                                        preset='medium')
            final_duration = final_video.duration
            if final_duration >= original_duration:
                error_msg = t("trim_silences_error").format(
                    error="Output video is not shorter than input video"
                )
                video.close()
                final_video.close()
                for clip in clips:
                    clip.close()
                if os.path.exists(output_file):
                    os.remove(output_file)
                return error_msg, "0%", original_duration, 0.0
            reduction_percentage = ((original_duration - final_duration) /
                                   original_duration * 100)
            reduction_str = f"{reduction_percentage:.1f}%"
            video.close()
            final_video.close()
            for clip in clips:
                clip.close()
            if progress_callback:
                progress_callback(100)
            return output_file, reduction_str, original_duration, final_duration
        except Exception as e:
            return t("trim_silences_error").format(error=str(e)), "0%", 0.0, 0.0

    def normalize_audio(self, input_file: str, reference_audio_path: str):
        try:
            normalize_full_audio(input_file, reference_audio_path, make_backup=True)
            return f"Normalization completed for {input_file}", "100%"
        except Exception as e:
            return t("trim_silences_error").format(error=str(e)), "0%"

    def run(self, config):
        st.header(t("trim_silences_header"))

        st.subheader(t("trim_silences_params"))
        if st.session_state.temp_silence_params is None:
            st.session_state.temp_silence_params = {
                "silence_threshold": config['trimsilences']['silence_threshold'],
                "silence_duration": config['trimsilences']['silence_duration'],
                "keep_duration": config['trimsilences']['keep_duration']
            }

        col1, col2, col3 = st.columns(3)
        with col1:
            temp_threshold = st.slider(
                t("trim contrato_silences_threshold_label"),
                min_value=-90,
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

        st.session_state.temp_silence_params = {
            "silence_threshold": temp_threshold,
            "silence_duration": temp_duration,
            "keep_duration": temp_keep_duration
        }

        # Récupérer les fichiers vidéo (.mp4 et .mkv) avec list_video_files2
        all_videos = list_video_files2(config['common']['work_directory'], extensions=['.mp4', '.mkv'])
        st.session_state['list_video_files'] = all_videos

        # Sélecteur multiple pour les fichiers vidéo
        video_options = [(file, full_path) for file, full_path, _ in all_videos]
        video_names = [file for file, _ in video_options]
        selected_videos = st.multiselect(
            t("select_videos_label"),
            options=video_names,
            default=[],
            key="selected_videos"
        )

        # Boutons pour le traitement en masse
        col_batch1, col_batch2, col_batch3 = st.columns(3)
        with col_batch1:
            if st.button(t("trim_silences_button") + " (Batch)"):
                if not selected_videos:
                    st.warning("Please select at least one video.")
                else:
                    for file in selected_videos:
                        full_path = next(full_path for fname, full_path in video_options if fname == file)
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        def update_progress(progress):
                            progress_bar.progress(progress)
                            progress_text.text(
                                t("trim_silences_progress").format(progress=progress))

                        with st.spinner(t("trim_silences_processing").format(file=file)):
                            result, reduction, _, _ = self.remove_silence_legacy(
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
                            st.error(f"{file}: {result}")
                        else:
                            st.success(
                                f"{file}: {t('trim_silences_success').format(result=result)} - Reduction: {reduction}"
                            )
                    st.rerun()

        with col_batch2:
            if st.button(t("trim_silences_simple_button") + " (Batch)"):
                if not selected_videos:
                    st.warning("Please select at least one video.")
                else:
                    for file in selected_videos:
                        full_path = next(full_path for fname, full_path in video_options if fname == file)
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        def update_progress(progress):
                            progress_bar.progress(progress)
                            progress_text.text(
                                t("trim_silences_progress").format(progress=progress))

                        with st.spinner(t("trim_silences_processing").format(file=file)):
                            result, reduction, _, _ = self.remove_silence_simple(
                                full_path,
                                st.session_state.temp_silence_params["silence_threshold"],
                                config['common']['work_directory'],
                                update_progress
                            )

                        progress_bar.empty()
                        progress_text.empty()

                        if result.startswith(t("trim_silences_error").format(error="")):
                            st.error(f"{file}: {result}")
                        else:
                            st.success(
                                f"{file}: {t('trim_silences_success').format(result=result)} - Reduction: {reduction}"
                            )
                    st.rerun()

        with col_batch3:
            if st.button("Normalize (Batch)"):
                if not selected_videos:
                    st.warning("Please select at least one video.")
                else:
                    reference_audio_path = config.get("movied", {}).get("movied_reference_audio", "")
                    for file in selected_videos:
                        full_path = next(full_path for fname, full_path in video_options if fname == file)
                        with st.spinner(f"Normalizing {file}..."):
                            result, _ = self.normalize_audio(full_path, reference_audio_path)
                        if result.startswith(t("trim_silences_error").format(error="")):
                            st.error(f"{file}: {result}")
                        else:
                            st.success(f"{file}: {result}")
                    st.rerun()

        st.subheader(t("trim_silences_original_videos"))
        for file, full_path, _ in all_videos:
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
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
                        result, reduction, _, _ = self.remove_silence_legacy(
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
            with col3:
                if st.button(t("trim_silences_simple_button"), key=f"remove_silence_simple_{file}"):
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def update_progress(progress):
                        progress_bar.progress(progress)
                        progress_text.text(
                            t("trim_silences_progress").format(progress=progress))

                    with st.spinner(t("trim_silences_processing").format(file=file)):
                        result, reduction, _, _ = self.remove_silence_simple(
                            full_path,
                            st.session_state.temp_silence_params["silence_threshold"],
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
            with col4:
                if st.button("Normalize", key=f"normalize_{file}"):
                    reference_audio_path = config.get("movied", {}).get("movied_reference_audio", "")
                    with st.spinner("Normalizing audio..."):
                        result, _ = self.normalize_audio(full_path, reference_audio_path)
                    if result.startswith(t("trim_silences_error").format(error="")):
                        st.error(result)
                    else:
                        st.success(result)
                        st.rerun()
            with col5:
                if st.button(t("analyze_button"), key=f"analyze_{file}"):
                    with st.spinner("Analyzing audio..."):
                        video = VideoFileClip(full_path)
                        audio_array = video.audio.to_soundarray(fps=video.audio.fps)
                        sample_rate = video.audio.fps
                        video.close()

                        st.session_state.analyzed_audio[file] = {
                            "audio_array": audio_array,
                            "sample_rate": sample_rate,
                            "duration": len(audio_array) / sample_rate
                        }
                        st.session_state.current_analyzed_file = file

        if st.session_state.current_analyzed_file and st.session_state.current_analyzed_file in st.session_state.analyzed_audio:
            file = st.session_state.current_analyzed_file
            audio_data = st.session_state.analyzed_audio[file]
            audio_array = audio_data["audio_array"]
            sample_rate = audio_data["sample_rate"]

            st.subheader(f"Analysis for {file}")
            granularity = st.selectbox(
                t("analyze_granularity_label"),
                [t("analyze_granularity_seconds"), t("analyze_granularity_frames")],
                key=f"granularity_{file}"
            )
            granularity_value = "seconds" if granularity == t("analyze_granularity_seconds") else "frames"
            max_level_db, min_level_db, times, volume_levels_db = analyze_audio(
                audio_array, sample_rate, granularity_value
            )
            st.write(t("analyze_max_level").format(max_level=max_level_db))
            st.write(t("analyze_min_level").format(min_level=min_level_db))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(times, volume_levels_db)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Volume (dB)")
            ax.set_title(t("analyze_volume_plot"))
            st.pyplot(fig)
            analyze_threshold = st.slider(
                t("analyze_silence_threshold_label"),
                min_value=-90,
                max_value=0,
                value=-35,
                key=f"analyze_threshold_{file}"
            )
            silence_duration, silence_percentage = calculate_silence_duration(
                audio_array,
                sample_rate,
                analyze_threshold,
                st.session_state.temp_silence_params["silence_duration"]
            )
            st.write(t("analyze_silence_removed").format(
                seconds=f"{silence_duration:.2f}",
                percentage=f"{silence_percentage:.1f}"
            ))
