from lib.global_vars import translations, t
import os
import numpy as np
from typing import List, Tuple
import streamlit as st
from app import Plugin
from plugins.common import list_video_files2
from moviepy import VideoFileClip, concatenate_videoclips
from lib.video_utils import normalize_full_audio
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import subprocess
import pandas as pd

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
    "compression_graph_button": "Generate Compression Graph",
    "compression_graph_title": "Silence Removal Percentage by dB Threshold",
    "ff_normalize_button": "FF Normalize",
    "ff_normalize_processing": "FFmpeg normalization in progress for {file}...",
    "ff_normalize_success": "FFmpeg normalization completed. Output file: {result}",
    "ff_normalize_error": "Error during FFmpeg normalization: {error}",
    "enhance_button": "Enhance Audio",
    "enhance_processing": "Enhancing audio for {file}...",
    "enhance_success": "Audio enhancement completed. Output file: {result}",
    "enhance_error": "Error during audio enhancement: {error}",
    "resemble_enhance_dir_label": "Resemble Enhance Directory",
    "merge_videos_label": "Merge processed videos",
    "merge_only_button": "Merge Selected Videos",
    "merge_success": "Successfully merged {count} videos",
    "merge_error": "Error merging videos: {error}",
    "no_videos_selected": "No videos selected for processing",
    "batchsilences_normalizing_audio": "Normalizing audio...",
    "batchsilences_reorder_videos": "Reorder Videos",
    "batchsilences_reorder_instructions": "Drag and drop to reorder the videos for merging.",
    "batchsilences_normalize_audio" : "FF normalize",
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
    "compression_graph_button": "Générer le graphique de compression",
    "compression_graph_title": "Pourcentage de suppression des silences par seuil de dB",
    "ff_normalize_button": "FF Normaliser",
    "ff_normalize_processing": "Normalisation FFmpeg en cours pour {file}...",
    "ff_normalize_success": "Normalisation FFmpeg terminée. Fichier de sortie : {result}",
    "ff_normalize_error": "Erreur lors de la normalisation FFmpeg : {error}",
    "enhance_button": "Améliorer l'audio",
    "enhance_processing": "Amélioration de l'audio pour {file}...",
    "enhance_success": "Amélioration de l'audio terminée. Fichier de sortie : {result}",
    "enhance_error": "Erreur lors de l'amélioration de l'audio : {error}",
    "resemble_enhance_dir_label": "Répertoire de Resemble Enhance",
    "merge_videos_label": "Fusionner les vidéos traitées",
    "merge_only_button": "Fusionner les vidéos sélectionnées",
    "merge_success": "{count} vidéos fusionnées avec succès",
    "merge_error": "Erreur lors de la fusion des vidéos : {error}",
    "no_videos_selected": "Aucune vidéo sélectionnée pour le traitement",
    "batchsilences_normalizing_audio": "Normalisation de l'audio en cours...",
    "batchsilences_reorder_videos": "Réorganiser les Vidéos",
    "batchsilences_reorder_instructions": "Glissez-déposez pour réorganiser l'ordre des vidéos avant la fusion.",
    "batchsilences_normalize_audio" : "FF normalize",
})

import os
import subprocess
import shutil
from pathlib import Path
import sys
import torchaudio
import torch

def enhance_audio_with_resemble(input_file: str, output_dir: str, resemble_enhance_dir: str, run_dir: str = None,
                                device: str = "cuda", lambd: float = 1.0, tau: float = 0.5, solver: str = "midpoint",
                                nfe: int = 64, progress_callback=None) -> tuple[str, str]:
    """
    Extrait l'audio d'une vidéo, applique l'amélioration avec resemble-enhance, et recombine avec la vidéo.
    Crée un fichier de backup avant modification.

    Args:
        input_file (str): Chemin vers le fichier vidéo d'entrée.
        output_dir (str): Répertoire où sauvegarder la vidéo améliorée.
        resemble_enhance_dir (str): Chemin vers le dossier resemble-enhance.
        run_dir (str, optional): Chemin vers le dossier du modèle resemble-enhance.
        device (str): Device pour le calcul ("cuda" ou "cpu").
        lambd (float): Force de débruitage (0.0 à 1.0).
        tau (float): Température du prior CFM (0.0 à 1.0).
        solver (str): Solveur numérique ("midpoint", "rk4", "euler").
        nfe (int): Nombre d'évaluations de fonction.
        progress_callback (callable, optional): Fonction pour mettre à jour la progression.

    Returns:
        tuple[str, str]: (Message de résultat, Pourcentage de progression)
    """
    try:
        if progress_callback:
            progress_callback(0)

        # Vérifier si resemble-enhance est accessible
        try:
            sys.path.append(resemble_enhance_dir)
            from resemble_enhance.enhancer.inference import enhance
        except ImportError as e:
            return (
                f"Erreur : Impossible d'importer resemble-enhance : {str(e)}. "
                "Veuillez installer le module via 'git clone https://github.com/resemble-ai/resemble-enhance.git' "
                "et configurer le chemin du répertoire dans les paramètres du plugin."
            ), "0%"

        # Créer un fichier de backup
        backup_file = input_file + ".backup"
        shutil.copy2(input_file, backup_file)

        # Extraire l'audio en .wav
        temp_audio = os.path.join(output_dir, "temp_audio.wav")
        ffmpeg_extract_cmd = [
            "ffmpeg", "-y", "-i", input_file, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", temp_audio
        ]
        process = subprocess.run(ffmpeg_extract_cmd, capture_output=True, text=True, encoding='utf-8')
        if process.returncode != 0:
            return f"Erreur lors de l'extraction audio : {process.stderr}", "0%"

        if progress_callback:
            progress_callback(33)

        # Charger l'audio pour l'amélioration
        dwav, sr = torchaudio.load(temp_audio)
        dwav = dwav.mean(0)  # Convertir en mono

        # Appliquer l'amélioration avec resemble-enhance
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        hwav, sr = enhance(
            dwav=dwav,
            sr=sr,
            device=device,
            nfe=nfe,
            solver=solver,
            lambd=lambd,
            tau=tau,
            run_dir=run_dir,
        )

        # Sauvegarder l'audio amélioré
        enhanced_audio = os.path.join(output_dir, "enhanced_audio.wav")
        torchaudio.save(enhanced_audio, hwav[None], sr)

        if progress_callback:
            progress_callback(66)

        # Recombiner l'audio amélioré avec la vidéo
        output_filename = f"enhanced_{os.path.basename(input_file)}"
        output_file = os.path.join(output_dir, output_filename)
        ffmpeg_combine_cmd = [
            "ffmpeg", "-y", "-i", input_file, "-i", enhanced_audio,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0", output_file
        ]
        try:
            process = subprocess.run(
                ffmpeg_combine_cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            if process.returncode != 0:
                return f"Erreur lors de la recombination audio-vidéo : {process.stderr}", "0%"
        except UnicodeDecodeError as e:
            return f"Erreur de décodage Unicode lors de la recombination : {str(e)}", "0%"

        # Nettoyer les fichiers temporaires
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        if os.path.exists(enhanced_audio):
            os.remove(enhanced_audio)

        if progress_callback:
            progress_callback(100)

        return f"Amélioration audio terminée. Fichier de sortie : {output_file}", "100%"

    except Exception as e:
        return f"Erreur lors de l'amélioration audio : {str(e)}", "0%"

def get_video_metadata(_file_path: str) -> tuple[str, str]:
    """Calcule la durée et la résolution d'une vidéo avec mise en cache dans st.session_state."""
    if 'video_metadata' not in st.session_state:
        st.session_state.video_metadata = {}

    if _file_path in st.session_state.video_metadata:
        return st.session_state.video_metadata[_file_path]

    try:
        video = VideoFileClip(_file_path)
        duration = video.duration
        duration_str = f"{int(duration // 60)}:{int(duration % 60):02d}"  # Format MM:SS
        resolution = video.size
        resolution_str = f"{resolution[0]}x{resolution[1]}"
        video.close()
        st.session_state.video_metadata[_file_path] = (duration_str, resolution_str)
        return duration_str, resolution_str
    except Exception as e:
        return f"Erreur ({e})", f"Erreur ({e})"

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

def calculate_compression_data(audio_array: np.ndarray, sample_rate: int, min_duration: float) -> tuple[list, list]:
    """Calcule le pourcentage de silence supprimé pour une plage de seuils de dB."""
    db_thresholds = list(range(-90, 1, 1))  # De -90 dB à 0 dB, par pas de 5
    percentages = []
    for threshold_db in db_thresholds:
        _, percentage = calculate_silence_duration(audio_array, sample_rate, threshold_db, min_duration)
        percentages.append(percentage)
    return db_thresholds, percentages

def analyze_spectrum(audio_array: np.ndarray, sample_rate: int, num_bands: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Analyse le spectre fréquentiel d'un signal audio et retourne les amplitudes moyennes par bande."""
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1).astype(np.float32)

    # Calcul de la FFT
    N = len(audio_array)
    fft_result = fft(audio_array)
    freqs = fftfreq(N, 1 / sample_rate)
    magnitudes = np.abs(fft_result[:N//2])  # Prendre la moitié positive du spectre

    # Définir les bandes de fréquences (logarithmique)
    freq_bins = np.logspace(np.log10(20), np.log10(sample_rate/2), num_bands + 1)
    band_magnitudes = []

    for i in range(num_bands):
        mask = (freqs[:N//2] >= freq_bins[i]) & (freqs[:N//2] < freq_bins[i+1])
        band_magnitude = np.mean(magnitudes[mask]) if np.any(mask) else 0
        band_magnitudes.append(band_magnitude)

    band_centers = (freq_bins[:-1] + freq_bins[1:]) / 2  # Fréquences centrales des bandes
    return band_centers, np.array(band_magnitudes)

def generate_equalizer_filters(ref_magnitudes: np.ndarray, target_magnitudes: np.ndarray, band_centers: np.ndarray) -> str:
    """Génère une chaîne de filtres equalizer pour FFmpeg basée sur les différences spectrales."""
    # Calculer les différences en dB (logarithmique)
    ref_magnitudes = np.clip(ref_magnitudes, 1e-10, None)  # Éviter division par zéro
    target_magnitudes = np.clip(target_magnitudes, 1e-10, None)
    gains_db = 20 * np.log10(ref_magnitudes / target_magnitudes)

    # Limiter les gains pour éviter des ajustements extrêmes
    gains_db = np.clip(gains_db, -12, 12)

    # Générer les filtres equalizer avec width_type=o et w=1 (1 octave)
    equalizer_filters = []
    for freq, gain in zip(band_centers, gains_db):
        equalizer_filters.append(f"equalizer=f={freq}:width_type=o:width=1:g={gain}")

    return ",".join(equalizer_filters)

class TrimsilencesPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        if 'temp_silence_params' not in st.session_state:
            st.session_state.temp_silence_params = None
        if 'analyzed_audio' not in st.session_state:
            st.session_state.analyzed_audio = {}
        if 'current_analyzed_file' not in st.session_state:
            st.session_state.current_analyzed_file = None
        if 'compression_data' not in st.session_state:
            st.session_state.compression_data = {}  # Pour stocker les données de compression

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
            },
            "resemble_enhance_dir": {
                "type": "text",
                "label": t("resemble_enhance_dir_label"),
                "default": str(Path.home() / "Evaluation" / "resemble-enhance")
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
        updated_config["resemble_enhance_dir"] = st.text_input(
            t("resemble_enhance_dir_label"),
            value=config.get("resemble_enhance_dir", str(Path.home() / "Evaluation" / "resemble-enhance"))
        )
        return updated_config

    def get_tabs(self):
        return [{"name": t("trim_silences_tab"), "plugin": "trimsilences"}]

    def remove_silence(self, input_file: str, directory: str, progress_callback=None):
        threshold = self.plugin_manager.config['trimsilences']['silence_threshold']
        return self.remove_silence_simple(input_file, threshold, directory, progress_callback)

    def ff_normalize(self, input_file: str, reference_audio_path: str, videos_dir: str, progress_callback=None) -> tuple[str, str]:
        try:
            if progress_callback:
                progress_callback(0)

            # Charger l'audio de référence pour analyse
            if not reference_audio_path or not os.path.exists(reference_audio_path):
                return t("ff_normalize_error").format(error="Reference audio file not found"), "0%"

            reference_video = VideoFileClip(reference_audio_path)
            reference_audio_array = reference_video.audio.to_soundarray(fps=reference_video.audio.fps)
            reference_sample_rate = reference_video.audio.fps
            reference_video.close()

            # Analyser l'audio de référence
            max_level_db, min_level_db, _, _ = analyze_audio(reference_audio_array, reference_sample_rate, granularity="seconds")
            target_loudness = max_level_db - 3  # Viser 3 dB en dessous du max pour éviter le clipping

            # Construire la commande FFmpeg avec des filtres pour réduire l'écho
            output_filename = f"ff_normalized_{os.path.basename(input_file)}"
            output_file = os.path.join(videos_dir, output_filename)

            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-i", input_file,
                "-af", f"afftdn=nr=15:nf=-35,highpass=f=150,lowpass=f=8000,acompressor=threshold=-30dB:ratio=4:attack=20:release=100,deesser,loudnorm=I={target_loudness}:TP=-1.5:LRA=11",
                "-c:v", "copy",  # Conserver la vidéo intacte
                "-c:a", "aac",
                "-b:a", "192k",
                output_file
            ]

            if progress_callback:
                progress_callback(33)

            # Exécuter la commande FFmpeg
            import subprocess
            process = subprocess.run(ffmpeg_command, capture_output=True, text=True)

            if process.returncode != 0:
                return t("ff_normalize_error").format(error=process.stderr), "0%"

            if progress_callback:
                progress_callback(100)

            return t("ff_normalize_success").format(result=output_file), "100%"

        except Exception as e:
            return t("ff_normalize_error").format(error=str(e)), "0%"

    def ff_normalize_spectrum(self, input_file: str, reference_audio_path: str, videos_dir: str, progress_callback=None) -> tuple[str, str]:
        try:
            if progress_callback:
                progress_callback(0)

            # Charger l'audio de référence et l'audio cible pour analyse
            if not reference_audio_path or not os.path.exists(reference_audio_path):
                return t("ff_normalize_error").format(error="Reference audio file not found"), "0%"

            reference_video = VideoFileClip(reference_audio_path)
            reference_audio_array = reference_video.audio.to_soundarray(fps=reference_video.audio.fps)
            reference_sample_rate = reference_video.audio.fps
            reference_video.close()

            target_video = VideoFileClip(input_file)
            target_audio_array = target_video.audio.to_soundarray(fps=target_video.audio.fps)
            target_sample_rate = target_video.audio.fps
            target_video.close()

            # Analyser l'audio de référence
            max_level_db, min_level_db, _, _ = analyze_audio(reference_audio_array, reference_sample_rate, granularity="seconds")
            target_loudness = max_level_db - 3  # Viser 3 dB en dessous du max pour éviter le clipping

            # Analyser les spectres
            band_centers, ref_magnitudes = analyze_spectrum(reference_audio_array, reference_sample_rate)
            _, target_magnitudes = analyze_spectrum(target_audio_array, target_sample_rate)

            # Générer les filtres equalizer
            equalizer_filter = generate_equalizer_filters(ref_magnitudes, target_magnitudes, band_centers)

            # Construire la commande FFmpeg avec égalisation
            output_filename = f"ff_normalized_{os.path.basename(input_file)}"
            output_file = os.path.join(videos_dir, output_filename)

            st.write(equalizer_filter)

            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-i", input_file,
                "-af", f"afftdn=nr=15:nf=-35,{equalizer_filter},highpass=f=150,lowpass=f=8000,acompressor=threshold=-30dB:ratio=4:attack=20:release=100,deesser,loudnorm=I={target_loudness}:TP=-1.5:LRA=11",
                "-c:v", "copy",  # Conserver la vidéo intacte
                "-c:a", "aac",
                "-b:a", "192k",
                output_file
            ]

            if progress_callback:
                progress_callback(33)

            # Exécuter la commande FFmpeg
            process = subprocess.run(ffmpeg_command, capture_output=True, text=True)

            if process.returncode != 0:
                return t("ff_normalize_error").format(error=process.stderr), "0%"

            if progress_callback:
                progress_callback(100)

            return t("ff_normalize_success").format(result=output_file), "100%"

        except Exception as e:
            return t("ff_normalize_error").format(error=str(e)), "0%"

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
                clip = video.subcliped(start, end)
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
                clip = video.subclipped(start, end)
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

    def analyze_audio_ui(self, config):
        """Gère l'interface et la logique pour l'analyse audio d'un fichier vidéo."""
        st.subheader(t("analyze_button"))

        all_videos = list_video_files2(config['common']['work_directory'], extensions=['.mp4', '.mkv'])
        video_options = [(file, full_path) for file, full_path, _ in all_videos]
        video_names = [file for file, _ in video_options]

        selected_videos = st.multiselect(
            t("select_videos_label"),
            options=video_names,
            default=[],
            key="analyze_selected_videos"
        )

        if len(selected_videos) > 1:
            st.warning("Please select only one video for audio analysis.")
            return

        if selected_videos:
            file = selected_videos[0]
            full_path = next(full_path for fname, full_path in video_options if fname == file)

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

            if st.session_state.current_analyzed_file == file and file in st.session_state.analyzed_audio:
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

                if st.button(t("compression_graph_button"), key=f"compression_graph_{file}"):
                    with st.spinner("Generating compression graph..."):
                        db_thresholds, percentages = calculate_compression_data(
                            audio_array,
                            sample_rate,
                            st.session_state.temp_silence_params["silence_duration"]
                        )
                        st.session_state.compression_data[file] = {
                            "db_thresholds": db_thresholds,
                            "percentages": percentages
                        }

                if file in st.session_state.compression_data:
                    db_thresholds = st.session_state.compression_data[file]["db_thresholds"]
                    percentages = st.session_state.compression_data[file]["percentages"]
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(db_thresholds, percentages, marker='o')
                    ax.set_xlabel("dB Threshold")
                    ax.set_ylabel("Percentage of Video Removed (%)")
                    ax.set_title(t("compression_graph_title"))
                    ax.grid(True)
                    ax.set_ylim(0, 100)
                    st.pyplot(fig)

    def merge_videos(self, video_paths: List[str], output_path: str) -> str:
        """
        Fusionne plusieurs vidéos en une seule.

        Args:
            video_paths: Liste des chemins des vidéos à fusionner
            output_path: Chemin de sortie pour la vidéo fusionnée

        Returns:
            Chemin de la vidéo fusionnée ou message d'erreur
        """
        try:
            clips = [VideoFileClip(path) for path in video_paths]
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True, audio_bitrate="192k")
            for clip in clips:
                clip.close()
            final_clip.close()
            return output_path
        except Exception as e:
            return t("merge_error").format(error=str(e))

    def post_process_videos(self, video_paths: List[str], config: dict, output_dir: str) -> tuple[str, str]:
        """
        Post-traite les vidéos : fusionne si activé et normalise l'audio si activé.

        Args:
            video_paths: Liste des chemins des vidéos à traiter
            config: Configuration du plugin
            output_dir: Répertoire de sortie

        Returns:
            tuple[str, str]: (Message de résultat, Pourcentage de progression)
        """
        try:
            final_output = video_paths
            result_message = ""

            # Fusion si activée
            if st.session_state.get("merge_videos", False) and len(video_paths) > 1:
                output_path = os.path.join(output_dir, "merged_video.mp4")
                merge_result = self.merge_videos(video_paths, output_path)
                if merge_result.startswith(t("merge_error").format(error="")):
                    return merge_result, "0%"
                final_output = [merge_result]
                result_message = t("merge_success").format(count=len(video_paths))

            # Normalisation si activée
            if st.session_state.get("normalize_audio", False):
                reference_audio_path = config.get("movied", {}).get("movied_reference_audio", "")
                for video_path in final_output:
                    with st.spinner(t("batchsilences_normalizing_audio")):
                        norm_result, _ = self.ff_normalize(video_path, reference_audio_path, output_dir)
                        if norm_result.startswith(t("ff_normalize_error").format(error="")):
                            return norm_result, "0%"
                        result_message = f"{result_message} {norm_result}" if result_message else norm_result

            return result_message or "Processing completed", "100%"
        except Exception as e:
            return t("merge_error").format(error=str(e)), "0%"

    def run(self, config):
        st.header(t("trim_silences_header"))

        # Paramètres de détection des silences
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
                t("trim_silences_threshold_label"),
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

        # Liste des vidéos avec sélection multiple
        st.subheader(t("trim_silences_original_videos"))
        from widgets.file_selector import FileSelectorWidget
        file_selector = FileSelectorWidget("trimsilences", "trimsilences", plugin_manager=self.plugin_manager)
        selected_files = file_selector.display(
            mode="video",
            allowed_extensions=['.mp4', '.mkv']
        )
        selected_names = [os.path.basename(f) for f in selected_files]

        # Checkbox pour fusionner les vidéos traitées
        st.checkbox(
            t("merge_videos_label"),
            value=False,
            key="merge_videos"
        )

        # Checkbox pour normaliser l'audio
        st.checkbox(
            t("batchsilences_normalize_audio"),
            value=True,
            key="normalize_audio"
        )

        # Initialiser les listes ordonnées avec les valeurs par défaut
        ordered_files = selected_files
        ordered_names = selected_names

        # Boutons pour les opérations de masse
        col_batch1, col_batch2, col_batch3, col_batch4, col_batch5 = st.columns(5)
        processed_videos = []
        with col_batch1:
            if st.button(t("trim_silences_button")):
                if not selected_files:
                    st.warning(t("no_videos_selected"))
                else:
                    processed_videos = []
                    for file, name in zip(ordered_files, ordered_names):
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        def update_progress(progress):
                            progress_bar.progress(progress)
                            progress_text.text(t("trim_silences_progress").format(progress=progress))

                        with st.spinner(t("trim_silences_processing").format(file=name)):
                            result, reduction, _, _ = self.remove_silence_legacy(
                                file,
                                st.session_state.temp_silence_params["silence_threshold"],
                                st.session_state.temp_silence_params["silence_duration"],
                                st.session_state.temp_silence_params["keep_duration"],
                                config['common']['work_directory'],
                                update_progress
                            )

                        progress_bar.empty()
                        progress_text.empty()

                        if result.startswith(t("trim_silences_error").format(error="")):
                            st.error(f"{name}: {result}")
                        else:
                            st.success(f"{name}: {t('trim_silences_success').format(result=result)} - Reduction: {reduction}")
                            processed_videos.append(result)

                    if processed_videos:
                        with st.spinner(t("batchsilences_processing")):
                            result, _ = self.post_process_videos(processed_videos, config, config['common']['work_directory'])
                            if result.startswith(t("merge_error").format(error="")) or result.startswith(t("ff_normalize_error").format(error="")):
                                st.error(result)
                            else:
                                st.success(result)

        with col_batch2:
            if st.button(t("trim_silences_simple_button")):
                if not selected_files:
                    st.warning(t("no_videos_selected"))
                else:
                    processed_videos = []
                    for file, name in zip(ordered_files, ordered_names):
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        def update_progress(progress):
                            progress_bar.progress(progress)
                            progress_text.text(t("trim_silences_progress").format(progress=progress))

                        with st.spinner(t("trim_silences_processing").format(file=name)):
                            result, reduction, _, _ = self.remove_silence_simple(
                                file,
                                st.session_state.temp_silence_params["silence_threshold"],
                                config['common']['work_directory'],
                                update_progress
                            )

                        progress_bar.empty()
                        progress_text.empty()

                        if result.startswith(t("trim_silences_error").format(error="")):
                            st.error(f"{name}: {result}")
                        else:
                            st.success(f"{name}: {t('trim_silences_success').format(result=result)} - Reduction: {reduction}")
                            processed_videos.append(result)

                    if processed_videos:
                        with st.spinner(t("batchsilences_processing")):
                            result, _ = self.post_process_videos(processed_videos, config, config['common']['work_directory'])
                            if result.startswith(t("merge_error").format(error="")) or result.startswith(t("ff_normalize_error").format(error="")):
                                st.error(result)
                            else:
                                st.success(result)

        with col_batch3:
            if st.button(t("ff_normalize_button")):
                if not selected_files:
                    st.warning(t("no_videos_selected"))
                else:
                    processed_videos = []
                    reference_audio_path = config.get("movied", {}).get("movied_reference_audio", "")
                    for file, name in zip(ordered_files, ordered_names):
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        def update_progress(progress):
                            progress_bar.progress(progress)
                            progress_text.text(t("ff_normalize_processing").format(file=name))

                        with st.spinner(t("ff_normalize_processing").format(file=name)):
                            result, _ = self.ff_normalize(
                                file,
                                reference_audio_path,
                                config['common']['work_directory'],
                                update_progress
                            )

                        progress_bar.empty()
                        progress_text.empty()

                        if result.startswith(t("ff_normalize_error").format(error="")):
                            st.error(f"{name}: {result}")
                        else:
                            st.success(f"{name}: {result}")
                            processed_videos.append(result)

                    if processed_videos:
                        with st.spinner(t("batchsilences_processing")):
                            result, _ = self.post_process_videos(processed_videos, config, config['common']['work_directory'])
                            if result.startswith(t("merge_error").format(error="")) or result.startswith(t("ff_normalize_error").format(error="")):
                                st.error(result)
                            else:
                                st.success(result)

        with col_batch4:
            if st.button(t("enhance_button")):
                if not selected_files:
                    st.warning(t("no_videos_selected"))
                else:
                    processed_videos = []
                    resemble_enhance_dir = config['trimsilences'].get('resemble_enhance_dir', str(Path.home() / "Evaluation" / "resemble-enhance"))
                    for file, name in zip(ordered_files, ordered_names):
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        def update_progress(progress):
                            progress_bar.progress(progress)
                            progress_text.text(t("enhance_processing").format(file=name))

                        with st.spinner(t("enhance_processing").format(file=name)):
                            result, _ = enhance_audio_with_resemble(
                                file,
                                config['common']['work_directory'],
                                resemble_enhance_dir,
                                progress_callback=update_progress
                            )

                        progress_bar.empty()
                        progress_text.empty()

                        if result.startswith(t("enhance_error").format(error="")):
                            st.error(f"{name}: {result}")
                        else:
                            st.success(f"{name}: {t('enhance_success').format(result=result)}")
                            processed_videos.append(result)

                    if processed_videos:
                        with st.spinner(t("batchsilences_processing")):
                            result, _ = self.post_process_videos(processed_videos, config, config['common']['work_directory'])
                            if result.startswith(t("merge_error").format(error="")) or result.startswith(t("ff_normalize_error").format(error="")):
                                st.error(result)
                            else:
                                st.success(result)

        with col_batch5:
            if st.button(t("merge_only_button")):
                if not selected_files:
                    st.warning(t("no_videos_selected"))
                else:
                    with st.spinner(t("batchsilences_processing")):
                        result, _ = self.post_process_videos(ordered_files, config, config['common']['work_directory'])
                        if result.startswith(t("merge_error").format(error="")) or result.startswith(t("ff_normalize_error").format(error="")):
                            st.error(result)
                        else:
                            st.success(result)

        # Section pour l'analyse audio
        self.analyze_audio_ui(config)
