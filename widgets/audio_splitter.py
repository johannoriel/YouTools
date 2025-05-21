from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import os
import subprocess
import tempfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
import spacy
from pyannote.audio import Pipeline
from plugins.transcript import TranscriptPlugin
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("YOUR_HF_TOKEN")

translations["en"].update({
    "step_saving_file": "Saving the uploaded file...",
    "step_extracting_audio": "Extracting audio from the file...",
    "step_splitting_audio": "Splitting audio into chunks...",
    "step_transcribing_audio": "Transcribing audio chunks...",
    "step_extracting_images": "Extracting images from video chunks...",
    "step_extracting_video": "Extracting video chunks without audio...",
    "split_method": "Split method",
    "split_by_whisper": "Whisper (medium)",
    "split_by_silence": "Silence (many chunks)",
    "split_by_phrase": "Phrase (fewer chunks)",
    "audio_splitter": "Audio Splitter",
    "select_podcast": "Select a podcast file (audio or video)",
    "split_audio": "Split Audio",
    "split_completed": "Audio splitting completed successfully!",
})

translations["fr"].update({
    "step_saving_file": "Enregistrement du fichier téléchargé...",
    "step_extracting_audio": "Extraction de l'audio du fichier...",
    "step_splitting_audio": "Découpage de l'audio en segments...",
    "step_transcribing_audio": "Transcription des segments audio...",
    "step_extracting_images": "Extraction des images des segments vidéo...",
    "step_extracting_video": "Extraction des segments vidéo sans audio...",
    "split_method": "Méthode de découpage",
    "split_by_whisper": "Whisper (intermédiaire)",
    "split_by_silence": "Silence (beaucoup de segments)",
    "split_by_phrase": "Phrase (moins de segments)",
    "audio_splitter": "Découpeur Audio",
    "select_podcast": "Sélectionnez un fichier podcast (audio ou vidéo)",
    "split_audio": "Découper l'Audio",
    "split_completed": "Découpage audio terminé avec succès !",
})

class AudioSplitterWidget(Widget):
    def __init__(self, name, prefix, plugin_manager, output_dir):
        super().__init__(name, prefix, plugin_manager)
        self.output_dir = os.path.expanduser(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.transcript_plugin = TranscriptPlugin("transcript", plugin_manager)

    def extract_audio(self, input_file, output_file):
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            output_file
        ]
        subprocess.run(ffmpeg_command, check=True)

    def extract_image_from_video(self, video_path, output_image_path, timestamp_ms):
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(timestamp_ms / 1000.0),
            "-vframes", "1",
            output_image_path
        ]
        subprocess.run(ffmpeg_command, check=True)

    def extract_video_chunk(self, input_file, output_video_path, start_ms, end_ms):
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-an",  # Remove audio
            "-ss", str(start_ms / 1000.0),
            "-to", str(end_ms / 1000.0),
            output_video_path
        ]
        subprocess.run(ffmpeg_command, check=True)

    def split_audio_by_silence(self, audio_file, min_silence_len, silence_thresh):
        audio = AudioSegment.from_wav(audio_file)
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        audio_paths = []
        start_times = []
        end_times = []
        current_time = 0
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(self.output_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            audio_paths.append(chunk_path)
            start_times.append(current_time)
            end_times.append(current_time + len(chunk))
            current_time += len(chunk)
        return chunks, audio_paths, start_times, end_times

    def split_audio_by_phrase(self, audio_file):
        sentence_timestamps = self.align_audio_with_text(audio_file)
        audio = AudioSegment.from_wav(audio_file)
        audio_paths = []
        chunks = []
        start_times = []
        end_times = []
        for i, (start, end) in enumerate(sentence_timestamps):
            chunk = audio[start:end]
            chunk_path = os.path.join(self.output_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            audio_paths.append(chunk_path)
            chunks.append(chunk)
            start_times.append(start)
            end_times.append(end)
        return chunks, audio_paths, start_times, end_times

    def align_audio_with_text(self, audio_file):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0", use_auth_token=hf_token)
        diarization = pipeline({"uri": "audio", "audio": audio_file})
        sentence_timestamps = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start * 1000
            end = turn.end * 1000
            sentence_timestamps.append((start, end))
        return sentence_timestamps

    def split_audio_by_whisper(self, audio_file):
        transcript = self.transcript_plugin.transcribe_video(audio_file, "srt")
        lines = transcript.strip().split('\n')
        segments = []
        i = 0
        current_text = ""
        start_ms = None
        while i < len(lines):
            if lines[i].isdigit():
                i += 1
                time_range = lines[i].strip()
                start, end = time_range.split(' --> ')
                if not start_ms:
                    start_ms = self.time_to_ms(start.replace(',', '.'))
                end_ms = self.time_to_ms(end.replace(',', '.'))
                i += 1
                current_text += " " + lines[i].strip()
            if current_text.endswith('.'):
                segments.append((start_ms, end_ms, current_text.strip()))
                current_text = ""
                start_ms = None
            i += 1
        if current_text:
            segments.append((start_ms, end_ms, current_text.strip()))

        audio = AudioSegment.from_wav(audio_file)
        chunks = []
        audio_paths = []
        start_times = []
        end_times = []
        for i, (start, end, _) in enumerate(segments):
            chunk = audio[start:end]
            chunk_path = os.path.join(self.output_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk)
            audio_paths.append(chunk_path)
            start_times.append(start)
            end_times.append(end)
        return chunks, audio_paths, start_times, end_times

    def time_to_ms(self, time_str):
        h, m, s = time_str.split(':')
        s, ms = s.split('.')
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    def transcribe_chunks(self, chunks, audio_paths):
        transcriptions = []
        progress_bar = st.progress(0)
        total_chunks = len(chunks)
        for i, (chunk, audio_path) in enumerate(zip(chunks, audio_paths)):
            transcript = self.transcript_plugin.transcribe_video(audio_path, "txt")
            transcript_path = os.path.join(self.output_dir, f"chunk_{i}.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            transcriptions.append(transcript)
            progress_bar.progress((i + 1) / total_chunks)
        return transcriptions

    def display(self, config):
        st.header(t("audio_splitter"))
        uploaded_file = st.file_uploader(t("select_podcast"), type=["mp3", "wav", "mp4", "avi", "mov", "mkv"], key=f"{self.prefix}_uploader")
        split_method = st.selectbox(
            t("split_method"),
            ["whisper", "silence", "phrase"],
            format_func=lambda x: {
                "whisper": t("split_by_whisper"),
                "silence": t("split_by_silence"),
                "phrase": t("split_by_phrase"),
            }[x],
            key=f"{self.prefix}_split_method"
        )

        if st.button(t("split_audio"), key=f"{self.prefix}_split_button"):
            if uploaded_file is None:
                st.warning(t("please_upload_file"))
                return []

            with st.spinner(t("processing")):
                st.info(t("step_saving_file"))
                input_file = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(input_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.info(t("step_extracting_audio"))
                audio_file = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
                self.extract_audio(input_file, audio_file)

                st.info(t("step_splitting_audio"))
                if split_method == "silence":
                    min_silence_len = int(config['podcasttovideo']['min_silence_len'])
                    silence_thresh = int(config['podcasttovideo']['silence_thresh'])
                    chunks, audio_paths, start_times, end_times = self.split_audio_by_silence(
                        audio_file, min_silence_len, silence_thresh)
                elif split_method == "phrase":
                    chunks, audio_paths, start_times, end_times = self.split_audio_by_phrase(audio_file)
                else:  # whisper
                    chunks, audio_paths, start_times, end_times = self.split_audio_by_whisper(audio_file)

                st.info(t("step_transcribing_audio"))
                transcriptions = self.transcribe_chunks(chunks, audio_paths)

                chunks_data = []
                is_video = uploaded_file.type.startswith("video")
                if is_video:
                    st.info(t("step_extracting_images"))
                    progress_bar = st.progress(0)
                    total_chunks = len(start_times)
                    for i, (start_ms, end_ms) in enumerate(zip(start_times, end_times)):
                        image_path = os.path.join(self.output_dir, f"chunk_{i}.png")
                        video_path = os.path.join(self.output_dir, f"chunk_{i}.mp4")
                        mid_time = (start_ms + end_ms) / 2
                        self.extract_image_from_video(input_file, image_path, mid_time)
                        self.extract_video_chunk(input_file, video_path, start_ms, end_ms)
                        chunks_data.append({
                            "audio": audio_paths[i],
                            "text": os.path.join(self.output_dir, f"chunk_{i}.txt"),
                            "image": image_path,
                            "video": video_path
                        })
                        progress_bar.progress((i + 1) / total_chunks)
                else:
                    for i in range(len(audio_paths)):
                        chunks_data.append({
                            "audio": audio_paths[i],
                            "text": os.path.join(self.output_dir, f"chunk_{i}.txt"),
                            "image": None,
                            "video": None
                        })

                self.cleanup(input_file, audio_file)
                st.success(t("split_completed"))
                return chunks_data
        return []

    def cleanup(self, input_file, audio_file):
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(audio_file):
            os.unlink(audio_file)
