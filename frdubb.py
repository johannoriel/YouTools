import whisper
import torch
from TTS.api import TTS
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips, AudioClip
import moviepy.video.fx.all as vfx
from moviepy.editor import concatenate_videoclips
import litellm
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import tempfile
import re

class VideoDubber:
    def __init__(self, input_video_path, output_video_path, model, threshold):
        self.whisper_model = whisper.load_model("medium")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model = model
        self.threshold = threshold
        
        video = VideoFileClip(input_video_path)
        self.reference_audio_path = self.temp_dir / "original_audio.wav"
        video.audio.write_audiofile(str(self.reference_audio_path))
        self.video_duration = video.duration
        video.close()

    def trim_silence(self, audio_path):
        # Charger l'audio
        y, sr = librosa.load(str(audio_path))
        
        # Détecter les non-silences
        non_silent = librosa.effects.split(
            y,
            top_db=self.threshold,  # Ajustez cette valeur selon vos besoins
            ref=0,
            frame_length=2048,
            hop_length=512
        )
        
        if len(non_silent) > 0:
            # Prendre le premier et dernier segment non-silencieux
            start_sample = non_silent[0][0]
            end_sample = non_silent[-1][1]
            
            # Trimmer l'audio
            y_trimmed = y[start_sample:end_sample]
            
            # Sauvegarder le fichier trimé
            sf.write(str(audio_path), y_trimmed, sr)
    
    def generate_french_audio(self, text, segment_index):
        output_path = self.temp_dir / f"segment_{segment_index}.wav"
        
        # Générer l'audio avec XTTS v2
        self.tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=str(self.reference_audio_path),
            language="fr"
        )
        
        # Nettoyer les silences
        self.trim_silence(output_path)
        
        return output_path
    
    def clean_translation(self, translated_text):
        cleaned = re.sub(r'\(Note:.*?\)', '', translated_text)
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def transcribe_audio(self):
        result = self.whisper_model.transcribe(str(self.reference_audio_path))
        return result['segments']
    
    def translate_segment(self, text):
        response = litellm.completion(
            model=self.model,
            messages=[{
                "role": "system",
                "content": "Tu es un traducteur professionnel anglais-français. Traduis uniquement le texte fourni sans ajouter de commentaires ou de notes. Conserve le ton et le style."
            }, {
                "role": "user",
                "content": f"Traduis en français : {text}"
            }]
        )
        translated = response.choices[0].message.content
        return self.clean_translation(translated)
    
    def process_segments(self):
        segments = self.transcribe_audio()
        processed_segments = []
        
        for idx, segment in enumerate(segments):
            translated_text = self.translate_segment(segment['text'])
            audio_path = self.generate_french_audio(translated_text, idx)
            
            audio = AudioFileClip(str(audio_path))
            
            processed_segments.append({
                'text': translated_text,
                'audio_path': audio_path,
                'start': segment['start'],
                'end': segment['end'],
                'original_duration': segment['end'] - segment['start'],
                'french_duration': audio.duration
            })
            
            audio.close()
            print(f"Segment {idx + 1}/{len(segments)} traité")
        
        return processed_segments
    
    def create_dubbed_video(self):
        processed_segments = self.process_segments()
        video = VideoFileClip(self.input_video_path)
        
        video_clips = []
        audio_clips = []
        
        for segment in processed_segments:
            # Audio français
            french_audio = AudioFileClip(str(segment['audio_path']))
            audio_clips.append(french_audio)
            
            # Ajuster la vitesse de la vidéo pour correspondre à l'audio français
            video_segment = video.subclip(segment['start'], segment['end'])
            speed_factor = segment['french_duration'] / segment['original_duration']
            adjusted_video = video_segment.fx(vfx.speedx, 1/speed_factor)
            video_clips.append(adjusted_video)
        
        # Concaténer tous les clips
        final_audio = concatenate_audioclips(audio_clips)
        final_video = concatenate_videoclips(video_clips)
        
        # Assembler la vidéo finale
        final_video = final_video.set_audio(final_audio)
        
        # Écrire le fichier final
        final_video.write_videofile(
            self.output_video_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps
        )
        
        # Nettoyage
        video.close()
        final_video.close()
        for clip in audio_clips:
            clip.close()
        for clip in video_clips:
            clip.close()
        
        for file in self.temp_dir.glob("*.wav"):
            file.unlink()
        self.temp_dir.rmdir()

if __name__ == "__main__":
    dubber = VideoDubber('video_source.mp4', 'video_doublee.mp4', 'ollama/qwen2.5', -30)
    dubber.create_dubbed_video()
