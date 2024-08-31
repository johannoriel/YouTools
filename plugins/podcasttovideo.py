import streamlit as st
from app import Plugin
from global_vars import t, translations
import os
import subprocess
import tempfile
from pydub import AudioSegment
from pydub.silence import split_on_silence
from plugins.transcript import TranscriptPlugin
from plugins.articletovideo import ArticletovideoPlugin
from plugins.imggen import ImggenPlugin
from plugins.ragllm import RagllmPlugin
import traceback
import spacy
from pyannote.audio import Pipeline

from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("YOUR_HF_TOKEN")

# Add translations for this plugin
translations["en"].update({
    "podcasttovideo": "Podcast to Video",
    "select_podcast": "Select a podcast file (audio or video)",
    "process_podcast": "Process Podcast",
    "processing": "Processing...",
    "video_generated": "Video generated successfully!",
    "error_processing": "Error during processing: ",
    "step_saving_file": "Saving the uploaded file...",
    "step_extracting_audio": "Extracting audio from the file...",
    "step_splitting_audio": "Splitting audio into chunks...",
    "step_transcribing_audio": "Transcribing audio chunks...",
    "step_generating_images": "Generating images for each transcription...",
    "step_assembling_video": "Assembling the final video...",
    "step_generating_prompts": "Generating prompts for each transcription...",
    "step_generating_images": "Generating images for each transcription...",
    "generated_prompts": "Generated Prompts",
    "cut_at_silence_or_phrase" : "Split at silence (or at paragraphs) ?",
    "allow_prompt_editing": "Allow editing of generated prompts",
    "edit_prompts_instruction": "Edit the generated prompts below. You can modify them to better suit your needs.",
    "continue_processing": "Continue processing",
    "process_podcast_one_click": "Process in one click",
    "process_podcast_step_by_step": "Process with prompt editing",
    "scan_and_assemble" : "Only scan working dir and assemble video",
    "split_method": "Split method",
    "split_by_whisper" : "Whisper (medium)",
    "split_by_silence" : "Silence (many images)",
    "split_by_phrase": "Phrase (less images)",
    "summurize_transcript": "Prompt to summurize transcript to give context for image preprompt",
    "regenerate_prompts": "Regenerate Prompts",
})

translations["fr"].update({
    "podcasttovideo": "Podcast en Vidéo",
    "select_podcast": "Sélectionnez un fichier podcast (audio ou vidéo)",
    "process_podcast": "Traiter le Podcast",
    "processing": "Traitement en cours...",
    "video_generated": "Vidéo générée avec succès !",
    "error_processing": "Erreur lors du traitement : ",
    "step_saving_file": "Enregistrement du fichier téléchargé...",
    "step_extracting_audio": "Extraction de l'audio du fichier...",
    "step_splitting_audio": "Découpage de l'audio en segments...",
    "step_transcribing_audio": "Transcription des segments audio...",
    "step_generating_images": "Génération des images pour chaque transcription...",
    "step_assembling_video": "Assemblage de la vidéo finale...",
    "step_generating_prompts": "Génération des prompts pour chaque transcription...",
    "step_generating_images": "Génération des images pour chaque transcription...",
    "generated_prompts": "Prompts générés",
    "cut_at_silence_or_phrase" : "Découper aux silences (ou bien aux paragraphes) ?",
    "allow_prompt_editing": "Permettre la modification des prompts générés",
    "edit_prompts_instruction": "Modifiez les prompts générés ci-dessous. Vous pouvez les adapter pour mieux répondre à vos besoins.",
    "continue_processing": "Continuer le traitement",
    "process_podcast_one_click": "Procéder en un seul click",
    "process_podcast_step_by_step": "Editer les prompts puis procéder",
    "scan_and_assemble" : "Scaner le répertoire et assembler la vidéo seulement",
    "split_method": "Méthode de découpage",
    "split_by_whisper" : "Whisper (intermédiaire)",
    "split_by_silence" : "Silence (beaucoup d'images)",
    "split_by_phrase": "Phrase (moins d'images)",
    "summurize_transcript": "Prompt pour résumer le transcript de contexte de génération de preprompt d'image",
    "regenerate_prompts": "Régénérer les prompts",
})

class PodcasttovideoPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.transcript_plugin = TranscriptPlugin("transcript", plugin_manager)
        self.articlevideo_plugin = ArticletovideoPlugin("articletovideo", plugin_manager)
        self.imggen_plugin = ImggenPlugin("imggen", plugin_manager)
        self.ragllm_plugin = RagllmPlugin("ragllm", plugin_manager)

    def get_config_fields(self):
        return {
            "output_dir": {
                "type": "text",
                "label": t("output_directory"),
                "default": "~/Videos/PodcastToVideo"
            },
            "min_silence_len": {
                "type": "number",
                "label": "Minimum silence length (ms)",
                "default": 500
            },
            "silence_thresh": {
                "type": "number",
                "label": "Silence threshold (dB)",
                "default": -40
            },
            "image_prompt": {
                "type": "text",
                "label": t("image_prompt"),
                "default": "Generate an image depicting : {text} In the global context of this script : {resume}\n",
            },
            "summurize_transcript" : {
                "type" : "text",
                "label" : t("summurize_transcript"),
                "default" : "Summurize in one sentence the theme, do not comment : {transcript}. Do what you are told and nothing else."
            }
        }

    def get_tabs(self):
        return [{"name": t("podcasttovideo"), "plugin": "podcasttovideo"}]

    def extract_audio(self, input_file, output_file):
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            output_file
        ]
        subprocess.run(ffmpeg_command, check=True)

    def split_audio(self, audio_file, min_silence_len, silence_thresh):
        audio = AudioSegment.from_wav(audio_file)
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        return chunks

    def transcribe_chunks(self, chunks, config):
        transcriptions = []
        progress_bar = st.progress(0)
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                chunk.export(temp_audio.name, format="wav")
                transcript = self.transcript_plugin.transcribe_video(
                    temp_audio.name,
                    "txt",
                    config['transcript']['whisper_path'],
                    config['transcript']['whisper_model'],
                    config['transcript']['ffmpeg_path'],
                    config['common']['language']
                )
                transcriptions.append(transcript)
                os.unlink(temp_audio.name)
                progress_bar.progress((i + 1) / total_chunks)
        return transcriptions

    def split_audio_files_by_silence(self, audio_file, output_dir, min_silence_len, silence_thresh):
        chunks = self.split_audio(
            audio_file,
            min_silence_len,
            silence_thresh
        )
        audio_paths = []
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            audio_paths.append(chunk_path)
        return chunks, audio_paths

    def split_audio_files_by_phrase(self, config, audio_file, output_dir, mode="sentence"):
        # Alignement de l'audio avec le texte découpé
        sentence_timestamps = self.align_audio_with_text(audio_file)

        print(sentence_timestamps)

        # Découpage de l'audio à partir des timestamps
        audio = AudioSegment.from_wav(audio_file)
        audio_paths = []
        chunks = []
        for i, (start, end) in enumerate(sentence_timestamps):
            chunk = audio[start:end]
            chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            audio_paths.append(chunk_path)
            chunks.append(chunk)

        return chunks, audio_paths

    def transcribe_audio(self, audio_file, config):
        transcript = self.transcript_plugin.transcribe_video(
            audio_file,
            "srt",
            config['transcript']['whisper_path'],
            config['transcript']['whisper_model'],
            config['transcript']['ffmpeg_path'],
            config['common']['language']
        )
        return transcript

    def split_text_into_sentences(self, transcript):
        nlp = spacy.load("en_core_web_sm")  # Ou "fr_core_news_sm" pour le français
        doc = nlp(transcript)
        sentences = [sent for sent in doc.sents]
        return sentences

    def align_audio_with_text(self, audio_file):
        # Utilisation de pyannote.audio pour l'alignement
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=hf_token)

        # Générer un alignement temporel des segments audio
        diarization = pipeline({"uri": "audio", "audio": audio_file})

        sentence_timestamps = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = turn.start * 1000  # Convertir en millisecondes
            end = turn.end * 1000  # Convertir en millisecondes
            sentence_timestamps.append((start, end))

        return sentence_timestamps

    def split_audio_files_whisper(self, config, audio_file, output_dir):
        # Transcribe the audio using Whisper
        transcript = self.transcribe_audio(audio_file, config)
        print("-----------------------------")
        # Parse the transcript to extract timestamps and text
        lines = transcript.strip().split('\n')
        segments = []
        i = 0
        current_text = ""
        start_ms = None
        while i < len(lines):
            # Vérifie si la ligne est un numéro indiquant le début d'un nouveau segment
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
                print(f" {start_ms} : {current_text}")
                current_text = ""
                start_ms = None

            i += 1

        # Ajouter le dernier segment s'il reste du texte non traité
        if current_text:
            segments.append((start_ms, end_ms, current_text.strip()))

        # Split the audio based on the timestamps
        audio = AudioSegment.from_wav(audio_file)
        chunks = []
        audio_paths = []
        for i, (start, end, _) in enumerate(segments):
            chunk = audio[start:end]
            chunks.append(chunk)
            chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            audio_paths.append(chunk_path)

        return chunks, audio_paths

    def time_to_ms(self, time_str):
        h, m, s = time_str.split(':')
        s, ms = s.split('.')
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    def scan_and_assemble(self):
        output_dir = os.path.expanduser(config['podcasttovideo']['output_dir'])
        audio_files = self.articlevideo_plugin.get_sorted_files(output_dir, r'chunk_(\d+)\.wav')
        image_files = self.articlevideo_plugin.get_sorted_files(output_dir, r'image_(\d+)\.png')
        output_path = os.path.join(output_dir, "final_video.mp4")
        self.articlevideo_plugin.assemble_final_video(
            config,
            use_zoom_and_transitions=use_zoom_and_transitions,
            audio_paths=audio_files,
            image_paths=image_files,
            output_path=output_path
        )

    def run(self, config):
        st.header(t("podcasttovideo"))
        output_dir = os.path.expanduser(config['podcasttovideo']['output_dir'])
        os.makedirs(output_dir, exist_ok=True)
        uploaded_file = st.file_uploader(t("select_podcast"), type=["mp3", "wav", "mp4", "avi", "mov", "mkv"])
        use_zoom_and_transitions = st.checkbox(t("use_zoom_and_transitions"), value=True, key="use_zoom_transitions_checkbox")
        split_method = st.selectbox(
            t("split_method"),
            ["whisper", "silence", "phrase"],
            format_func=lambda x: {
                "whisper": t("split_by_whisper"),
                "silence": t("split_by_silence"),
                "phrase": t("split_by_phrase"),
            }[x]
        )
        if st.button(t("process_podcast_one_click")):
            self.process_podcast_one_click(uploaded_file, config, use_zoom_and_transitions, split_method)

        if st.button(t("scan_and_assemble")):
            self.scan_and_assemble()

        if st.button(t('process_podcast_step_by_step')):
            st.session_state.edited_prompts = None
            st.session_state.prompts = None
            st.session_state.transcriptions = None
            if uploaded_file is None:
                st.warning(t("please_upload_file"))
                return

            with st.spinner(t("processing")):
                input_file, audio_file, chunks, audio_paths = self.prepare_audio(config, uploaded_file, output_dir, split_method)
                st.session_state.transcriptions = self.transcribe_chunks(chunks, config)
                st.session_state.prompts = self.generate_prompts(st.session_state.transcriptions, config)
                st.session_state.chunks = chunks
                st.session_state.audio_paths = audio_paths

        # st.session_state.edited_prompts =  self.edit_prompts(st.session_state.prompts)
        # if 'edited_prompts' in st.session_state:
        #     if st.button(t("continue_processing")):
        #         with st.spinner(t("generating_images")):
        #             st.session_state.image_paths = self.generate_images_from_prompts(st.session_state.edited_prompts, config)
        #         with st.spinner(t("assembling_video")):
        #             self.assemble_video(config, use_zoom_and_transitions, st.session_state.audio_paths, st.session_state.image_paths, output_dir)
        #             st.success(t("video_generated"))
        if 'prompts' in st.session_state and 'transcriptions' in st.session_state:
            edited_prompts = self.edit_prompts(st.session_state.prompts, st.session_state.transcriptions)
            if edited_prompts is None and st.session_state.get('regenerate_prompts', False):
                st.session_state.prompts = self.generate_prompts(st.session_state.transcriptions, config)
                st.session_state.regenerate_prompts = False
                st.rerun()
            elif edited_prompts is not None:
                with st.spinner(t("generating_images")):
                    st.session_state.image_paths = self.generate_images_from_prompts(edited_prompts, config)
                with st.spinner(t("assembling_video")):
                    self.assemble_video(config, use_zoom_and_transitions, st.session_state.audio_paths, st.session_state.image_paths, output_dir)
                    st.success(t("video_generated"))

    def process_podcast_one_click(self, uploaded_file, config, use_zoom_and_transitions, split_method):
        if uploaded_file is None:
            st.warning(t("please_upload_file"))
            return

        with st.spinner(t("processing")):
            try:
                input_file, audio_file, chunks, audio_paths = self.prepare_audio(config, uploaded_file, os.path.expanduser(config['podcasttovideo']['output_dir']), split_method)
                transcriptions = self.transcribe_chunks(chunks, config)
                prompts = self.generate_prompts(transcriptions, config)
                image_paths = self.generate_images_from_prompts(prompts, config)
                self.assemble_video(config, use_zoom_and_transitions, audio_paths, image_paths, os.path.expanduser(config['podcasttovideo']['output_dir']))
                st.success(t("video_generated"))
            except Exception as e:
                self.handle_error(e)
            finally:
                self.cleanup(input_file, audio_file)

    def prepare_audio(self, config, uploaded_file, output_dir, split_method):
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
            chunks, audio_paths = self.split_audio_files_by_silence(audio_file, output_dir, min_silence_len, silence_thresh)
        elif split_method == "phrase":
            chunks, audio_paths = self.split_audio_files_by_phrase(config, audio_file, output_dir, mode="sentence")
        else:  # whisper
            chunks, audio_paths = self.split_audio_files_whisper(config, audio_file, output_dir)

        return input_file, audio_file, chunks, audio_paths

    def generate_prompts(self, transcriptions, config):
        st.info(t('step_generating_prompts'))
        prompts = []
        transcript = " ".join(transcriptions)
        resume = self.ragllm_plugin.call_llm(
            config['podcasttovideo']['summurize_transcript'].format(transcript=transcript),
            config['articletovideo']['image_sysprompt']
        )
        st.info(f"Context for the prompt : {resume}")
        progress_bar = st.progress(0)
        total_prompts = len(transcriptions)
        for i, text in enumerate(transcriptions):
            context = config['podcasttovideo']['image_prompt'].format(text=text, resume=resume)
            #print(context)
            prompt = self.articlevideo_plugin.generate_image_prompt(context, config['articletovideo']['image_sysprompt'])
            prompts.append(prompt)
            progress_bar.progress((i + 1) / total_prompts)
        return prompts

    # def edit_prompts(self, prompts):
    #     st.write(t("edit_prompts_instruction"))
    #     combined_prompts = "\n\n".join(prompts)
    #     edited_prompts = st.text_area(t("edit_prompts"), value=combined_prompts, height=400)
    #     return [prompt.strip() for prompt in edited_prompts.split("\n\n") if prompt.strip()]

    def edit_prompts(self, prompts, transcriptions):
        st.write(t("edit_prompts_instruction"))

        edited_prompts = []
        for i, (prompt, transcription) in enumerate(zip(prompts, transcriptions)):
            st.subheader(f"Prompt {i+1}")
            #st.text_area(f"Transcription {i+1}", value=transcription, height=100, key=f"transcription_{i}", disabled=True)
            st.write(transcription)
            edited_prompt = st.text_area(f"Edit prompt {i+1}", value=prompt, height=200, key=f"prompt_{i}")
            edited_prompts.append(edited_prompt)

        if st.button(t("regenerate_prompts")):
            st.session_state.regenerate_prompts = True
            return None  # This will trigger prompt regeneration

        if st.button(t("continue_processing")):
            return edited_prompts

        return None  # Return None if neither button is pressed

    def generate_images_from_prompts(self, prompts, config):
        st.info(t('step_generating_images'))
        self.articlevideo_plugin.unload_ollama_model()
        image_paths = []
        progress_bar = st.progress(0)
        total_prompts = len(prompts)
        for i, prompt in enumerate(prompts):
            output_dir = os.path.expanduser(config['podcasttovideo']['output_dir'])
            image_path = os.path.join(output_dir, f"image_{i}.png")
            self.articlevideo_plugin.generate_image(prompt, image_path)
            image_paths.append(image_path)
            progress_bar.progress((i + 1) / total_prompts)
        cols = st.columns(3)
        for i, image_path in enumerate(image_paths):
            with cols[i % 3]:
                st.image(image_path)
        return image_paths

    def assemble_video(self, config, use_zoom_and_transitions, audio_paths, image_paths, output_dir):
        st.info(t('step_assembling_video'))
        output_path = os.path.join(output_dir, "final_video.mp4")
        self.articlevideo_plugin.assemble_final_video(
            config,
            use_zoom_and_transitions=use_zoom_and_transitions,
            audio_paths=audio_paths,
            image_paths=image_paths,
            output_path=output_path
        )

    def handle_error(self, e):
        error_details = traceback.format_exc()
        st.error(f"{t('error_processing')}\n{str(e)}\n\nDetails:\n{error_details}")

    def cleanup(self, input_file, audio_file):
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(audio_file):
            os.unlink(audio_file)
