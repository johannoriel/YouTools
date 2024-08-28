import streamlit as st
from app import Plugin
from global_vars import t, translations
import requests
from bs4 import BeautifulSoup
from transformers import MarianMTModel, MarianTokenizer
import torch
from TTS.api import TTS
from plugins.imggen import ImggenPlugin
from plugins.ragllm import RagllmPlugin
import os
import subprocess
from moviepy.editor import *
from PIL import Image
import io
import glob
import re
from num2words import num2words
from datetime import datetime
import babel.numbers
import locale
import soundfile as sf

# Add translations for this plugin
translations["en"].update({
    "article_to_video": "Article to Video",
    "article_url": "Article URL",
    "retrieve": "Retrieve",
    "article_text": "Article Text",
    "translate": "Translate",
    "translated_text": "Translated Text",
    "generate_video": "Generate audio and image sequence",
    "processing": "Processing...",
    "video_generated": "Video generated successfully!",
    "split_by": "Split text by",
    "sentence": "Sentence",
    "paragraph": "Paragraph",
    "edit_prompts": "Générating image prompts",
    "translating": "Translating...",
    "generating_prompts": "Generating image prompts...",
    "generating_audio": "Generating audio...",
    "generating_images": "Generating images...",
    "creating_video": "Creating final video...",
    "zoom_factor": "Zoom Factor",
    "use_zoom_and_transitions": "Use Zoom and Transitions",
    "edit_images_and_prompts": "Edit Images and Prompts",
    "assemble_video": "Assemble Video",
    "image_prompt": "Image Prompt",
    "translated_text": "Translated Text",
    "regenerate_image": "Regenerate Image",
    "upload_image": "Upload Image",
    "generate_video_from_url": "Generate Video in One Click",
    "show_detailed_steps": "Show Detailed Steps",
    "processing_video": "Processing video...",
    "scan_files": "Scan Files",
    "assemble_video": "Assemble Video",
    "no_files_found": "No audio or image files found in the output directory.",
    "files_scanned_successfully": "Files scanned successfully!",
    "scanned_files": "Scanned Files",
    "video_files_generated": "Video files generated successfully. You can now assemble the video.",
    "regenerate_audio": "Regenerate Audio",
    "regenerate_images": "Regenerate Images",
    "audio_regenerated": "Audio files regenerated successfully!",
    "images_regenerated": "Image files regenerated successfully!",
    "regenerating_audio": "Regenerating audio files...",
    "regenerating_images": "Regenerating image files...",
    "percent": "percent",
    "decimal_point": "point",
    "decimal_comma": "comma",
    "tts_model": "TTS Model",
    "convert_numbers": "Convert numbers to words",
})
translations["fr"].update({
    "article_to_video": "Article vers Vidéo",
    "article_url": "URL de l'article",
    "retrieve": "Récupérer",
    "article_text": "Texte de l'article",
    "translate": "Traduire",
    "translated_text": "Texte traduit",
    "generate_video": "Générer l'audio et la séquence d'images",
    "processing": "Traitement en cours...",
    "video_generated": "Vidéo générée avec succès !",
    "split_by": "Découper le texte par",
    "sentence": "Phrase",
    "paragraph": "Paragraphe",
    "edit_prompts": "Générer les prompts d'image",
    "translating": "Traduction en cours...",
    "generating_prompts": "Génération des prompts d'image...",
    "generating_audio": "Génération de l'audio...",
    "generating_images": "Génération des images...",
    "creating_video": "Création de la vidéo finale...",
    "zoom_factor": "Facteur de Zoom",
    "use_zoom_and_transitions": "Utiliser le Zoom et les Transitions",
    "edit_images_and_prompts": "Éditer les Images et les Prompts",
    "assemble_video": "Assembler la Vidéo",
    "image_prompt": "Prompt de l'Image",
    "translated_text": "Texte Traduit",
    "regenerate_image": "Régénérer l'Image",
    "upload_image": "Télécharger une Image",
    "generate_video_from_url": "Générer une Vidéo en UN Click",
    "show_detailed_steps": "Afficher les Étapes Détaillées",
    "processing_video": "Traitement de la vidéo en cours...",
    "scan_files": "Scanner les Fichiers",
    "assemble_video": "Assembler la Vidéo",
    "no_files_found": "Aucun fichier audio ou image trouvé dans le répertoire de sortie.",
    "files_scanned_successfully": "Fichiers scannés avec succès !",
    "scanned_files": "Fichiers Scannés",
    "video_files_generated": "Fichiers vidéo générés avec succès. Vous pouvez maintenant assembler la vidéo.",
    "regenerate_audio": "Régénérer l'Audio",
    "regenerate_images": "Régénérer les Images",
    "audio_regenerated": "Fichiers audio régénérés avec succès !",
    "images_regenerated": "Fichiers image régénérés avec succès !",
    "regenerating_audio": "Régénération des fichiers audio en cours...",
    "regenerating_images": "Régénération des fichiers image en cours...",
    "percent": "pourcent",
    "decimal_point": "point",
    "decimal_comma": "virgule",
    "tts_model": "Modèle TTS",
    "convert_numbers": "Convertir les nombres en mots"
})

class ArticletovideoPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.translator = None
        self.tts_model = None
        self.imggen_plugin = ImggenPlugin("imggen", plugin_manager)
        self.ragllm_plugin = RagllmPlugin("ragllm", plugin_manager)

    def get_config_fields(self):
        return {
            "output_dir": {
                "type": "text",
                "label": t("output_directory"),
                "default": "~/Videos/ArticleToVideo"
            },
            "target_language": {
                "type": "select",
                "label": t("target_language"),
                "options": [("fr-fr", "Français"), ("en", "English")],
                "default": "fr-fr"
            },
            "tts_speaker": {
                "type": "text",
                "label": t("tts_speaker"),
                "default": "p326"  # Default speaker for YourTTS
            },
            "split_by": {
                "type": "select",
                "label": t("split_by"),
                "options": [("sentence", t("sentence")), ("paragraph", t("paragraph"))],
                "default": "paragraph"
            },
            "zoom_factor": {
                "type": "number",
                "label": t("zoom_factor"),
                "default": 0.01,
                "min": 0,
                "max": 0.1,
                "step": 0.001
            },
            "tts_model": {
                "type": "select",
                "label": t("tts_model"),
                "options": [
                    ("your_tts", "YourTTS"),
                    ("tacotron", "Tacotron"),
                    ("xtts_v2", "XTTS2"),
                    ("bark", "Bark"),
                ],
                "default": "your_tts"
            },
            "convert_numbers": {
                "type": "checkbox",
                "label": t("convert_numbers"),
                "default": True
            },
        }

    def get_tabs(self):
        return [{"name": t("article_to_video"), "plugin": "articletovideo"}]

    def run(self, config):
        st.header(t("article_to_video"))
        url = st.text_input(t("article_url"), key="url")

        show_detailed_steps = st.checkbox(t("show_detailed_steps"), value=False, key="show_detailed_steps_checkbox")
        if show_detailed_steps:
            self.show_detailed_interface(config)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(t("generate_video_from_url"), key="generate_video_button"):
                self.generate_video_one_click(url, config)

        with col2:
            if st.button(t("scan_files"), key="scan_files_button"):
                self.scan_files(config)

        with col3:
            if st.button(t("assemble_video"), key="assemble_video_button1"):
                self.assemble_final_video(config, use_zoom_and_transitions=True)

        use_zoom_and_transitions = st.checkbox(t("use_zoom_and_transitions"), value=True, key="use_zoom_transitions_checkbox")

    def get_sorted_files(self, directory, pattern):
        files = []
        for filename in os.listdir(directory):
            match = re.match(pattern, filename)
            if match:
                index = int(match.group(1))
                files.append((index, os.path.join(directory, filename)))
        return [file for _, file in sorted(files)]

    def scan_files(self, config):
        output_dir = os.path.expanduser(config['articletovideo']['output_dir'])

        # Scan for audio files
        audio_files = self.get_sorted_files(output_dir, r'audio_(\d+)\.wav')

        # Scan for image files
        image_files = self.get_sorted_files(output_dir, r'image_(\d+)\.png')

        if not audio_files or not image_files:
            st.warning(t("no_files_found"))
            return

        st.session_state.audio_paths = audio_files
        st.session_state.image_paths = image_files

        # Reconstruct segments and prompts based on file count
        st.session_state.segments = [f"Segment {i+1}" for i in range(len(audio_files))]
        st.session_state.prompts = [f"Prompt for image {i+1}" for i in range(len(image_files))]

        st.success(t("files_scanned_successfully"))

        # Display scanned files
        st.subheader(t("scanned_files"))
        st.write(f"Audio files: {len(audio_files)}")
        st.write(f"Image files: {len(image_files)}")

    def generate_video_one_click(self, url, config):
        with st.spinner(t("processing_video")):
            # Step 1: Retrieve article
            article_text = self.retrieve_article(url)
            st.session_state.article_text = article_text

            # Step 2: Translate text
            translated_text = self.translate_text(article_text, config['articletovideo']['target_language'])
            st.session_state.translated_text = translated_text

            # Step 3: Generate prompts
            split_by = config['articletovideo']['split_by']
            segments = self.get_segments(translated_text, split_by)
            prompts = self.generate_all_image_prompts(segments)
            st.session_state.edited_prompts = prompts

            # Step 4: Generate video
            self.generate_all_images(translated_text, prompts, config)

        st.success(t("video_files_generated"))

    def show_detailed_interface(self, config):

        if st.button(t("retrieve"), key="retrieve_button"):
            article_text = self.retrieve_article(st.session_state.get('url', ''))
            st.session_state.article_text = article_text

        if 'article_text' in st.session_state:
            article_text = st.text_area(t("article_text"), st.session_state.article_text, height=300, key="article_text_area")

            if st.button(t("translate"), key="translate_button"):
                translated_text = self.translate_text(article_text, config['articletovideo']['target_language'], config)
                st.session_state.translated_text = translated_text

        if 'translated_text' in st.session_state:
            translated_text = st.text_area(t("translated_text"), st.session_state.translated_text, height=300, key="translated_text_area")

            if st.button(t("edit_prompts"), key="edit_prompts_button"):
                st.session_state.edited_prompts = self.generate_and_edit_prompts(translated_text, config)

        if 'edited_prompts' in st.session_state:
            self.display_prompts(st.session_state.edited_prompts)
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(t("generate_video"), key="generate_video_button_detailed"):
                    self.generate_all_images(st.session_state.translated_text, st.session_state.edited_prompts, config)
            with col2:
                if st.button(t("regenerate_audio"), key="regenerate_audio_button"):
                    self.regenerate_audios(st.session_state.translated_text, config)
            with col3:
                if st.button(t("regenerate_images"), key="regenerate_images_button"):
                    self.regenerate_images(st.session_state.edited_prompts, config)

        if 'image_paths' in st.session_state and 'prompts' in st.session_state and 'segments' in st.session_state:
            st.header(t("edit_images_and_prompts"))
            self.display_image_gallery()

            if st.button(t("assemble_video"), key="assemble_video_button"):
                self.assemble_final_video(config, use_zoom_and_transitions)

    def get_segments(self, text, split_by):
        if split_by == "sentence":
            segments = text.split(". ")
        else:
            segments = text.split("\n\n")

        # Filtrer les segments
        filtered_segments = []
        for segment in segments:
            if segment is None:
                continue
            segment = segment.strip()
            if not segment or ' ' not in segment:
                continue
            filtered_segments.append(segment)

        return filtered_segments

    def regenerate_audios(self, text, config):
        split_by = config['articletovideo']['split_by']
        segments = self.get_segments(text, split_by)
        output_dir = os.path.expanduser(config['articletovideo']['output_dir'])

        with st.spinner(t("regenerating_audio")):
            progress_bar = st.progress(0)
            audio_paths = []
            for i, segment in enumerate(segments):
                audio_path = os.path.join(output_dir, f"audio_{i}.wav")
                self.generate_one_audio(segment, audio_path, config)
                audio_paths.append(audio_path)
                progress_bar.progress((i + 1) / len(segments))

        st.session_state.audio_paths = audio_paths
        st.success(t("audio_regenerated"))

    def regenerate_images(self, prompts, config):
        output_dir = os.path.expanduser(config['articletovideo']['output_dir'])

        with st.spinner(t("regenerating_images")):
            progress_bar = st.progress(0)
            image_paths = []
            for i, prompt in enumerate(prompts):
                image_path = os.path.join(output_dir, f"image_{i}.png")
                self.generate_image(prompt, image_path)
                image_paths.append(image_path)
                progress_bar.progress((i + 1) / len(prompts))

        st.session_state.image_paths = image_paths
        st.success(t("images_regenerated"))

    def convert_numbers_to_words_llm(self, text, lang):
        sysprompt = f"You are an AI assistant that converts numbers including real numbers and percentages to words in {lang}. Maintain the original sentence structure and only convert the numbers and real numbers and percentages. DO NOT ADD COMMENTS"
        prompt = f"Without adding any comment, convert all numbers and real numbers and percentages in the following text to words in {lang}:\n\n{text}"
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        result = ragllm_plugin.call_llm(prompt, sysprompt)
        return result

    def convert_numbers_to_words(self, text, lang):
        def convert_number(match):
            number = match.group(0)

            # Convert percentages
            if number == '%':
                return t("percent")

            if '%' in number:
                num = float(number.replace('%', ''))
                return num2words(num, lang=lang) + " " + t("percent")

            # Convert dates
            if re.match(r'\d{1,2}/\d{1,2}/\d{4}', number):
                date = datetime.strptime(number, "%d/%m/%Y")
                locale.setlocale(locale.LC_TIME, lang)
                return date.strftime("%d %B %Y").lower()

            # Convert decimal numbers
            if '.' in number or ',' in number:
                number = number.replace(',', '.')  # Normalize to dot as decimal separator
                integer_part, decimal_part = number.split('.')
                integer_words = num2words(int(integer_part), lang=lang)
                decimal_words = ' '.join(num2words(int(digit), lang=lang) for digit in decimal_part)
                decimal_separator = t("decimal_point") if lang.startswith('en') else t("decimal_comma")
                return f"{integer_words} {decimal_separator} {decimal_words}"

            # Convert integers
            return num2words(int(number), lang=lang)

        # Regular expression to match numbers, percentages, and dates
        pattern = r'\b\d+(?:[.,]\d+)?%?|%|\b\d{1,2}/\d{1,2}/\d{4}\b'

        return re.sub(pattern, convert_number, text)

        return re.sub(pattern, convert_number, text)

    def retrieve_article(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return '\n\n'.join([p.text for p in paragraphs])

    def translate_text(self, text, target_lang, config):
        if self.translator is None:
            model_name = f'Helsinki-NLP/opus-mt-en-{target_lang[:2]}'
            self.translator = MarianMTModel.from_pretrained(model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)

        paragraphs = text.split('\n\n')
        translated_paragraphs = []
        progress_bar = st.progress(0)

        print("Traductions")
        with st.spinner(t("processing")):
            for i, paragraph in enumerate(paragraphs):
                print('--------------------------------')
                sentences = paragraph.split('. ')
                translated_sentences = []

                for sentence in sentences:
                    if not sentence.strip():
                        continue

                    inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
                    translated = self.translator.generate(**inputs)
                    translated_sentence = self.tokenizer.decode(translated[0], skip_special_tokens=True)
                    # Convert numbers to words after translation
                    if config['articletovideo']['convert_numbers']:
                        translated_sentence = self.convert_numbers_to_words(translated_sentence, target_lang)
                    truncated_sentence = re.split(r'[.\n]', translated_sentence, maxsplit=1)[0]
                    translated_sentences.append(truncated_sentence)
                    print(f"\033[1m\033[31m{sentence}\033[0m")
                    print(translated_sentence)


                translated_paragraph = '. '.join(translated_sentences)
                translated_paragraphs.append(translated_paragraph)
                progress_bar.progress((i + 1) / len(paragraphs))

        return '\n\n'.join(translated_paragraphs)

    def translate_text_withllm(self, text, target_lang):
        paragraphs = text.split('\n\n')
        translated_paragraphs = []
        progress_bar = st.progress(0)

        with st.spinner(t("translating")):
            for i, paragraph in enumerate(paragraphs):
                sysprompt = f"You are a professional translator. Translate the given text from English to {target_lang}. Maintain the original tone and style. Convert all numbers, including real numbers and percentages, to words in the target language. DO NOT ADD COMMENTS OR EXPLANATIONS."
                prompt = f"Translate the following text to {target_lang}, converting all numbers and real numbers and percent sign to words, do not add any comment:\n\n{paragraph}"

                ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
                translated_paragraph = ragllm_plugin.call_llm(prompt, sysprompt)

                truncated_paragraph = translated_paragraph.split('\n\n', maxsplit=1)[0]
                translated_paragraphs.append(truncated_paragraph)
                progress_bar.progress((i + 1) / len(paragraphs))

        return '\n\n'.join(translated_paragraphs)

    def generate_and_edit_prompts(self, text, config):
        split_by = config['articletovideo']['split_by']
        segments = self.get_segments(text, split_by)

        with st.spinner(t("generating_prompts")):
            prompts = self.generate_all_image_prompts(segments)
            return prompts

    def display_prompts(self, prompts):
        prompts_text = "\n\n".join(prompts)
        edited_prompts = st.text_area(t("edit_prompts"), prompts_text, height=300, key="edit_prompts_text_area")
        st.session_state.edited_prompts = edited_prompts.split("\n\n")

    def generate_all_images(self, text, prompts, config):
        split_by = config['articletovideo']['split_by']
        segments = self.get_segments(text, split_by)
        output_dir = os.path.expanduser(config['articletovideo']['output_dir'])
        os.makedirs(output_dir, exist_ok=True)

        # Generate audio
        with st.spinner(t("generating_audio")):
            progress_bar = st.progress(0)
            audio_paths = []
            for i, segment in enumerate(segments):
                audio_path = os.path.join(output_dir, f"audio_{i}.wav")
                self.generate_one_audio(segment, audio_path, config)
                audio_paths.append(audio_path)
                progress_bar.progress((i + 1) / len(segments))

        self.unload_ollama_model()
        # Generate images
        with st.spinner(t("generating_images")):
            progress_bar = st.progress(0)
            image_paths = []
            for i, prompt in enumerate(prompts):
                image_path = os.path.join(output_dir, f"image_{i}.png")
                self.generate_image(prompt, image_path)
                image_paths.append(image_path)
                progress_bar.progress((i + 1) / len(prompts))

        st.session_state.audio_paths = audio_paths
        st.session_state.image_paths = image_paths
        st.session_state.prompts = prompts
        st.session_state.segments = segments

    def generate_one_audio(self, text, output_path, config):

        # https://theroamingworkshop.cloud/b/en/2425/%F0%9F%90%B8coqui-ai-tts-ultra-fast-voice-generation-and-cloning-from-multilingual-text/
        tts_model = config['articletovideo']['tts_model']
        target_lang = config['articletovideo']['target_language']

        os.environ['COQUI_TTS_CACHE_DIR'] = os.path.expanduser('~/.cache/coqui') # https://github.com/coqui-ai/TTS/issues/3608

        if tts_model == "your_tts": # fast but not best quality
            if self.tts_model is None:
                self.tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts")
            self.tts_model.tts_to_file(text=text, file_path=output_path, speaker_wav=os.path.expanduser(config['articletovideo']['tts_speaker']), language=target_lang)
        elif tts_model == "xtts_v2": # best quality but very slow
            if self.tts_model is None:
                self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self.tts_model.tts_to_file(text=text, file_path=output_path, speaker_wav=os.path.expanduser(config['articletovideo']['tts_speaker']), language=target_lang[:2])
        elif tts_model == "tacotron": # not very good quality and predermined voice
            if self.tts_model is None:
                self.tts_model = TTS("tts_models/fr/mai/tacotron2-DDC")
            self.tts_model.tts_to_file(text=text, file_path=output_path, speaker_wav=os.path.expanduser(config['articletovideo']['tts_speaker']))
        elif tts_model == "bark":
            if self.tts_model is None:
                self.tts_model = TTS("tts_models/multilingual/multi-dataset/bark")# https://github.com/coqui-ai/TTS/issues/3567
            self.tts_model.tts_to_file(text=text, file_path=output_path, speaker_wav=os.path.expanduser(config['articletovideo']['tts_speaker']), language=target_lang)
        else:
            raise ValueError(f"Unsupported TTS model: {tts_model}")


    def split_text(self, text, split_by):
        if split_by == "sentence":
            return text.split(". ")
        else:
            return text.split("\n\n")

    def generate_all_image_prompts(self, paragraphs):
        progress_bar = st.progress(0)
        prompts = []
        for i, paragraph in enumerate(paragraphs):
            prompt = self.generate_image_prompt(paragraph)
            prompts.append(prompt)
            progress_bar.progress((i + 1) / len(paragraphs))
        return prompts

    def unload_ollama_model(self):
        # Effectuer un appel bidon à Ollama pour décharger le modèle
        self.ragllm_plugin.free_llm()

    def generate_image_prompt(self, text):
        sysprompt = "You are an AI assistant tasked with creating image prompts. The prompt should be vivid and descriptive, suitable for image generation. Reply only the prompt."
        prompt = f"Create an image prompt, 77 word max, based on this paragraph. Describe, without commenting, just give the prompt for a scene that illustrate the following situation : {text}"
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        result = ragllm_plugin.call_llm(prompt, sysprompt)
        return result.replace("\n", " ")

    def generate_image(self, prompt, output_path):
        image, _ = self.imggen_plugin.generate_image(
            None,
            prompt,
            aspect_ratio="16:9",
            remove_background=False,
            background_removal_method="ai",
            seed=None,
            face=False,
            steps=2
        )
        image.save(output_path)

    def display_image_gallery(self):
        image_paths = st.session_state.image_paths
        prompts = st.session_state.prompts
        segments = st.session_state.segments

        cols = st.columns(3)
        for i, (image_path, prompt, segment) in enumerate(zip(image_paths, prompts, segments)):
            with cols[i % 3]:
                st.image(image_path)
                new_prompt = st.text_area(f"{t('image_prompt')} {i+1}", prompt, key=f"prompt_{i}")
                #st.text_area(f"{t('translated_text')} {i+1}", segment, key=f"segment_{i}")
                st.markdown(f"""
                    <div style="border: 1px solid #e6e6e6; padding: 10px; width: 100%; height: 100px; overflow-y: scroll;">
                        {segment}
                    </div>
                    """, unsafe_allow_html=True)


                if st.button(t("regenerate_image"), key=f"regenerate_{i}"):
                    self.regenerate_image(i, new_prompt)
                    st.rerun()

                uploaded_file = st.file_uploader(t("upload_image"), type=["png", "jpg", "jpeg"], key=f"upload_{i}")
                if uploaded_file is not None and st.session_state.image_paths[i] != uploaded_file.name:
                    self.replace_image(i, uploaded_file)
                    st.rerun()

    def regenerate_image(self, index, new_prompt):
        output_dir = os.path.dirname(st.session_state.image_paths[index])
        new_image_path = os.path.join(output_dir, f"image_{index}_new.png")
        self.generate_image(new_prompt, new_image_path)
        st.session_state.image_paths[index] = new_image_path
        st.session_state.prompts[index] = new_prompt

    def replace_image(self, index, uploaded_file):
        img = Image.open(uploaded_file)
        img = img.convert('RGB')
        img = img.resize((1280, 720), Image.LANCZOS)  # Resize to 16:9 aspect ratio
        output_dir = os.path.dirname(st.session_state.image_paths[index])
        new_image_path = os.path.join(output_dir, f"image_{index}_uploaded.png")
        img.save(new_image_path)
        st.session_state.image_paths[index] = new_image_path

    def assemble_final_video(self, config, use_zoom_and_transitions):
        audio_paths = st.session_state.audio_paths
        image_paths = st.session_state.image_paths
        output_dir = os.path.expanduser(config['articletovideo']['output_dir'])

        with st.spinner(t("creating_video")):
            video_clips = []
            for audio_path, image_path in zip(audio_paths, image_paths):
                audio_clip = AudioFileClip(audio_path)
                image_clip = ImageClip(image_path).set_duration(audio_clip.duration)

                if use_zoom_and_transitions:
                    zoom_factor = float(config['articletovideo']['zoom_factor'])
                    zoomed_clip = image_clip.resize(lambda t: 1 + zoom_factor * t)
                    video_clip = zoomed_clip.set_audio(audio_clip)
                else:
                    video_clip = image_clip.set_audio(audio_clip)

                video_clips.append(video_clip)

            if use_zoom_and_transitions:
                final_clips = []
                for i, clip in enumerate(video_clips):
                    if i > 0:
                        final_clips.append(CompositeVideoClip([clip.crossfadein(1)]))
                    else:
                        final_clips.append(clip)
            else:
                final_clips = video_clips

            print("Concatenating video")
            final_video = concatenate_videoclips(final_clips, method="compose")
            output_path = os.path.join(output_dir, "final_video.mp4")
            print("Writing final video to file")

            # Create a progress container
            progress_container = st.empty()

            # Create a custom logger
            logger = StreamlitProgressBarLogger(progress_container)

            # Write the video file with the custom logger
            final_video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=24,
                #logger=logger
            )

        st.success(t("video_generated"))
        st.video(output_path)

from proglog import ProgressBarLogger
class StreamlitProgressBarLogger(ProgressBarLogger):
    def __init__(self, progress_container):
        super().__init__()
        self.progress_container = progress_container

    def callback(self, **changes):
        if 'index' in changes and 'total' in changes:
            progress = min(changes['index'] / changes['total'], 1.0)
            self.progress_container.progress(progress)
