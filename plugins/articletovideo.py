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
    "generate_video_from_url": "Generate Video from URL",
    "show_detailed_steps": "Show Detailed Steps",
    "processing_video": "Processing video...",
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
    "generate_video_from_url": "Générer une Vidéo depuis l'URL",
    "show_detailed_steps": "Afficher les Étapes Détaillées",
    "processing_video": "Traitement de la vidéo en cours...",
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
        }

    def get_tabs(self):
        return [{"name": t("article_to_video"), "plugin": "articletovideo"}]

    def run(self, config):
        st.header(t("article_to_video"))

        url = st.text_input(t("article_url"), key="article_url_input")

        if st.button(t("generate_video_from_url"), key="generate_video_button"):
            self.generate_video_from_url(url, config)

        show_detailed_steps = st.checkbox(t("show_detailed_steps"), value=False, key="show_detailed_steps_checkbox")

        if show_detailed_steps:
            self.show_detailed_interface(config)

    def generate_video_from_url(self, url, config):
        with st.spinner(t("processing_video")):
            # Step 1: Retrieve article
            article_text = self.retrieve_article(url)
            st.session_state.article_text = article_text

            # Step 2: Translate text
            translated_text = self.translate_text(article_text, config['articletovideo']['target_language'])
            st.session_state.translated_text = translated_text

            # Step 3: Generate prompts
            split_by = config['articletovideo']['split_by']
            segments = self.split_text(translated_text, split_by)
            prompts = self.generate_all_image_prompts(segments)
            st.session_state.edited_prompts = prompts

            # Step 4: Generate video
            self.generate_video(translated_text, prompts, config)

            # Step 5: Assemble final video
            use_zoom_and_transitions = True  # You can make this configurable if needed
            self.assemble_final_video(config, use_zoom_and_transitions)

        st.success(t("video_generated"))
        st.video(os.path.join(os.path.expanduser(config['articletovideo']['output_dir']), "final_video.mp4"))

    def show_detailed_interface(self, config):
        use_zoom_and_transitions = st.checkbox(t("use_zoom_and_transitions"), value=True, key="use_zoom_transitions_checkbox")

        if st.button(t("retrieve"), key="retrieve_button"):
            article_text = self.retrieve_article(st.session_state.get('url', ''))
            st.session_state.article_text = article_text

        if 'article_text' in st.session_state:
            article_text = st.text_area(t("article_text"), st.session_state.article_text, height=300, key="article_text_area")

            if st.button(t("translate"), key="translate_button"):
                translated_text = self.translate_text(article_text, config['articletovideo']['target_language'])
                st.session_state.translated_text = translated_text

        if 'translated_text' in st.session_state:
            translated_text = st.text_area(t("translated_text"), st.session_state.translated_text, height=300, key="translated_text_area")

            if st.button(t("edit_prompts"), key="edit_prompts_button"):
                self.generate_and_edit_prompts(translated_text, config)

        if 'edited_prompts' in st.session_state:
            self.display_prompts(st.session_state.edited_prompts)
            if st.button(t("generate_video"), key="generate_video_button_detailed"):
                self.generate_video(st.session_state.translated_text, st.session_state.edited_prompts, config)

        if 'image_paths' in st.session_state and 'prompts' in st.session_state and 'segments' in st.session_state:
            st.header(t("edit_images_and_prompts"))
            self.display_image_gallery()

            if st.button(t("assemble_video"), key="assemble_video_button"):
                self.assemble_final_video(config, use_zoom_and_transitions)

    def convert_numbers_to_words(self, text, lang):
        sysprompt = f"You are an AI assistant that converts numbers including real numbers and percentages to words in {lang}. Maintain the original sentence structure and only convert the numbers and real numbers and percentages. DO NOT ADD COMMENTS"
        prompt = f"Without comment, convert all numbers and real numbers and percentages in the following text to words in {lang}:\n\n{text}"
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        result = ragllm_plugin.call_llm(prompt, sysprompt)
        return result

    def retrieve_article(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return '\n\n'.join([p.text for p in paragraphs])

    def translate_text(self, text, target_lang):
        if self.translator is None:
            model_name = f'Helsinki-NLP/opus-mt-en-{target_lang[:2]}'
            self.translator = MarianMTModel.from_pretrained(model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)

        paragraphs = text.split('\n\n')
        translated_paragraphs = []
        progress_bar = st.progress(0)

        with st.spinner(t("processing")):
            for i, paragraph in enumerate(paragraphs):
                sentences = paragraph.split('. ')
                translated_sentences = []

                for sentence in sentences:
                    inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
                    translated = self.translator.generate(**inputs)
                    translated_sentence = self.tokenizer.decode(translated[0], skip_special_tokens=True)
                    translated_sentences.append(translated_sentence)

                translated_paragraph = '. '.join(translated_sentences)
                translated_paragraphs.append(translated_paragraph)
                progress_bar.progress((i + 1) / len(paragraphs))

        return '\n\n'.join(translated_paragraphs)

    def generate_and_edit_prompts(self, text, config):
        split_by = config['articletovideo']['split_by']
        segments = self.split_text(text, split_by)

        with st.spinner(t("generating_prompts")):
            prompts = self.generate_all_image_prompts(segments)

    def display_prompts(self, prompts):
        prompts_text = "\n\n".join(prompts)
        edited_prompts = st.text_area(t("edit_prompts"), prompts_text, height=300, key="edit_prompts_text_area")
        st.session_state.edited_prompts = edited_prompts.split("\n\n")

    def generate_video(self, text, prompts, config):
        split_by = config['articletovideo']['split_by']
        segments = self.split_text(text, split_by)
        output_dir = os.path.expanduser(config['articletovideo']['output_dir'])
        os.makedirs(output_dir, exist_ok=True)

        # Generate audio
        with st.spinner(t("generating_audio")):
            progress_bar = st.progress(0)
            audio_paths = []
            for i, segment in enumerate(segments):
                audio_path = os.path.join(output_dir, f"audio_{i}.wav")
                self.generate_audio(segment, audio_path, config['articletovideo']['tts_speaker'], config['articletovideo']['target_language'])
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

    def generate_audio(self, text, output_path, speaker, lang='fr'):
        if self.tts_model is None:
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts")
        # Convert numbers and percentages to words
        text_with_words = self.convert_numbers_to_words(text, lang)

        self.tts_model.tts_to_file(text=text_with_words, file_path=output_path, speaker_wav=os.path.expanduser(speaker), language=lang)

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
        sysprompt = "You are an AI assistant tasked with creating image prompts. The prompt should be vivid and descriptive, suitable for image generation."
        prompt = f"Create an image prompt, 77 word max, based on this paragraph. Describe a scene that illustrate the following situation : {text}"
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

            final_video = concatenate_videoclips(final_clips, method="compose")
            output_path = os.path.join(output_dir, "final_video.mp4")
            final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)

        st.success(t("video_generated"))
        st.video(output_path)
# Don't forget to register this plugin in your main application
