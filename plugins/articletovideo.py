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

# Add translations for this plugin
translations["en"].update({
    "article_to_video": "Article to Video",
    "article_url": "Article URL",
    "retrieve": "Retrieve",
    "article_text": "Article Text",
    "translate": "Translate",
    "translated_text": "Translated Text",
    "generate_video": "Generate Video",
    "processing": "Processing...",
    "video_generated": "Video generated successfully!",
    "split_by": "Split text by",
    "sentence": "Sentence",
    "paragraph": "Paragraph",
    "edit_prompts": "Générating image prompts",
    "generate_video": "Generate Video",
    "translating": "Translating...",
    "generating_prompts": "Generating image prompts...",
    "generating_audio": "Generating audio...",
    "generating_images": "Generating images...",
    "creating_video": "Creating final video...",
})
translations["fr"].update({
    "article_to_video": "Article vers Vidéo",
    "article_url": "URL de l'article",
    "retrieve": "Récupérer",
    "article_text": "Texte de l'article",
    "translate": "Traduire",
    "translated_text": "Texte traduit",
    "generate_video": "Générer la vidéo",
    "processing": "Traitement en cours...",
    "video_generated": "Vidéo générée avec succès !",
    "split_by": "Découper le texte par",
    "sentence": "Phrase",
    "paragraph": "Paragraphe",
    "edit_prompts": "Générer les prompts d'image",
    "generate_video": "Générer la vidéo",
    "translating": "Traduction en cours...",
    "generating_prompts": "Génération des prompts d'image...",
    "generating_audio": "Génération de l'audio...",
    "generating_images": "Génération des images...",
    "creating_video": "Création de la vidéo finale...",
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
        }

    def get_tabs(self):
        return [{"name": t("article_to_video"), "plugin": "articletovideo"}]

    def run(self, config):
        st.header(t("article_to_video"))

        url = st.text_input(t("article_url"))
        if st.button(t("retrieve")):
            article_text = self.retrieve_article(url)
            st.session_state.article_text = article_text

        if 'article_text' in st.session_state:
            article_text = st.text_area(t("article_text"), st.session_state.article_text, height=300)

            if st.button(t("translate")):
                translated_text = self.translate_text(article_text, config['articletovideo']['target_language'])
                st.session_state.translated_text = translated_text

        if 'translated_text' in st.session_state:
            translated_text = st.text_area(t("translated_text"), st.session_state.translated_text, height=300)

            if st.button(t("edit_prompts")):
                self.generate_and_edit_prompts(translated_text, config)

        if 'edited_prompts' in st.session_state:
            if st.button(t("generate_video")):
                self.generate_video(translated_text, st.session_state.edited_prompts, config)

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

        prompts_text = "\n".join(prompts)
        edited_prompts = st.text_area(t("edit_prompts"), prompts_text, height=300)
        st.session_state.edited_prompts = edited_prompts.split("\n")
        return edited_prompts

    def generate_video(self, text, prompts, config):
        split_by = config['articletovideo']['split_by']
        segments = self.split_text(text, split_by)
        output_dir = os.path.expanduser(config['articletovideo']['output_dir'])
        os.makedirs(output_dir, exist_ok=True)

        # Générer les audios
        with st.spinner(t("generating_audio")):
            progress_bar = st.progress(0)
            audio_paths = []
            for i, segment in enumerate(segments):
                audio_path = os.path.join(output_dir, f"audio_{i}.wav")
                self.generate_audio(segment, audio_path, config['articletovideo']['tts_speaker'], config['articletovideo']['target_language'])
                audio_paths.append(audio_path)
                progress_bar.progress((i + 1) / len(segments))

        # Générer les images
        with st.spinner(t("generating_images")):
            progress_bar = st.progress(0)
            image_paths = []
            for i, prompt in enumerate(prompts):
                image_path = os.path.join(output_dir, f"image_{i}.png")
                self.generate_image(prompt, image_path)
                image_paths.append(image_path)
                progress_bar.progress((i + 1) / len(prompts))

        # Créer la vidéo finale
        with st.spinner(t("creating_video")):
            self.create_final_video(audio_paths, image_paths, output_dir)

        st.success(t("video_generated"))

    def generate_audio(self, text, output_path, speaker, lang='fr'):
        if self.tts_model is None:
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts")

        self.tts_model.tts_to_file(text=text, file_path=output_path, speaker_wav=os.path.expanduser(speaker), language=lang)

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
        sysprompt = "You are an AI assistant tasked with creating image prompts. Given a paragraph of text, create a detailed image prompt of exactly 77 words that captures the essence of the paragraph. The prompt should be vivid and descriptive, suitable for image generation."
        prompt = f"Create an image prompt, 77 word max, based on this paragraph: {text}"
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        return ragllm_plugin.call_llm(prompt, sysprompt)

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

    def create_final_video(self, audio_paths, image_paths, output_dir):
        video_clips = []
        for audio_path, image_path in zip(audio_paths, image_paths):
            audio_clip = AudioFileClip(audio_path)
            image_clip = ImageClip(image_path).set_duration(audio_clip.duration)

            # Ajouter un zoom progressif
            zoomed_clip = image_clip.resize(lambda t: 1 + 0.01*t)

            video_clip = zoomed_clip.set_audio(audio_clip)
            video_clips.append(video_clip)

        # Ajouter des transitions
        final_clips = []
        for i, clip in enumerate(video_clips):
            if i > 0:
                final_clips.append(CompositeVideoClip([clip.crossfadein(1)]))
            else:
                final_clips.append(clip)

        final_video = concatenate_videoclips(final_clips, method="compose")
        output_path = os.path.join(output_dir, "final_video.mp4")
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)

        st.video(output_path)
# Don't forget to register this plugin in your main application
