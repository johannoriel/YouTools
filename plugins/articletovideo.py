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
import random
import math
import cv2

# Add translations for this plugin
translations["en"].update(
    {
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
        "speed_factor": "Speed Factor",
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
        "image_sysprompt": "Prompt système pour la génération d'images",
    }
)
translations["fr"].update(
    {
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
        "speed_factor": "Facteur de vitesse",
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
        "convert_numbers": "Convertir les nombres en mots",
        "image_sysprompt": "Prompt système pour la génération d'images",
    }
)


class ArticletovideoPlugin(Plugin):
    """
    A plugin that converts articles to videos, including translation,
    text-to-speech (TTS), and image generation. The plugin supports both
    detailed step-by-step processing and one-click video generation.

    Parameters
    ----------
    name : str
        The name of the plugin.
    plugin_manager : PluginManager
        The plugin manager that handles plugin lifecycle and dependencies.

    Attributes
    ----------
    translator : MarianMTModel or None
        The machine translation model used to translate article text.
    tts_model : TTS or None
        The text-to-speech model used to generate audio from text.
    imggen_plugin : ImggenPlugin
        An instance of `ImggenPlugin` for generating images.
    ragllm_plugin : RagllmPlugin
        An instance of `RagllmPlugin` for calling LLM (Language Models) for prompts.

    Methods
    -------
    get_config_fields()
        Returns a dictionary defining the configurable fields for the plugin.

    get_tabs()
        Returns the tabs for the plugin interface.

    run(config)
        Main entry point for running the plugin within the Streamlit interface.

    generate_video_one_click(url, config)
        Generates a video from an article URL using one-click mode.

    scan_files(config)
        Scans the output directory for generated audio and image files.

    assemble_final_video(config, use_zoom_and_transitions, audio_paths, image_paths, output_path)
        Assembles the final video from the generated audio and image files.

    Examples
    --------
    # Example usage within a Streamlit app:
    plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
    config = plugin.get_config_fields()
    plugin.run(config)
    """

    def __init__(self, name, plugin_manager):
        """
        Initializes the ArticletovideoPlugin with a name and plugin manager.

        This constructor sets up the plugin by initializing its name and plugin manager,
        as well as creating instances of the `ImggenPlugin` and `RagllmPlugin` for
        image generation and language model interactions, respectively.

        Parameters
        ----------
        name : str
            The name of the plugin, used for identification within the application.
        plugin_manager : PluginManager
            The plugin manager responsible for managing the lifecycle and dependencies
            of the plugin within the application.

        Attributes
        ----------
        translator : MarianMTModel or None
            Initialized as `None`. This will hold the translation model used to translate
            article text when needed.
        tts_model : TTS or None
            Initialized as `None`. This will hold the text-to-speech (TTS) model used to
            generate audio from text.
        imggen_plugin : ImggenPlugin
            An instance of `ImggenPlugin`, used to generate images based on text prompts.
        ragllm_plugin : RagllmPlugin
            An instance of `RagllmPlugin`, used to interact with language models for
            generating text prompts and other LLM-related tasks.

        Examples
        --------
        plugin_manager = PluginManager()
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        """
        super().__init__(name, plugin_manager)
        self.translator = None
        self.tts_model = None
        self.imggen_plugin = ImggenPlugin("imggen", plugin_manager)
        self.ragllm_plugin = RagllmPlugin("ragllm", plugin_manager)

    def get_config_fields(self):
        """
        Returns a dictionary defining the configurable fields for the plugin.

        Returns
        -------
        dict
            A dictionary where keys are field names and values are dictionaries
            specifying field type, label, default values, and other options.
        """
        return {
            "output_dir": {
                "type": "text",
                "label": t("output_directory"),
                "default": "~/Videos/ArticleToVideo",
            },
            "target_language": {
                "type": "select",
                "label": t("target_language"),
                "options": [("fr-fr", "Français"), ("en", "English")],
                "default": "fr-fr",
            },
            "tts_speaker": {
                "type": "text",
                "label": t("tts_speaker"),
                "default": "p326",  # Default speaker for YourTTS
            },
            "split_by": {
                "type": "select",
                "label": t("split_by"),
                "options": [("sentence", t("sentence")), ("paragraph", t("paragraph"))],
                "default": "paragraph",
            },
            "zoom_factor": {
                "type": "number",
                "label": t("zoom_factor"),
                "default": 0.01,
                "min": 0,
                "max": 0.2,
                "step": 0.001,
            },
            "speed_factor": {
                "type": "number",
                "label": t("speed_factor"),
                "default": 0.01,
                "min": 0,
                "max": 0.2,
                "step": 0.001,
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
                "default": "your_tts",
            },
            "convert_numbers": {
                "type": "checkbox",
                "label": t("convert_numbers"),
                "default": True,
            },
            "image_sysprompt": {
                "type": "text",
                "label": t("image_sysprompt"),
                "default": "You are an AI assistant tasked with creating image prompts. The prompt should be vivid and descriptive, suitable for image generation. Reply only the prompt.",
            },
        }

    def get_tabs(self):
        return [{"name": t("article_to_video"), "plugin": "articletovideo"}]

    def run(self, config):
        """
        Executes the main logic of the plugin within the Streamlit interface.

        This method serves as the entry point for the plugin's functionality
        when used in a Streamlit application. It displays the main interface
        for the user, allowing them to input an article URL, choose options
        for video generation, and either generate the video in one click or
        follow a detailed step-by-step process.

        Parameters
        ----------
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin. This includes settings like the output directory,
            target language, TTS model, zoom factor, and others.

        Raises
        ------
        ValueError
            If the configuration is missing required fields or contains invalid values.
        requests.exceptions.RequestException
            If there is an issue retrieving the article from the provided URL.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        config = plugin.get_config_fields()
        plugin.run(config)
        """
        st.header(t("article_to_video"))
        url = st.text_input(t("article_url"), key="url")

        show_detailed_steps = st.checkbox(
            t("show_detailed_steps"), value=False, key="show_detailed_steps_checkbox"
        )
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
            if st.button(t("assemble_video"), key="assemble_video_button"):
                audio_paths = st.session_state.audio_paths
                image_paths = st.session_state.image_paths
                output_dir = os.path.expanduser(config["articletovideo"]["output_dir"])
                self.assemble_final_video(
                    config,
                    use_zoom_and_transitions=True,
                    audio_paths=audio_paths,
                    image_paths=image_paths,
                    output_path=os.path.join(output_dir, "final_video.mp4"),
                )

        use_zoom_and_transitions = st.checkbox(
            t("use_zoom_and_transitions"),
            value=True,
            key="use_zoom_transitions_checkbox",
        )

    def get_sorted_files(self, directory, pattern):
        """
        Retrieves and sorts files from a directory based on a specified pattern.

        This method searches for files in the given directory that match the
        provided regular expression pattern, sorts them by a numeric index,
        and returns a list of file paths.

        Parameters
        ----------
        directory : str
            The directory where files are located.
        pattern : str
            A regular expression pattern used to identify and extract numeric
            indices from filenames.

        Returns
        -------
        list of str
            A list of sorted file paths that match the specified pattern.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        files = plugin.get_sorted_files("~/output", r"image_(\d+)\.png")
        """
        files = []
        for filename in os.listdir(directory):
            match = re.match(pattern, filename)
            if match:
                index = int(match.group(1))
                files.append((index, os.path.join(directory, filename)))
        return [file for _, file in sorted(files)]

    def scan_files(self, config):
        """
        Scans the output directory for generated audio and image files.

        This method identifies and sorts the audio and image files based on
        their naming convention in the specified output directory.

        Parameters
        ----------
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin, including the output directory.

        Raises
        ------
        FileNotFoundError
            If the specified output directory does not exist.
        Warning
            If no audio or image files are found in the output directory.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        config = plugin.get_config_fields()
        plugin.scan_files(config)
        """
        output_dir = os.path.expanduser(config["articletovideo"]["output_dir"])

        # Scan for audio files
        audio_files = self.get_sorted_files(output_dir, r"audio_(\d+)\.wav")

        # Scan for image files
        image_files = self.get_sorted_files(output_dir, r"image_(\d+)\.png")

        if not audio_files or not image_files:
            st.warning(t("no_files_found"))
            return

        st.session_state.audio_paths = audio_files
        st.session_state.image_paths = image_files

        # Reconstruct segments and prompts based on file count
        st.session_state.segments = [f"Segment {i+1}" for i in range(len(audio_files))]
        st.session_state.prompts = [
            f"Prompt for image {i+1}" for i in range(len(image_files))
        ]

        st.success(t("files_scanned_successfully"))

        # Display scanned files
        st.subheader(t("scanned_files"))
        st.write(f"Audio files: {len(audio_files)}")
        st.write(f"Image files: {len(image_files)}")

    def generate_video_one_click(self, url, config):
        """
        Generates a video from an article URL using one-click mode.

        This method retrieves the article, translates it, generates prompts,
        creates images, and assembles the final video.

        Parameters
        ----------
        url : str
            The URL of the article to be converted into a video.
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin, such as output directory, language, TTS model, etc.

        Raises
        ------
        ValueError
            If the URL is invalid or the article cannot be retrieved.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        config = plugin.get_config_fields()
        plugin.generate_video_one_click("https://example.com/article", config)
        """
        with st.spinner(t("processing_video")):
            # Step 1: Retrieve article
            article_text = self.retrieve_article(url)
            st.session_state.article_text = article_text

            # Step 2: Translate text
            translated_text = self.translate_text(
                article_text, config["articletovideo"]["target_language"]
            )
            st.session_state.translated_text = translated_text

            # Step 3: Generate prompts
            split_by = config["articletovideo"]["split_by"]
            segments = self.get_segments(translated_text, split_by)
            prompts = self.generate_all_image_prompts(
                segments, config["articletovideo"]["image_sysprompt"]
            )
            st.session_state.edited_prompts = prompts

            # Step 4: Generate video
            self.generate_all_images(translated_text, prompts, config)

        st.success(t("video_files_generated"))

    def show_detailed_interface(self, config):
        """
        Displays a detailed user interface for generating videos step by step.

        This method provides an interface in Streamlit that allows users to
        retrieve an article, translate it, generate image prompts, and
        assemble the final video in a detailed, step-by-step manner.

        Parameters
        ----------
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        plugin.show_detailed_interface(config)
        """
        if st.button(t("retrieve"), key="retrieve_button"):
            article_text = self.retrieve_article(st.session_state.get("url", ""))
            st.session_state.article_text = article_text

        if "article_text" in st.session_state:
            article_text = st.text_area(
                t("article_text"),
                st.session_state.article_text,
                height=300,
                key="article_text_area",
            )

            if st.button(t("translate"), key="translate_button"):
                translated_text = self.translate_text(
                    article_text, config["articletovideo"]["target_language"], config
                )
                st.session_state.translated_text = translated_text

        if "translated_text" in st.session_state:
            translated_text = st.text_area(
                t("translated_text"),
                st.session_state.translated_text,
                height=300,
                key="translated_text_area",
            )

            if st.button(t("edit_prompts"), key="edit_prompts_button"):
                st.session_state.edited_prompts = self.generate_and_edit_prompts(
                    translated_text, config
                )

        if "edited_prompts" in st.session_state:
            self.display_prompts(st.session_state.edited_prompts)
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(t("generate_video"), key="generate_video_button_detailed"):
                    self.generate_all_images(
                        st.session_state.translated_text,
                        st.session_state.edited_prompts,
                        config,
                    )
            with col2:
                if st.button(t("regenerate_audio"), key="regenerate_audio_button"):
                    self.regenerate_audios(st.session_state.translated_text, config)
            with col3:
                if st.button(t("regenerate_images"), key="regenerate_images_button"):
                    self.regenerate_images(st.session_state.edited_prompts, config)

        if (
            "image_paths" in st.session_state
            and "prompts" in st.session_state
            and "segments" in st.session_state
        ):
            st.header(t("edit_images_and_prompts"))
            self.display_image_gallery()

            if st.button(t("assemble_video"), key="assemble_video_button"):
                audio_paths = st.session_state.audio_paths
                image_paths = st.session_state.image_paths
                output_dir = os.path.expanduser(config["articletovideo"]["output_dir"])
                self.assemble_final_video(
                    config,
                    use_zoom_and_transitions=True,
                    audio_paths=audio_paths,
                    image_paths=image_paths,
                    output_path=os.path.join(output_dir, "final_video.mp4"),
                )

    def get_segments(self, text, split_by):
        """
        Splits the provided text into segments based on the specified criterion.

        This method splits the text either by sentences or paragraphs, based on
        the `split_by` parameter, and filters out empty or invalid segments.

        Parameters
        ----------
        text : str
            The text to split into segments.
        split_by : str
            The criterion for splitting the text, either "sentence" or "paragraph".

        Returns
        -------
        list of str
            A list of text segments after splitting.

        Raises
        ------
        ValueError
            If `split_by` is not "sentence" or "paragraph".

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        segments = plugin.get_segments("This is the first sentence. This is the second.", "sentence")
        """
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
            if not segment or " " not in segment:
                continue
            filtered_segments.append(segment)

        return filtered_segments

    def regenerate_audios(self, text, config):
        """
        Regenerates audio files from the given text using the configured TTS model.

        This method splits the text into segments, generates audio for each
        segment, and saves the audio files in the specified output directory.

        Parameters
        ----------
        text : str
            The text for which to regenerate audio.
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin, such as the TTS model and output directory.

        Raises
        ------
        FileNotFoundError
            If the output directory does not exist.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        plugin.regenerate_audios(translated_text, config)
        """
        split_by = config["articletovideo"]["split_by"]
        segments = self.get_segments(text, split_by)
        output_dir = os.path.expanduser(config["articletovideo"]["output_dir"])

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
        """
        Regenerates images based on the provided prompts.

        This method generates new images based on the provided prompts and
        saves them in the specified output directory.

        Parameters
        ----------
        prompts : list of str
            The image prompts to use for generating new images.
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin, such as the output directory.

        Raises
        ------
        FileNotFoundError
            If the output directory does not exist.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        plugin.regenerate_images(prompts, config)
        """
        output_dir = os.path.expanduser(config["articletovideo"]["output_dir"])

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
        """
        Converts numbers to words in the provided text using a language model.

        This method leverages an LLM (Language Model) to convert numbers,
        including percentages and dates, into words in the specified language.

        Parameters
        ----------
        text : str
            The text containing numbers to be converted to words.
        lang : str
            The language code indicating the target language for conversion (e.g., 'en' for English, 'fr' for French).

        Returns
        -------
        str
            The text with all numbers converted to words by the LLM.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        result = plugin.convert_numbers_to_words_llm("Le prix est 20,5 %.", "fr")
        """
        sysprompt = f"You are an AI assistant that converts numbers including real numbers and percentages to words in {lang}. Maintain the original sentence structure and only convert the numbers and real numbers and percentages. DO NOT ADD COMMENTS"
        prompt = f"Without adding any comment, convert all numbers and real numbers and percentages in the following text to words in {lang}:\n\n{text}"
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        result = ragllm_plugin.call_llm(prompt, sysprompt)
        return result

    def convert_numbers_to_words(self, text, lang):
        """
        Converts all numbers in the provided text to words in the specified language.

        This method handles the conversion of integers, decimal numbers, percentages,
        and dates to their word equivalents in the target language.

        Parameters
        ----------
        text : str
            The text containing numbers to be converted to words.
        lang : str
            The language code indicating the target language for conversion (e.g., 'en' for English, 'fr' for French).

        Returns
        -------
        str
            The text with all numbers converted to words.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        result = plugin.convert_numbers_to_words("The temperature is 25.5 degrees.", "en")
        """

        def convert_number(match):
            number = match.group(0)

            # Convert percentages
            if number == "%":
                return t("percent")

            if "%" in number:
                num = float(number.replace("%", ""))
                return num2words(num, lang=lang) + " " + t("percent")

            # Convert dates
            if re.match(r"\d{1,2}/\d{1,2}/\d{4}", number):
                date = datetime.strptime(number, "%d/%m/%Y")
                locale.setlocale(locale.LC_TIME, lang)
                return date.strftime("%d %B %Y").lower()

            # Convert decimal numbers
            if "." in number or "," in number:
                number = number.replace(
                    ",", "."
                )  # Normalize to dot as decimal separator
                integer_part, decimal_part = number.split(".")
                integer_words = num2words(int(integer_part), lang=lang)
                decimal_words = " ".join(
                    num2words(int(digit), lang=lang) for digit in decimal_part
                )
                decimal_separator = (
                    t("decimal_point") if lang.startswith("en") else t("decimal_comma")
                )
                return f"{integer_words} {decimal_separator} {decimal_words}"

            # Convert integers
            return num2words(int(number), lang=lang)

        # Regular expression to match numbers, percentages, and dates
        pattern = r"\b\d+(?:[.,]\d+)?%?|%|\b\d{1,2}/\d{1,2}/\d{4}\b"

        return re.sub(pattern, convert_number, text)

        return re.sub(pattern, convert_number, text)

    def retrieve_article(self, url):
        """
        Retrieves and extracts the text content of an article from a given URL.

        This method fetches the web page at the provided URL and extracts the
        text from the paragraph (`<p>`) tags.

        Parameters
        ----------
        url : str
            The URL of the article to retrieve.

        Returns
        -------
        str
            The extracted text content of the article.

        Raises
        ------
        requests.exceptions.RequestException
            If there is an issue with the HTTP request.
        ValueError
            If the URL is invalid or the article content cannot be retrieved.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        text = plugin.retrieve_article("https://example.com/article")
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n\n".join([p.text for p in paragraphs])

    def translate_text(self, text, target_lang, config):
        """
        Translates the given text into the target language using a MarianMT model.

        This method translates text by splitting it into paragraphs and sentences,
        processing each individually, and then reassembling the translated text.
        Optionally, numbers in the text can be converted to words.

        Parameters
        ----------
        text : str
            The text to translate.
        target_lang : str
            The target language code (e.g., 'fr-fr' for French).
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin, such as whether to convert numbers to words.

        Returns
        -------
        str
            The translated text.

        Raises
        ------
        ValueError
            If the target language is not supported or translation fails.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        translated_text = plugin.translate_text("Hello, world!", "fr-fr", config)
        """
        if self.translator is None:
            model_name = f"Helsinki-NLP/opus-mt-en-{target_lang[:2]}"
            self.translator = MarianMTModel.from_pretrained(model_name)
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)

        paragraphs = text.split("\n\n")
        translated_paragraphs = []
        progress_bar = st.progress(0)

        print("Traductions")
        with st.spinner(t("processing")):
            for i, paragraph in enumerate(paragraphs):
                print("--------------------------------")
                sentences = paragraph.split(". ")
                translated_sentences = []

                for sentence in sentences:
                    if not sentence.strip():
                        continue

                    inputs = self.tokenizer(
                        sentence, return_tensors="pt", padding=True, truncation=True
                    )
                    translated = self.translator.generate(**inputs)
                    translated_sentence = self.tokenizer.decode(
                        translated[0], skip_special_tokens=True
                    )
                    # Convert numbers to words after translation
                    if config["articletovideo"]["convert_numbers"]:
                        translated_sentence = self.convert_numbers_to_words(
                            translated_sentence, target_lang
                        )
                    truncated_sentence = re.split(
                        r"[.\n]", translated_sentence, maxsplit=1
                    )[0]
                    translated_sentences.append(truncated_sentence)
                    print(f"\033[1m\033[31m{sentence}\033[0m")
                    print(translated_sentence)

                translated_paragraph = ". ".join(translated_sentences)
                translated_paragraphs.append(translated_paragraph)
                progress_bar.progress((i + 1) / len(paragraphs))

        return "\n\n".join(translated_paragraphs)

    def translate_text_withllm(self, text, target_lang):
        """
        Translates text into the target language using a language model (LLM).

        This method translates the given text into the specified target language
        by leveraging a language model (LLM). The translation includes converting
        numbers and percentages to words in the target language. The text is
        split into paragraphs, each translated separately, and the resulting
        translated paragraphs are then combined.

        Parameters
        ----------
        text : str
            The text to be translated.
        target_lang : str
            The target language code (e.g., 'fr' for French, 'en' for English).

        Returns
        -------
        str
            The translated text, with numbers and percentages converted to words
            in the target language.

        Raises
        ------
        ValueError
            If the translation fails or the language model returns an invalid response.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        translated_text = plugin.translate_text_withllm("The price is 25.5%.", "fr")
        """
        paragraphs = text.split("\n\n")
        translated_paragraphs = []
        progress_bar = st.progress(0)

        with st.spinner(t("translating")):
            for i, paragraph in enumerate(paragraphs):
                sysprompt = f"You are a professional translator. Translate the given text from English to {target_lang}. Maintain the original tone and style. Convert all numbers, including real numbers and percentages, to words in the target language. DO NOT ADD COMMENTS OR EXPLANATIONS."
                prompt = f"Translate the following text to {target_lang}, converting all numbers and real numbers and percent sign to words, do not add any comment:\n\n{paragraph}"

                ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
                translated_paragraph = ragllm_plugin.call_llm(prompt, sysprompt)

                truncated_paragraph = translated_paragraph.split("\n\n", maxsplit=1)[0]
                translated_paragraphs.append(truncated_paragraph)
                progress_bar.progress((i + 1) / len(paragraphs))

        return "\n\n".join(translated_paragraphs)

    def generate_and_edit_prompts(self, text, config):
        """
        Generates and optionally edits image prompts based on the translated text.

        This method splits the text into segments, generates image prompts for
        each segment, and allows the user to edit these prompts.

        Parameters
        ----------
        text : str
            The translated text for which to generate image prompts.
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin, such as the system prompt for generating image prompts.

        Returns
        -------
        list of str
            A list of generated (and possibly edited) image prompts.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        prompts = plugin.generate_and_edit_prompts(translated_text, config)
        """
        split_by = config["articletovideo"]["split_by"]
        segments = self.get_segments(text, split_by)

        with st.spinner(t("generating_prompts")):
            prompts = self.generate_all_image_prompts(
                segments, config["articletovideo"]["image_sysprompt"]
            )
            return prompts

    def display_prompts(self, prompts):
        """
        Displays and allows editing of image prompts in the Streamlit interface.

        This method takes a list of image prompts, displays them in a text area
        within the Streamlit interface, and allows the user to edit the prompts.
        The edited prompts are then saved back into the session state for further
        processing.

        Parameters
        ----------
        prompts : list of str
            A list of image prompts to be displayed and edited. Each prompt corresponds
            to a specific segment of the text and is intended to guide the generation
            of an image.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        prompts = ["A sunny day at the beach.", "A crowded market in the city."]
        plugin.display_prompts(prompts)
        """
        prompts_text = "\n\n".join(prompts)
        edited_prompts = st.text_area(
            t("edit_prompts"), prompts_text, height=300, key="edit_prompts_text_area"
        )
        st.session_state.edited_prompts = edited_prompts.split("\n\n")

    def generate_all_images(self, text, prompts, config):
        """
        Generates all images and audio files for the provided text and prompts.

        This method processes the text by splitting it into segments, generates
        corresponding audio files using the configured TTS model, and creates
        images based on the provided prompts.

        Parameters
        ----------
        text : str
            The text to be used for generating audio and images.
        prompts : list of str
            The prompts to be used for generating images.
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin, such as the output directory, TTS model, and other parameters.

        Raises
        ------
        FileNotFoundError
            If the output directory does not exist.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        plugin.generate_all_images(translated_text, prompts, config)
        """
        split_by = config["articletovideo"]["split_by"]
        segments = self.get_segments(text, split_by)
        output_dir = os.path.expanduser(config["articletovideo"]["output_dir"])
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
        """
        Generates an audio file from the provided text using the configured TTS model.

        This method converts the provided text into speech and saves the audio file
        at the specified output path.

        Parameters
        ----------
        text : str
            The text to convert into speech.
        output_path : str
            The file path where the generated audio file will be saved.
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin, such as the TTS model and other parameters.

        Raises
        ------
        ValueError
            If the specified TTS model is not supported.
        FileNotFoundError
            If the output directory does not exist.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        plugin.generate_one_audio("Bonjour tout le monde.", "~/output/audio_0.wav", config)
        """
        # https://theroamingworkshop.cloud/b/en/2425/%F0%9F%90%B8coqui-ai-tts-ultra-fast-voice-generation-and-cloning-from-multilingual-text/
        tts_model = config["articletovideo"]["tts_model"]
        target_lang = config["articletovideo"]["target_language"]

        os.environ["COQUI_TTS_CACHE_DIR"] = os.path.expanduser(
            "~/.cache/coqui"
        )  # https://github.com/coqui-ai/TTS/issues/3608

        if tts_model == "your_tts":  # fast but not best quality
            if self.tts_model is None:
                self.tts_model = TTS("tts_models/multilingual/multi-dataset/your_tts")
            self.tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=os.path.expanduser(config["articletovideo"]["tts_speaker"]),
                language=target_lang,
            )
        elif tts_model == "xtts_v2":  # best quality but very slow
            if self.tts_model is None:
                self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self.tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=os.path.expanduser(config["articletovideo"]["tts_speaker"]),
                language=target_lang[:2],
            )
        elif tts_model == "tacotron":  # not very good quality and predermined voice
            if self.tts_model is None:
                self.tts_model = TTS("tts_models/fr/mai/tacotron2-DDC")
            self.tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=os.path.expanduser(config["articletovideo"]["tts_speaker"]),
            )
        elif tts_model == "bark":
            if self.tts_model is None:
                self.tts_model = TTS(
                    "tts_models/multilingual/multi-dataset/bark"
                )  # https://github.com/coqui-ai/TTS/issues/3567
            self.tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=os.path.expanduser(config["articletovideo"]["tts_speaker"]),
                language=target_lang,
            )
        else:
            raise ValueError(f"Unsupported TTS model: {tts_model}")

    def split_text(self, text, split_by):
        """
        Generates image prompts for each paragraph in the provided list.

        This method creates an image prompt for each paragraph by using a
        language model and a system prompt to guide the generation process.

        Parameters
        ----------
        paragraphs : list of str
            A list of paragraphs for which to generate image prompts.
        sysprompt : str
            The system prompt guiding the generation of the image prompts.

        Returns
        -------
        list of str
            A list of generated image prompts corresponding to each paragraph.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        prompts = plugin.generate_all_image_prompts(paragraphs, sysprompt)
        """
        if split_by == "sentence":
            return text.split(". ")
        else:
            return text.split("\n\n")

    def generate_all_image_prompts(self, paragraphs, sysprompt):
        """
        Generates image prompts for each paragraph in the provided list.

        This method iterates through a list of paragraphs and generates an image
        prompt for each one using a specified system prompt. The prompts are
        generated by calling a language model via the `RagllmPlugin`.

        Parameters
        ----------
        paragraphs : list of str
            A list of paragraphs for which to generate image prompts. Each paragraph
            represents a segment of the translated text.
        sysprompt : str
            The system prompt guiding the generation of the image prompts. This prompt
            provides context or instructions for the language model to produce relevant
            and descriptive image prompts.

        Returns
        -------
        list of str
            A list of generated image prompts corresponding to each paragraph in the
            provided list. Each prompt is a textual description intended for generating
            an image that visually represents the content of the paragraph.

        Raises
        ------
        ValueError
            If the `paragraphs` list is empty or if an error occurs during prompt generation.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        paragraphs = ["The sun rises over a tranquil beach.", "A bustling city street at night."]
        sysprompt = "You are an AI tasked with generating image prompts. Each prompt should vividly describe a scene."
        prompts = plugin.generate_all_image_prompts(paragraphs, sysprompt)
        """
        progress_bar = st.progress(0)
        prompts = []
        for i, paragraph in enumerate(paragraphs):
            prompt = self.generate_image_prompt(paragraph, sysprompt)
            prompts.append(prompt)
            progress_bar.progress((i + 1) / len(paragraphs))
        return prompts

    def unload_ollama_model(self):
        """
        Unloads the currently loaded Ollama model to free resources.

        This method makes a dummy call to unload the model from memory,
        preventing unnecessary resource consumption.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        plugin.unload_ollama_model()
        """
        # Effectuer un appel bidon à Ollama pour décharger le modèle
        self.ragllm_plugin.free_llm()

    def generate_image_prompt(self, text, sysprompt):
        """
        Generates an image prompt based on the provided text and system prompt.

        This method uses a language model to create an image prompt that describes
        the visual content corresponding to the given text.

        Parameters
        ----------
        text : str
            The text for which to generate an image prompt.
        sysprompt : str
            The system prompt guiding the generation of the image prompt.

        Returns
        -------
        str
            The generated image prompt.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        prompt = plugin.generate_image_prompt("A beautiful sunset over the ocean.", sysprompt)
        """
        prompt = f"Create an image prompt, 77 word max. Describe, without commenting, just give the prompt for a scene that illustrate the following situation : {text}"
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        result = ragllm_plugin.call_llm(prompt, sysprompt)
        return result.replace("\n", " ")

    def generate_image(self, prompt, output_path):
        """
        Generates an image from the provided prompt and saves it to the specified output path.

        This method uses the `ImggenPlugin` to create an image based on the given
        textual prompt. The generated image is saved as a PNG file at the specified path.

        Parameters
        ----------
        prompt : str
            The prompt describing the content of the image to generate.
        output_path : str
            The file path where the generated image will be saved, typically ending in `.png`.

        Raises
        ------
        FileNotFoundError
            If the specified directory in `output_path` does not exist.
        ValueError
            If the image generation process fails or returns an invalid result.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        plugin.generate_image("A beautiful sunrise over the mountains", "~/output/image_1.png")
        """
        image, _ = self.imggen_plugin.generate_image(
            None,
            prompt,
            aspect_ratio="16:9",
            remove_background=False,
            background_removal_method="ai",
            seed=None,
            face=False,
            steps=2,
        )
        image.save(output_path)

    def display_image_gallery(self):
        """
        Displays an image gallery in the Streamlit interface for editing and
        replacing images.

        This method allows users to view generated images, edit the associated
        prompts, and upload new images to replace the existing ones.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        plugin.display_image_gallery()
        """
        image_paths = st.session_state.image_paths
        prompts = st.session_state.prompts
        segments = st.session_state.segments

        cols = st.columns(3)
        for i, (image_path, prompt, segment) in enumerate(
            zip(image_paths, prompts, segments)
        ):
            with cols[i % 3]:
                st.image(image_path)
                new_prompt = st.text_area(
                    f"{t('image_prompt')} {i+1}", prompt, key=f"prompt_{i}"
                )
                # st.text_area(f"{t('translated_text')} {i+1}", segment, key=f"segment_{i}")
                st.markdown(
                    f"""
                    <div style="border: 1px solid #e6e6e6; padding: 10px; width: 100%; height: 100px; overflow-y: scroll;">
                        {segment}
                    """, unsafe_allow_html=True)

                if st.button(t("regenerate_image"), key=f"regenerate_{i}"):
                    self.regenerate_image(i, new_prompt)
                    st.rerun()

                uploaded_file = st.file_uploader(
                    t("upload_image"), type=["png", "jpg", "jpeg"], key=f"upload_{i}"
                )
                if (
                    uploaded_file is not None
                    and st.session_state.image_paths[i] != uploaded_file.name
                ):
                    self.replace_image(i, uploaded_file)
                    st.rerun()

    def regenerate_image(self, index, new_prompt):
        """
        Regenerates a single image based on a new prompt provided by the user.

        This method generates a new image using the updated prompt and
        replaces the existing image with the new one.

        Parameters
        ----------
        index : int
            The index of the image to be regenerated in the list of image paths.
        new_prompt : str
            The new prompt to use for generating the image.

        Raises
        ------
        FileNotFoundError
            If the output directory does not exist.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        plugin.regenerate_image(1, "A serene beach at sunset.")
        """
        output_dir = os.path.dirname(st.session_state.image_paths[index])
        new_image_path = os.path.join(output_dir, f"image_{index}_new.png")
        self.generate_image(new_prompt, new_image_path)
        st.session_state.image_paths[index] = new_image_path
        st.session_state.prompts[index] = new_prompt

    def replace_image(self, index, uploaded_file):
        """
        Replaces an existing image with a new one uploaded by the user.

        This method resizes the uploaded image to match the aspect ratio
        of 16:9, converts it to RGB, and saves it as a PNG file.

        Parameters
        ----------
        index : int
            The index of the image to be replaced in the list of image paths.
        uploaded_file : UploadedFile
            The new image file uploaded by the user.

        Raises
        ------
        FileNotFoundError
            If the output directory does not exist.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        plugin.replace_image(2, uploaded_file)
        """
        img = Image.open(uploaded_file)
        img = img.convert("RGB")
        img = img.resize((1280, 720), Image.LANCZOS)  # Resize to 16:9 aspect ratio
        output_dir = os.path.dirname(st.session_state.image_paths[index])
        new_image_path = os.path.join(output_dir, f"image_{index}_uploaded.png")
        img.save(new_image_path)
        st.session_state.image_paths[index] = new_image_path

    def assemble_final_video(
        self, config, use_zoom_and_transitions, audio_paths, image_paths, output_path
    ):
        """
        Assembles the final video from the generated audio and image files.

        This method combines the audio and image files into a video, optionally
        adding zoom and transition effects.

        Parameters
        ----------
        config : dict
            The configuration dictionary containing user-defined settings for
            the plugin, such as zoom factor, speed factor, etc.
        use_zoom_and_transitions : bool
            Whether to apply zoom and transition effects to the video.
        audio_paths : list of str
            A list of file paths to the generated audio files.
        image_paths : list of str
            A list of file paths to the generated image files.
        output_path : str
            The file path where the final video will be saved.

        Raises
        ------
        ValueError
            If the number of audio files does not match the number of image files.

        Examples
        --------
        plugin = ArticletovideoPlugin("article_to_video", plugin_manager)
        config = plugin.get_config_fields()
        plugin.assemble_final_video(config, True, audio_paths, image_paths, "~/output/final_video.mp4")
        """
        with st.spinner(t("creating_video")):
            video_clips = []
            for audio_path, image_path in zip(audio_paths, image_paths):
                audio_clip = AudioFileClip(audio_path)
                image_clip = ImageClip(image_path).set_duration(audio_clip.duration)

                if use_zoom_and_transitions:
                    zoom_factor = float(config['articletovideo']['zoom_factor'])
                    speed_factor = float(config['articletovideo']['speed_factor'])
                    initial_x_shift = random.uniform(1, 100)
                    initial_y_shift = random.uniform(1, 100)
                    #zoomed_clip = image_clip.resize(lambda t: 1 + zoom_factor * t) # simple zoom
                    def dynamic_zoom(get_frame, t):
                        # Create a smooth, periodic motion for zoom
                        zoom = 1 + zoom_factor * (1 + math.sin(t * math.pi / 10)) / 2

                        # Generate wandering motion
                        x_shift = math.sin((t + initial_x_shift) / 3.0) * speed_factor
                        y_shift = math.cos((t + initial_y_shift) / 2.4) * speed_factor

                        # Create the transformation matrix
                        center_x, center_y = 0.5 + x_shift, 0.5 + y_shift
                        matrix = cv2.getRotationMatrix2D(
                            (center_x * image_clip.w, center_y * image_clip.h), 0, zoom
                        )
                        matrix[0, 2] += (0.5 - center_x) * image_clip.w
                        matrix[1, 2] += (0.5 - center_y) * image_clip.h

                        frame = get_frame(t)
                        return cv2.warpAffine(
                            frame,
                            matrix,
                            (frame.shape[1], frame.shape[0]),
                            borderMode=cv2.BORDER_REFLECT_101,
                        )

                    zoomed_clip = image_clip.fl(dynamic_zoom)
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
                # logger=logger
            )

        st.success(t("video_generated"))
        st.video(output_path)


from proglog import ProgressBarLogger


class StreamlitProgressBarLogger(ProgressBarLogger):
    """
    A custom logger for displaying progress in Streamlit during video rendering.

    This class extends the `ProgressBarLogger` to update a Streamlit progress bar
    container in real-time as the video rendering progresses.

    Parameters
    ----------
    progress_container : Streamlit
        A Streamlit container object used to display the progress bar.

    Methods
    -------
    callback(**changes)
        Updates the progress bar based on the current progress index and total tasks.
    """

    def __init__(self, progress_container):
        """
        Initializes the StreamlitProgressBarLogger with a progress container.

        Parameters
        ----------
        progress_container : Streamlit
            A Streamlit container object used to display the progress bar.
        """
        super().__init__()
        self.progress_container = progress_container

    def callback(self, **changes):
        """
        Updates the progress bar based on the current progress index and total tasks.

        Parameters
        ----------
        **changes : dict
            Keyword arguments containing the current progress index and total tasks.

        Raises
        ------
        KeyError
            If 'index' or 'total' keys are missing from the changes dictionary.

        Examples
        --------
        progress_container = st.empty()
        logger = StreamlitProgressBarLogger(progress_container)
        logger.callback(index=1, total=10)
        """
        if "index" in changes and "total" in changes:
            progress = min(changes["index"] / changes["total"], 1.0)
            self.progress_container.progress(progress)
