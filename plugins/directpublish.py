from global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import list_video_files, upload_video, remove_quotes, list_all_video_files
from plugins.trimsilences import TrimsilencesPlugin
from plugins.transcript import TranscriptPlugin
from plugins.ragllm import RagllmPlugin
from plugins.chromakey import ChromakeyPlugin
from chromakey_background import replace_background
import os

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "directpublish_tab": "Direct Publish",
    "directpublish_header": "Publish Video to YouTube",
    "directpublish_select_video": "Select a video to publish",
    "directpublish_remove_silences": "Remove silences before publishing",
    "directpublish_video_category": "Video category",
    "directpublish_publish_button": "Publish to YouTube",
    "directpublish_processing": "Processing video...",
    "directpublish_success": "Video successfully published! Video URL: https://www.youtube.com/watch?v={video_id}",
    "directpublish_error": "An error occurred during publication: {error}",
    "directpublish_generating_title": "Generating video title...",
    "directpublish_generating_description": "Generating video description...",
    "directpublish_generating_transcription": "Generating video transcription...",
    "directpublish_silence_trim": "Suppression of silences...",
    "directpublish_upload": "Uploading...",
    "publish_signature": "Signature to add in video description",
    "publish_signature_default": "",
    "directpublish_replace_green_screen": "Replace green screen background",
    "directpublish_select_background": "Select a background video",
    "directpublish_replacing_background": "Replacing green screen background...",
})

translations["fr"].update({
    "directpublish_tab": "Publication Directe",
    "directpublish_header": "Publier une Vidéo sur YouTube",
    "directpublish_select_video": "Sélectionner une vidéo à publier",
    "directpublish_remove_silences": "Retirer les silences avant la publication",
    "directpublish_video_category": "Catégorie de la vidéo",
    "directpublish_publish_button": "Publier sur YouTube",
    "directpublish_processing": "Traitement de la vidéo en cours...",
    "directpublish_success": "Vidéo publiée avec succès ! URL de la vidéo : https://www.youtube.com/watch?v={video_id}",
    "directpublish_error": "Une erreur s'est produite lors de la publication : {error}",
    "directpublish_generating_title": "Génération du titre de la vidéo...",
    "directpublish_generating_description": "Génération de la description de la vidéo...",
    "directpublish_generating_transcription": "Génération de la transcription transcription...",
    "directpublish_silence_trim": "Suppression des silences...",
    "directpublish_upload": "Téléversement...",
    "publish_signature": "Signature à ajouter à la description de la vidéo",
    "publish_signature_default": "",
    "directpublish_replace_green_screen": "Remplacer le fond vert",
    "directpublish_select_background": "Sélectionner une vidéo de fond",
    "directpublish_replacing_background": "Remplacement du fond vert...",
})

class DirectpublishPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.trimsilences_plugin = self.plugin_manager.get_plugin('trimsilences')
        self.transcript_plugin = self.plugin_manager.get_plugin('transcript')
        self.ragllm_plugin = self.plugin_manager.get_plugin('ragllm')
        self.chromakey_plugin = self.plugin_manager.get_plugin('chromakey')


    def get_config_fields(self):
        return {
            "signature": {
                "type": "textarea",
                "label": t("publish_signature"),
                "default": t("publish_signature_default")
            },
        }

    def get_tabs(self):
        return [{"name": t("directpublish_tab"), "plugin": "directpublish"}]

    def run(self, config):
        st.header(t("directpublish_header"))

        # Sélection de la vidéo
        work_directory = config['common']['work_directory']
        video_files = list_all_video_files(work_directory)
        if not video_files:
            st.warning(t("transcript_no_videos"))
            return

        selected_video = st.selectbox(
            t("directpublish_select_video"),
            options=[v[0] for v in video_files],
            index=0  # Sélectionne par défaut la vidéo la plus récente
        )
        selected_video_path = next(v[1] for v in video_files if v[0] == selected_video)

        # Option pour retirer les silences
        remove_silences = st.checkbox(t("directpublish_remove_silences"))
        replace_green_screen = st.checkbox(t("directpublish_replace_green_screen"))

        # Sélection du fond si le remplacement du fond vert est activé
        background_video = None
        if replace_green_screen:
            background_directory = config['chromakey']['background_directory']
            background_files = [f for f in os.listdir(background_directory) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            background_video = st.selectbox(t("directpublish_select_background"), background_files)

        # Sélection de la catégorie
        categories = {
            "1": "Film & Animation",
            "2": "Autos & Vehicles",
            "10": "Music",
            "15": "Pets & Animals",
            "17": "Sports",
            "18": "Short Movies",
            "19": "Travel & Events",
            "20": "Gaming",
            "21": "Videoblogging",
            "22": "People & Blogs",
            "23": "Comedy",
            "24": "Entertainment",
            "25": "News & Politics",
            "26": "Howto & Style",
            "27": "Education",
            "28": "Science & Technology",
            "29": "Nonprofits & Activism",
            "30": "Movies",
            "31": "Anime/Animation",
            "32": "Action/Adventure",
            "33": "Classics",
            "34": "Comedy",
            "35": "Documentary",
            "36": "Drama",
            "37": "Family",
            "38": "Foreign",
            "39": "Horror",
            "40": "Sci-Fi/Fantasy",
            "41": "Thriller",
            "42": "Shorts",
            "43": "Shows",
            "44": "Trailers"
        }
        category_keys = list(categories.keys())
        selected_category = st.selectbox(
            t("directpublish_video_category"),
            options=list(categories.keys()),
            index=category_keys.index("28"),
            format_func=lambda x: categories[x]
        )

        if st.button(t("directpublish_publish_button")):
            with st.spinner(t("directpublish_processing")):
                try:
                    video_to_process = selected_video_path

                    # 1. Retirer les silences si demandé
                    if remove_silences:
                        st.text(t("directpublish_silence_trim"))
                        result = self.trimsilences_plugin.remove_silence(
                            video_to_process,
                            config['trimsilences']['silence_threshold'],
                            config['trimsilences']['silence_duration'],
                            work_directory
                        )
                        if isinstance(result, str) and (result.startswith("Erreur") or result.startswith("Une erreur")):
                            st.error(result)
                            return
                        video_to_process = result
                        st.text(video_to_process)

                    # 2. Remplacer le fond vert si demandé
                    if replace_green_screen and background_video:
                        st.text(t("directpublish_replacing_background"))
                        background_path = os.path.join(config['chromakey']['background_directory'], background_video)
                        result_filename = f"chroma_{os.path.basename(video_to_process)}"
                        result_path = os.path.join(work_directory, result_filename)
                        replace_background(video_to_process, background_path, result_path)
                        video_to_process = result_path
                        st.text(video_to_process)

                    # 3. Transcrire la vidéo
                    st.text(t("directpublish_generating_transcription"))
                    transcript = self.transcript_plugin.transcribe_video(
                        video_to_process,
                        "txt",
                        os.path.expanduser(config['transcript']['whisper_path']),
                        config['transcript']['whisper_model'],
                        config['transcript']['ffmpeg_path'],
                        config['common']['language']
                    )
                    st.code(transcript)
                    st.session_state.rag_text = transcript

                    # 4. Générer un résumé du transcript
                    st.text(t("directpublish_generating_description"))
                    summary_prompt = "Résume les grandes lignes du transcript, sous forme de liste à puce, sans commenter, pour écrire une introduction au sujet, sans parler du contexte ou de l'auteur. Décrit uniquement."
                    description = self.ragllm_plugin.process_with_llm(
                        summary_prompt,
                        config['ragllm']['llm_sys_prompt'],
                        transcript,
                        config['ragllm']['llm_model']
                    ) + config['directpublish']['signature']
                    st.code(description)

                    # 5. Générer un titre pour la vidéo
                    st.text(t("directpublish_generating_title"))
                    title_prompt = f"Génère un titre accrocheur pour une vidéo YouTube basée sur ce résumé, sans dépasser 100 caractères, sans commenter, juste le titre, sans guillemets."
                    title = remove_quotes(self.ragllm_plugin.process_with_llm(
                        title_prompt,
                        config['ragllm']['llm_sys_prompt'],
                        description,
                        config['ragllm']['llm_model']
                    ))
                    st.code(title)

                    # 6. Uploader la vidéo sur YouTube
                    st.text(t("directpublish_upload"))
                    video_id = upload_video(
                        video_to_process,
                        title,
                        description,
                        selected_category,
                        [],  # keywords (optionnel)
                        "unlisted"
                    )

                    st.success(t("directpublish_success").format(video_id=video_id))

                except Exception as e:
                    st.error(t("directpublish_error").format(error=str(e)))
