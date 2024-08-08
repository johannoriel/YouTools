from global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import upload_video, remove_quotes, list_all_video_files
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
    "directpublish_publish_button": "Process",
    "directpublish_processing": "Processing video...",
    "directpublish_success": "Video successfully published! Video URL: https://www.youtube.com/watch?v={video_id}, Edition : https://studio.youtube.com/video/{video_id}/edit",
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
    "directpublish_preprompt": "Change prompt is you wish (be specific):",
    "directpublish_title_generator" : "Generate a catchy title for a YouTube video based on this summary, not exceeding 100 characters, without commenting, just the title, without quotation marks.",
    "directpublish_tag_generator" : "Generate a comma list of keywords describing the subject, without any comment or adding, just a raw list of comma seaparated keywords",
    "directpublish_addings" : "Add any text to your description (will not be modified)",
    "directpublish_notags" : "Invalid tags - upload without them",
    "directpublish_dopublish" : "Publish to YouTube",
})

translations["fr"].update({
    "directpublish_tab": "Publication Directe",
    "directpublish_header": "Publier une Vidéo sur YouTube",
    "directpublish_select_video": "Sélectionner une vidéo à publier",
    "directpublish_remove_silences": "Retirer les silences avant la publication",
    "directpublish_video_category": "Catégorie de la vidéo",
    "directpublish_publish_button": "Lancer le traitement",
    "directpublish_processing": "Traitement de la vidéo en cours...",
    "directpublish_success": "Vidéo publiée avec succès ! URL de la vidéo : https://www.youtube.com/watch?v={video_id}, édition : https://studio.youtube.com/video/{video_id}/edit",
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
    "directpublish_preprompt": "Modifiez le prompt si besoin (rajoutez des éléments spécifiques):",
    "directpublish_title_generator" : "Génère un titre accrocheur pour une vidéo YouTube basée sur ce résumé, sans dépasser 100 caractères, sans commenter, juste le titre, sans guillemets.",
    "directpublish_tag_generator" : "Génère une liste de mots-clés décrivant le sujet, sans commentaire ni ajout, juste une liste brute séparée par des virgules",
    "directpublish_addings" : "Rajoutez du texte à votre description (ne sera pas modifié)",
    "directpublish_notags" : "Tags invalides - upload sans eux",
    "directpublish_dopublish" : "Publier sur YouTube",
})

def cut_string(text, limit=500):
    # Check if the length of the text is less than the limit
    if len(text) <= limit:
        return text

    # Cut the text at the limit of characters
    cut_text = text[:limit]

    # Find the last position of a space to avoid cutting in the middle of a word
    last_space = cut_text.rfind(' ')

    # Return the text up to the last space
    return cut_text[:last_space]

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
        do_publish = st.checkbox(t("directpublish_dopublish"), value=True)

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

        if 'rag_question' not in st.session_state:
            st.session_state.rag_question = config['llm']['llm_prompt']

        user_prompt = st.text_area(t("directpublish_preprompt"), value=st.session_state.rag_question, key="rag_prompt_key")
        st.session_state.rag_question = user_prompt

        if 'addings' not in st.session_state:
            st.session_state.addings = ""
        addings = st.text_area(t('directpublish_addings'), value=st.session_state.addings, key="directpublish_addings")
        st.session_state.addings = addings

        if st.button(t("directpublish_publish_button")):
            with st.spinner(t("directpublish_processing")):
                #try:
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

                    if do_publish:
                        # 3. Transcrire la vidéo
                        st.text(t("directpublish_generating_transcription"))
                        transcript = self.transcript_plugin.transcribe_video(
                            video_to_process,
                            "txt",
                            config['transcript']['whisper_path'],
                            config['transcript']['whisper_model'],
                            config['transcript']['ffmpeg_path'],
                            config['common']['language']
                        )
                        st.code(transcript)
                        st.session_state.transcript = transcript # May bu used by other plugins

                        # 4. Générer un résumé du transcript
                        st.text(t("directpublish_generating_description"))
                        description = self.ragllm_plugin.process_with_llm(
                            user_prompt,
                            config['ragllm']['llm_sys_prompt'],
                            transcript
                        )
                        st.code(description)
                        signature = config['directpublish']['signature']

                        # 5. Générer un titre pour la vidéo
                        st.text(t("directpublish_generating_title"))
                        title_prompt = t("directpublish_title_generator")
                        title = remove_quotes(self.ragllm_plugin.process_with_llm(
                            title_prompt,
                            config['ragllm']['llm_sys_prompt'],
                            transcript
                        )).split('\n')[0].strip()
                        st.code(title)

                        # 5. Générer un titre pour la vidéo
                        tag_prompt = t("directpublish_tag_generator")
                        tags = remove_quotes(cut_string(self.ragllm_plugin.process_with_llm(
                            tag_prompt,
                            config['ragllm']['llm_sys_prompt'],
                            transcript
                        )))
                        st.code(tags)

                        # 6. Uploader la vidéo sur YouTube
                        st.text(t("directpublish_upload"))
                        try:
                            video_id = upload_video(
                                video_to_process,
                                title,
                                f"{description}\n\n{addings}\n{signature}",
                                selected_category,
                                tags.split(','),  # keywords (optionnel)
                                "unlisted"
                            )
                        except :
                            video_id = upload_video(
                                video_to_process,
                                title,
                                f"{description}\n\n{addings}\n{signature}",
                                selected_category,
                                "unlisted"
                            )
                            st.success(t("directpublish_notags"))

                        st.success(t("directpublish_success").format(video_id=video_id))
                        print("Upload finished")

                    #except Exception as e:
                    #    st.error(t("directpublish_error").format(error=str(e)))
