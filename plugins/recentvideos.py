from global_vars import translations, t
from app import Plugin

import streamlit as st
import os
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript
import yt_dlp
from youtube_api import YoutubeAPI

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "recent_videos_tab": "Recent YouTube Videos",
    "recent_videos_header": "10 Latest Videos",
    "recent_videos_transcript_button": "Transcript",
    "recent_videos_transcript_header": "Transcript",
    "recent_videos_transcript_language": "Transcript Language:",
    "recent_videos_transcript_content": "Transcript Content",
    "recent_videos_copy_transcript_button": "Copy Transcript",
    "recent_videos_copy_success": "Transcript copied! Use Ctrl+C (or Cmd+C on Mac) to copy it from the code block above.",
    "recent_videos_download_transcript_button": "Download Transcript",
    "recent_videos_process_llm_button": "Process with LLM",
    "recent_videos_llm_response_header": "LLM Response",
    "recent_videos_llm_response_content": "LLM Response Content",
    "recent_videos_copy_llm_response_button": "Copy LLM Response",
    "recent_videos_llm_copy_success": "LLM Response copied! Use Ctrl+C (or Cmd+C on Mac) to copy it from the code block above.",
    "recent_videos_download_llm_response_button": "Download LLM Response",
    "recent_videos_error": "An error occurred: ",
    "recent_videos_transcripts_disabled": "Transcripts are disabled for this video.",
    "recent_videos_no_transcript_available": "No transcript is available for this video.",
    "recent_videos_transcript_error": "An error occurred while retrieving the transcript: ",
    "recent_videos_configure_channel_id": "Please configure the channel ID in the Configuration tab.",
    "recent_videos_download_button": "Download Video",
    "recent_videos_download_success": "Video downloaded successfully!",
    "recent_videos_download_error": "An error occurred while downloading the video: ",
    "recent_videos_save_success": "Transcript saved successfully!",
    "recent_videos_save_transcript": "Save transcript",
})

translations["fr"].update({
    "recent_videos_tab": "Vidéos récentes",
    "recent_videos_header": "10 dernières vidéos",
    "recent_videos_transcript_button": "Transcript",
    "recent_videos_transcript_header": "Transcript",
    "recent_videos_transcript_language": "Langue de la transcription :",
    "recent_videos_transcript_content": "Contenu du transcript",
    "recent_videos_copy_transcript_button": "Copier le transcript",
    "recent_videos_copy_success": "Transcript copié ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour le copier depuis le bloc de code ci-dessus.",
    "recent_videos_download_transcript_button": "Télécharger le transcript",
    "recent_videos_process_llm_button": "Traiter avec LLM",
    "recent_videos_llm_response_header": "Réponse du LLM",
    "recent_videos_llm_response_content": "Contenu de la réponse",
    "recent_videos_copy_llm_response_button": "Copier la réponse du LLM",
    "recent_videos_llm_copy_success": "Réponse du LLM copiée ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour la copier depuis le bloc de code ci-dessus.",
    "recent_videos_download_llm_response_button": "Télécharger la réponse du LLM",
    "recent_videos_error": "Une erreur s'est produite : ",
    "recent_videos_transcripts_disabled": "Les transcriptions sont désactivées pour cette vidéo.",
    "recent_videos_no_transcript_available": "Aucune transcription n'est disponible pour cette vidéo.",
    "recent_videos_transcript_error": "Une erreur s'est produite lors de la récupération du transcript : ",
    "recent_videos_configure_channel_id": "Veuillez configurer l'ID de la chaîne dans l'onglet Configuration.",
    "recent_videos_download_button": "Télécharger la vidéo",
    "recent_videos_download_success": "Vidéo téléchargée avec succès !",
    "recent_videos_download_error": "Une erreur s'est produite lors du téléchargement de la vidéo : ",
    "recent_videos_save_success": "Transcript sauvée avec succès!",
    "recent_videos_save_transcript": "Sauver le transcript",
})


class RecentvideosPlugin(Plugin):

    def get_tabs(self):
        return [{"name": t("recent_videos_tab"), "plugin": "recentvideos"}]

    def get_transcript(self, video_id, language):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=[language])
        except CouldNotRetrieveTranscript:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
            except Exception as e:
                return f"{t('recent_videos_transcript_error')}{str(e)}", "N/A"

        full_transcript = " ".join([entry['text'] for entry in transcript])

        return full_transcript, language

    def download_video(self, video_url, work_directory, title):
        # Nettoyer le titre pour enlever les caractères spéciaux et tout ce qui suit "|"
        clean_title = re.sub(r'[^\w\-_\. ]', '', title.split('|')[0].strip())

        # Chemin complet du fichier
        file_path = os.path.join(work_directory, f"{clean_title}.mp4")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': file_path,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return True, file_path
        except Exception as e:
            return False, str(e)

    def run(self, config):
        st.header(t("recent_videos_header"))
        api_key = config['common']['youtube_api_key']

        if 'channel_id' in config['common'] and config['common']['channel_id']:
            # Utilisation de YoutubeAPI pour récupérer toutes les vidéos
            youtube_api = YoutubeAPI(config)
            if 'all_videos' not in st.session_state:
                st.session_state.all_videos = youtube_api.get_channel_videos(
                    config['common']['channel_id'])

            # Ajout d'un champ de filtre
            filter_keywords = st.text_input("Filtrer les vidéos par mots-clés")

            # Filtrage des vidéos en fonction des mots-clés
            if filter_keywords:
                filtered_videos = [video for video in st.session_state.all_videos if filter_keywords.lower(
                ) in video['title'].lower()]
            else:
                filtered_videos = st.session_state.all_videos

            # Pagination des résultats filtrés
            page_size = 10  # Nombre de vidéos par page
            page_number = st.session_state.get('page_number', 0)
            total_pages = (len(filtered_videos) + page_size - 1) // page_size

            # Affichage des vidéos de la page actuelle
            start_index = page_number * page_size
            end_index = start_index + page_size
            videos_to_display = filtered_videos[start_index:end_index]

            for video in videos_to_display:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.image(video['thumbnail'])
                with col2:
                    st.subheader(video['title'])
                    st.markdown(
                        f"[Voir la vidéo](https://www.youtube.com/watch?v={video['video_id']})")
                    # Affichage du statut de la vidéo
                    st.write(f"Statut : {video['status']}")
                    if video['is_short']:
                        # Indication que la vidéo est un Short
                        st.write("**Short** 🎥")
                with col3:
                    if st.button(t("recent_videos_transcript_button"), key=f"transcript_{video['video_id']}"):
                        transcript, lang = self.get_transcript(
                            video['video_id'], config['common']['language'])
                        st.session_state.transcript = transcript
                        st.session_state.title = video['title']
                        st.session_state.transcript_lang = lang
                        st.session_state.show_transcript = True
                        st.session_state.current_video_id = video['video_id']

                    # Bouton pour télécharger la vidéo
                    if st.button(t("recent_videos_download_button"), key=f"download_{video['video_id']}"):
                        work_directory = config['common']['work_directory']
                        video_url = f"https://www.youtube.com/watch?v={video['video_id']}"
                        success, result = self.download_video(
                            video_url, work_directory, video['title'])
                        if success:
                            st.success(t("recent_videos_download_success"))
                        else:
                            st.error(
                                f"{t('recent_videos_download_error')}{result}")

            # Afficher les boutons de pagination
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if page_number > 0 and st.button("Page Précédente"):
                    st.session_state.page_number -= 1
                    st.experimental_rerun()
            with col2:
                st.write(f"Page {page_number + 1} / {total_pages}")
            with col3:
                if page_number < total_pages - 1 and st.button("Page Suivante"):
                    st.session_state.page_number += 1
                    st.experimental_rerun()

        else:
            st.info(t("recent_videos_configure_channel_id"))

        # Affichage du transcript
        if st.session_state.get('show_transcript', False):
            st.header(t("recent_videos_transcript_header"))
            st.write(
                f"{t('recent_videos_transcript_language')} {st.session_state.transcript_lang}")
            st.text_area(t("recent_videos_transcript_content"),
                         st.session_state.transcript, height=300)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button(t("recent_videos_copy_transcript_button")):
                    st.code(st.session_state.transcript)
                    st.success(t("recent_videos_copy_success"))
            with col2:
                st.download_button(
                    label=t("recent_videos_download_transcript_button"),
                    data=st.session_state.transcript,
                    file_name=f"transcript_{st.session_state.transcript_lang}.txt",
                    mime="text/plain"
                )
            with col3:
                if st.button(t("recent_videos_save_transcript")):
                    work_directory = config['common']['work_directory']
                    with open(os.path.join(work_directory, "transcript.txt"), "w", encoding="utf-8") as f:
                        f.write(st.session_state.transcript)
                    with open(os.path.join(work_directory, "url.txt"), "w", encoding="utf-8") as f:
                        f.write(
                            f"https://www.youtube.com/watch?v={st.session_state.current_video_id}")
                    st.success(t("recent_videos_save_success"))
            with col4:
                llm_plugin = self.plugin_manager.get_plugin('llm')
                llm_config = config.get('llm', {})
                prompt = llm_config.get('llm_prompt', '')
                with st.expander("Prompt"):
                    st.markdown("Voulez-vous changer le prompt ?")
                    prompt = st.text_input("Nouveau prompt", prompt)
                if st.button(t("recent_videos_process_llm_button")):
                    video_content = f"# {st.session_state.title} \n {st.session_state.transcript}"
                    llm_response = llm_plugin.process_with_llm(
                        prompt,
                        llm_config.get('llm_sys_prompt', ''),
                        video_content
                    )
                    st.session_state.llm_response = llm_response
                    st.session_state.show_llm_response = True

        # Affichage de la réponse du LLM
        if st.session_state.get('show_llm_response', False):
            st.header(t("recent_videos_llm_response_header"))
            st.text_area(t("recent_videos_llm_response_content"),
                         st.session_state.llm_response, height=300)
            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("recent_videos_copy_llm_response_button")):
                    st.code(st.session_state.llm_response)
                    st.success(t("recent_videos_llm_copy_success"))
            with col2:
                st.download_button(
                    label=t("recent_videos_download_llm_response_button"),
                    data=st.session_state.llm_response,
                    file_name=f"llm_response_{st.session_state.current_video_id}.txt",
                    mime="text/plain"
                )
