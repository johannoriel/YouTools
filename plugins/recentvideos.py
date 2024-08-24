from global_vars import translations, t
from app import Plugin

import streamlit as st
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable

from plugins.common import get_credentials

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "recent_videos_tab": "10 Latest YouTube Videos",
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
    "recent_videos_configure_channel_id": "Please configure the channel ID in the Configuration tab."
})

translations["fr"].update({
    "recent_videos_tab": "10 dernières vidéos Youtube",
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
    "recent_videos_configure_channel_id": "Veuillez configurer l'ID de la chaîne dans l'onglet Configuration."
})

class RecentvideosPlugin(Plugin):

    def get_tabs(self):
        return [{"name": t("recent_videos_tab"), "plugin": "recentvideos"}]

    def get_transcript(self, video_id, language):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        except NoTranscriptAvailable:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
            except TranscriptsDisabled:
                return t("recent_videos_transcripts_disabled"), "N/A"
            except NoTranscriptAvailable:
                return t("recent_videos_no_transcript_available"), "N/A"
            except Exception as e:
                return f"{t('recent_videos_transcript_error')}{str(e)}", "N/A"

        full_transcript = " ".join([entry['text'] for entry in transcript])

        return full_transcript, language

    def get_channel_public_videos(self, channel_id, api_key, page_token=None):
        youtube = build('youtube', 'v3', developerKey=api_key)

        try:
            channel_response = youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()

            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            playlist_response = youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_playlist_id,
                maxResults=10,  # Affiche 10 vidéos par page
                pageToken=page_token  # Ajout de la gestion des pages
            ).execute()

            videos = []
            for item in playlist_response['items']:
                video = {
                    'title': item['snippet']['title'],
                    'video_id': item['snippet']['resourceId']['videoId'],
                    'thumbnail': item['snippet']['thumbnails']['default']['url']
                }
                videos.append(video)

            next_page_token = playlist_response.get('nextPageToken')
            prev_page_token = playlist_response.get('prevPageToken')

            return videos, next_page_token, prev_page_token
        except HttpError as e:
            st.error(f"{t('recent_videos_error')}{e}")
            return [], None, None

    def get_channel_videos(self, channel_id, page_token=None):
        # Utilisation des credentials pour obtenir l'accès aux vidéos protégées
        credentials = get_credentials()
        youtube = build('youtube', 'v3', credentials=credentials)

        try:
            # Récupération des informations sur la chaîne
            channel_response = youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()

            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            # Récupération des vidéos avec gestion de la pagination
            playlist_response = youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_playlist_id,
                maxResults=10,  # Affiche 10 vidéos par page
                pageToken=page_token  # Gestion des pages
            ).execute()

            videos = []
            for item in playlist_response['items']:
                video = {
                    'title': item['snippet']['title'],
                    'video_id': item['snippet']['resourceId']['videoId'],
                    'thumbnail': item['snippet']['thumbnails']['default']['url']
                }
                videos.append(video)

            # Gestion des tokens pour la pagination
            next_page_token = playlist_response.get('nextPageToken')
            prev_page_token = playlist_response.get('prevPageToken')

            return videos, next_page_token, prev_page_token

        except HttpError as e:
            st.error(f"Une erreur s'est produite : {e}")
            return [], None, None

    def run(self, config):
        st.header(t("recent_videos_header"))
        api_key = config['api_key']

        if 'channel_id' in config['common'] and config['common']['channel_id']:
            page_token = st.session_state.get('page_token', None)

            videos, next_page_token, prev_page_token = self.get_channel_videos(
                config['common']['channel_id'], page_token
            )

            for video in videos:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.image(video['thumbnail'])
                with col2:
                    st.subheader(video['title'])
                    st.markdown(f"[Voir la vidéo](https://www.youtube.com/watch?v={video['video_id']})")
                with col3:
                    if st.button(t("recent_videos_transcript_button"), key=f"transcript_{video['video_id']}"):
                        transcript, lang = self.get_transcript(video['video_id'], config['common']['language'])
                        st.session_state.transcript = transcript
                        st.session_state.title = video['title']
                        st.session_state.transcript_lang = lang
                        st.session_state.show_transcript = True
                        st.session_state.current_video_id = video['video_id']

            # Afficher les boutons de pagination
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if prev_page_token and st.button("Page Précédente"):
                    st.session_state.page_token = prev_page_token
                    st.experimental_rerun()
            with col2:
                st.write("")  # Espace pour alignement
            with col3:
                if next_page_token and st.button("Page Suivante"):
                    st.session_state.page_token = next_page_token
                    st.experimental_rerun()

        else:
            st.info(t("recent_videos_configure_channel_id"))

        # Affichage du transcript
        if st.session_state.get('show_transcript', False):
            st.header(t("recent_videos_transcript_header"))
            st.write(f"{t('recent_videos_transcript_language')} {st.session_state.transcript_lang}")
            st.text_area(t("recent_videos_transcript_content"), st.session_state.transcript, height=300)
            col1, col2, col3 = st.columns(3)
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
            st.text_area(t("recent_videos_llm_response_content"), st.session_state.llm_response, height=300)
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
