from app import Plugin

import streamlit as st
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable

class RecentvideosPlugin(Plugin):

    def get_tabs(self):
        return [{"name": "Résumé des vidéos", "plugin": "recentvideos"}]

    def get_channel_videos(self, channel_id, api_key):
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
                maxResults=5
            ).execute()
            
            videos = []
            for item in playlist_response['items']:
                video = {
                    'title': item['snippet']['title'],
                    'video_id': item['snippet']['resourceId']['videoId'],
                    'thumbnail': item['snippet']['thumbnails']['default']['url']
                }
                videos.append(video)
            
            return videos
        except HttpError as e:
            st.error(f"Une erreur s'est produite : {e}")
            return []

    def get_transcript(self, video_id, language):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        except NoTranscriptAvailable:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
            except TranscriptsDisabled:
                return "Les transcriptions sont désactivées pour cette vidéo.", "N/A"
            except NoTranscriptAvailable:
                return "Aucune transcription n'est disponible pour cette vidéo.", "N/A"
            except Exception as e:
                return f"Une erreur s'est produite lors de la récupération du transcript : {str(e)}", "N/A"
        
        full_transcript = " ".join([entry['text'] for entry in transcript])
        
        return full_transcript, language

    def run(self, config):
        st.header("5 dernières vidéos")
        api_key = config['api_key']
        if 'channel_id' in config['common'] and config['common']['channel_id']:
            videos = self.get_channel_videos(config['common']['channel_id'], api_key)
            
            for video in videos:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.image(video['thumbnail'])
                with col2:
                    st.subheader(video['title'])
                    st.markdown(f"[Voir la vidéo](https://www.youtube.com/watch?v={video['video_id']})")
                with col3:
                    if st.button("Transcript", key=f"transcript_{video['video_id']}"):
                        transcript, lang = self.get_transcript(video['video_id'], config['common']['language'])
                        st.session_state.transcript = transcript
                        st.session_state.title = video['title']
                        st.session_state.transcript_lang = lang
                        st.session_state.show_transcript = True
                        st.session_state.current_video_id = video['video_id']
        else:
            st.info("Veuillez configurer l'ID de la chaîne dans l'onglet Configuration.")
        
        # Affichage du transcript
        if st.session_state.get('show_transcript', False):
            st.header("Transcript")
            st.write(f"Langue de la transcription : {st.session_state.transcript_lang}")
            st.text_area("Contenu du transcript", st.session_state.transcript, height=300)
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Copier le transcript"):
                    st.code(st.session_state.transcript)
                    st.success("Transcript copié ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour le copier depuis le bloc de code ci-dessus.")
            with col2:
                st.download_button(
                    label="Télécharger le transcript",
                    data=st.session_state.transcript,
                    file_name=f"transcript_{st.session_state.transcript_lang}.txt",
                    mime="text/plain"
                )
            with col3:
                llm_plugin = self.plugin_manager.get_plugin('llm')
                llm_config = config.get('llm', {})
                prompt = llm_config.get('llm_prompt', '')
                with st.popover("Correction du prompt"):
                    st.markdown("Voulez-vous changer le prompt ?")
                    prompt = st.text_input("Nouveau prompt", prompt)
                if st.button("Traiter avec LLM"):
                    video_content = f"# {st.session_state.title} \n {st.session_state.transcript}"
                    llm_response = llm_plugin.process_with_llm(
                        prompt, 
                        llm_config.get('llm_sys_prompt', ''), 
                        video_content, 
                        llm_config.get('llm_model', '')
                    )
                    st.session_state.llm_response = llm_response
                    st.session_state.show_llm_response = True
                    
        # Affichage de la réponse du LLM
        if st.session_state.get('show_llm_response', False):
            st.header("Réponse du LLM")
            st.text_area("Contenu de la réponse", st.session_state.llm_response, height=300)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Copier la réponse du LLM"):
                    st.code(st.session_state.llm_response)
                    st.success("Réponse du LLM copiée ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour la copier depuis le bloc de code ci-dessus.")
            with col2:
                st.download_button(
                    label="Télécharger la réponse du LLM",
                    data=st.session_state.llm_response,
                    file_name=f"llm_response_{st.session_state.current_video_id}.txt",
                    mime="text/plain"
                )
