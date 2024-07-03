import os
import json
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API depuis les variables d'environnement
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Nom du fichier de configuration
CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def get_channel_videos(channel_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
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

def get_transcript(video_id, language):
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

def main():
    st.set_page_config(page_title="YouTube Dashboard", layout="wide")
    
    st.title("YouTube Channel Dashboard")
    
    if not API_KEY:
        st.error("La clé API YouTube n'est pas définie. Veuillez vérifier votre fichier .env")
        return
    
    # Charger la configuration
    config = load_config()
    
    # Onglet Configuration
    with st.expander("Configuration"):
        channel_id = st.text_input("ID de la chaîne YouTube", value=config.get('channel_id', ''))
        language = st.selectbox("Langue préférée pour les transcriptions", 
                                options=['fr', 'en'], 
                                format_func=lambda x: 'Français' if x == 'fr' else 'Anglais',
                                index=0 if config.get('language', 'fr') == 'fr' else 1)
        if st.button("Sauvegarder"):
            config['channel_id'] = channel_id
            config['language'] = language
            save_config(config)
            st.success("Configuration sauvegardée !")
    
    # Onglet Liste des vidéos
    if config.get('channel_id'):
        st.header("5 dernières vidéos")
        videos = get_channel_videos(config['channel_id'])
        
        for video in videos:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.image(video['thumbnail'])
            with col2:
                st.subheader(video['title'])
                st.markdown(f"[Voir la vidéo](https://www.youtube.com/watch?v={video['video_id']})")
            with col3:
                if st.button("Transcript", key=f"transcript_{video['video_id']}"):
                    transcript, lang = get_transcript(video['video_id'], config['language'])
                    st.session_state.transcript = transcript
                    st.session_state.transcript_lang = lang
                    st.session_state.show_transcript = True
    else:
        st.info("Veuillez configurer l'ID de la chaîne dans l'onglet Configuration.")
    
    # Affichage du transcript
    if st.session_state.get('show_transcript', False):
        st.header("Transcript")
        st.write(f"Langue de la transcription : {st.session_state.transcript_lang}")
        st.text_area("Contenu du transcript", st.session_state.transcript, height=300)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Copier"):
                st.code(st.session_state.transcript)
                st.success("Transcript copié ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour le copier depuis le bloc de code ci-dessus.")
        with col2:
            st.download_button(
                label="Télécharger",
                data=st.session_state.transcript,
                file_name=f"transcript_{st.session_state.transcript_lang}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
