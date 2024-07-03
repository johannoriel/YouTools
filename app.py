import os
import json
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

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
        if st.button("Sauvegarder"):
            config['channel_id'] = channel_id
            save_config(config)
            st.success("ID de la chaîne sauvegardé !")
    
    # Onglet Liste des vidéos
    if config.get('channel_id'):
        st.header("5 dernières vidéos")
        videos = get_channel_videos(config['channel_id'])
        
        for video in videos:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(video['thumbnail'])
            with col2:
                st.subheader(video['title'])
                st.markdown(f"[Voir la vidéo](https://www.youtube.com/watch?v={video['video_id']})")
    else:
        st.info("Veuillez configurer l'ID de la chaîne dans l'onglet Configuration.")

if __name__ == "__main__":
    main()
