from global_vars import t, translations
from app import Plugin
import streamlit as st

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "channel_id": "YouTube Channel ID",
    "work_directory": "Work Directory",
    "preferred_language": "Preferred Language for Transcriptions",
    "upload_finished": "Upload finished ! ID of the video",
})
translations["fr"].update({
    "channel_id": "ID de la chaîne YouTube",
    "work_directory": "Répertoire de travail",
    "preferred_language": "Langue préférée pour les transcriptions",
    "upload_finished": "Upload terminé ! ID de la vidéo",
})

class CommonPlugin(Plugin):
    def get_config_fields(self):
        return {
            "channel_id": {
                "type": "text",
                "label": t("channel_id"),
                "default": ""
            },
            "work_directory": {
                "type": "text",
                "label": t("work_directory"),
                "default": "/home/joriel/Vidéos"
            },            
            "language": {
                "type": "select",
                "label": t("preferred_language"),
                "options": [("fr", "Français"), ("en", "Anglais")],
                "default": "fr"
            },
        }

    def get_tabs(self):
        return [{"name": "Commun", "plugin": "common"}]

    def run(self, config):
        st.header("Common Plugin")
        st.write(f"Channel: {config['common']['channel_id']}")
        st.write(f"{t('work_directory')}: {config['common']['work_directory']}")

import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
def get_credentials():
    creds = None
    os.environ['BROWSER'] = '/snap/bin/chromium'
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except:
            os.remove('token.json')
            creds = None
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                os.remove('token.json')
                creds = None
        
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', self.SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds
    
# Fonction pour uploader la vidéo sur YouTube
def upload_video(filename, title, description, category, keywords, privacy_status):
    credentials = get_credentials()
    youtube = build('youtube', 'v3', credentials=credentials)
    
    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': keywords,
            'categoryId': category
        },
        'status': {
            'privacyStatus': privacy_status
        }
    }

    media = MediaFileUpload(filename, resumable=True)
    
    request = youtube.videos().insert(
        part=','.join(body.keys()),
        body=body,
        media_body=media
    )

    response = None
    progress_bar = st.progress(0)
    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            progress_bar.progress(progress)
            st.write(f"Progression : {progress}%")

    st.success(t('upload_finished')+f" : {response['id']}")
    return response['id']

def list_video_files(directory):
    video_files = []
    outfile_videos = []
    chroma_videos = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.mkv', '.mp4')):
            full_path = os.path.join(directory, file)
            mod_time = os.path.getmtime(full_path)
            if file.startswith('outfile_'):
                outfile_videos.append((file, full_path, mod_time))
            elif file.startswith('chroma_'):
                chroma_videos.append((file, full_path, mod_time))
            else:
                video_files.append((file, full_path, mod_time))
    
    video_files.sort(key=lambda x: x[2], reverse=True)
    outfile_videos.sort(key=lambda x: x[2], reverse=True)
    chroma_videos.sort(key=lambda x: x[2], reverse=True)
    return video_files, outfile_videos, chroma_videos

def remove_quotes(s):
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s
