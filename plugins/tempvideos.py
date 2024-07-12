import streamlit as st
from app import Plugin
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import RefreshError

import os
import datetime
import re

class TempvideosPlugin(Plugin):
    def __init__(self, name):
        super().__init__(name)
        self.SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

    def get_tabs(self):
        return [{"name": "Vidéos temporaires", "plugin": "tempvideos"}]

    def get_credentials(self):
        creds = None
        os.environ['BROWSER'] = '/snap/bin/chromium'
        if os.path.exists('token.json'):
            try:
                creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
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

    def list_videos(self, youtube, channel_id):
        request = youtube.channels().list(part='contentDetails', id=channel_id)
        response = request.execute()
        playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        videos = []
        next_page_token = None
        while True:
            request = youtube.playlistItems().list(part='snippet,status', playlistId=playlist_id, maxResults=50, pageToken=next_page_token)
            response = request.execute()
            videos += response['items']
            next_page_token = response.get('nextPageToken')
            if next_page_token is None:
                break
        return videos

    def update_video_privacy(self, youtube, video_id, privacy_status='unlisted'):
        request_body = {
            'id': video_id,
            'status': {
                'privacyStatus': privacy_status
            }
        }
        request = youtube.videos().update(
            part='status',
            body=request_body
        )
        response = request.execute()
        return response

    def check_video_expiration(self, video):
        title = video['snippet']['title']
        published_at = datetime.datetime.strptime(video['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        match = re.match(r'\[(\d+)j\]', title)
        
        if match:
            days = int(match.group(1))
            delta_days = (datetime.datetime.utcnow() - published_at).days
            days_left = days - delta_days
            return {
                'title': title,
                'video_id': video['snippet']['resourceId']['videoId'],
                'published_at': published_at,
                'expiration_days': days,
                'days_left': days_left,
                'is_expired': days_left <= 0,
                'privacy_status': video['status']['privacyStatus']
            }
        return None

    def run(self, config):
        st.header("Gestion des vidéos temporaires")
        
        channel_id = config['common'].get('channel_id')
        if channel_id:
            credentials = self.get_credentials()
            youtube = build('youtube', 'v3', credentials=credentials)
            
            videos = self.list_videos(youtube, channel_id)
            temp_videos = [self.check_video_expiration(video) for video in videos if self.check_video_expiration(video)]
            
            if temp_videos:
                for video in temp_videos:
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(video['title'])
                    with col2:
                        st.write(f"Publié le: {video['published_at'].strftime('%Y-%m-%d')}")
                    with col3:
                        if video['is_expired']:
                            st.write(f"Expiré depuis {abs(video['days_left'])} jours")
                        else:
                            st.write(f"Expire dans {video['days_left']} jours")
                    with col4:
                        st.write(f"Statut: {video['privacy_status']}")
                
                if st.button("Dépublier les vidéos en dépassement"):
                    expired_videos = [video for video in temp_videos if video['is_expired'] and video['privacy_status'] == 'public']
                    for video in expired_videos:
                        self.update_video_privacy(youtube, video['video_id'])
                    st.success(f"{len(expired_videos)} vidéos ont été dépubliées.")
                    st.experimental_rerun()
            else:
                st.info("Aucune vidéo temporaire trouvée.")
        else:
            st.warning("Veuillez configurer l'ID de la chaîne dans l'onglet Configuration.")
