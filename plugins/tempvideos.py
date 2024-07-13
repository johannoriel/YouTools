from global_vars import translations, t
import streamlit as st
from app import Plugin
from plugins.common import get_credentials
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import RefreshError

import os
import datetime
import re

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "temp_videos_tab": "Temporary Videos",
    "temp_videos_header": "Managing Temporary Videos",
    "temp_videos_published_on": "Published on:",
    "temp_videos_expired_days": "Expired for {days} days",
    "temp_videos_expires_in_days": "Expires in {days} days",
    "temp_videos_status": "Status:",
    "temp_videos_no_temp_videos": "No temporary videos found.",
    "temp_videos_unpublish_button": "Unpublish Expired Videos",
    "temp_videos_unpublish_success": "{count} videos have been unpublished.",
    "temp_videos_configure_channel_id": "Please configure the channel ID in the Configuration tab."
})

translations["fr"].update({
    "temp_videos_tab": "Vidéos temporaires",
    "temp_videos_header": "Gestion des vidéos temporaires",
    "temp_videos_published_on": "Publié le :",
    "temp_videos_expired_days": "Expiré depuis {days} jours",
    "temp_videos_expires_in_days": "Expire dans {days} jours",
    "temp_videos_status": "Statut :",
    "temp_videos_no_temp_videos": "Aucune vidéo temporaire trouvée.",
    "temp_videos_unpublish_button": "Dépublier les vidéos en dépassement",
    "temp_videos_unpublish_success": "{count} vidéos ont été dépubliées.",
    "temp_videos_configure_channel_id": "Veuillez configurer l'ID de la chaîne dans l'onglet Configuration."
})

class TempvideosPlugin(Plugin):

    def get_tabs(self):
        return [{"name": t("temp_videos_tab"), "plugin": "tempvideos"}]

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
        st.header(t("temp_videos_header"))
        
        channel_id = config['common'].get('channel_id')
        if channel_id:
            credentials = get_credentials()
            youtube = build('youtube', 'v3', credentials=credentials)
            
            videos = self.list_videos(youtube, channel_id)
            temp_videos = [self.check_video_expiration(video) for video in videos if self.check_video_expiration(video)]
            
            if temp_videos:
                for video in temp_videos:
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(video['title'])
                    with col2:
                        st.write(f"{t('temp_videos_published_on')} {video['published_at'].strftime('%Y-%m-%d')}")
                    with col3:
                        if video['is_expired']:
                            st.write(t('temp_videos_expired_days').format(days=abs(video['days_left'])))
                        else:
                            st.write(t('temp_videos_expires_in_days').format(days=video['days_left']))
                    with col4:
                        st.write(f"{t('temp_videos_status')} {video['privacy_status']}")
                
                if st.button(t("temp_videos_unpublish_button")):
                    expired_videos = [video for video in temp_videos if video['is_expired'] and video['privacy_status'] == 'public']
                    for video in expired_videos:
                        self.update_video_privacy(youtube, video['video_id'])
                    st.success(t("temp_videos_unpublish_success").format(count=len(expired_videos)))
                    st.experimental_rerun()
            else:
                st.info(t("temp_videos_no_temp_videos"))
        else:
            st.warning(t("temp_videos_configure_channel_id"))

