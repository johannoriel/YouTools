import os
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from lib.global_vars import t, translations
from app import Plugin, unit_test
import streamlit as st
import torch
import pandas as pd  # For test result tables

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "channel_id": "YouTube Channel ID",
    "work_directory": "Work Directory",
    "test_directory": "Test Directory",
    "test_mode": "Test Mode",
    "run_tests": "Run Tests",
    "test_results": "Test Results",
    "no_tests_run": "No tests have been run yet. Click 'Run Tests' in the sidebar.",
    "preferred_language": "Preferred Language for Transcriptions",
    "upload_finished": "Upload finished ! ID of the video",
    "notification_sound": "Notification sound file",
    "play_sound": "Play notification sound",
    "test_header": "Test Plugin - Simple Addition",
    "test_num1_label": "Enter first number",
    "test_num2_label": "Enter second number",
    "test_add_button": "Add Numbers",
    "test_result": "Result: {result}",
})

translations["fr"].update({
    "channel_id": "ID de la chaîne YouTube",
    "work_directory": "Répertoire de travail",
    "test_directory": "Répertoire de test",
    "test_mode": "Mode de test",
    "run_tests": "Exécuter les tests",
    "test_results": "Résultats des Tests",
    "no_tests_run": "Aucun test n'a été lancé. Cliquez sur 'Lancer les Tests' dans la barre latérale.",
    "preferred_language": "Langue préférée pour les transcriptions",
    "upload_finished": "Upload terminé ! ID de la vidéo",
    "notification_sound": "Fichier son de notification",
    "play_sound": "Jouer le son de notification",
    "test_header": "Plugin de Test - Addition Simple",
    "test_num1_label": "Entrez le premier nombre",
    "test_num2_label": "Entrez le second nombre",
    "test_add_button": "Ajouter les Nombres",
    "test_result": "Résultat : {result}",
})

yt_categories = {
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

def get_category_id(category_name):
    for id, name in yt_categories.items():
        if name.lower() == category_name.lower():
            return id
    return "22"  # Default to "People & Blogs" if not found

class CommonPlugin(Plugin):
    def get_config_fields(self):
        return {
            "work_directory": {
                "type": "text",
                "label": t("work_directory"),
                "default": "/home/joriel/Vidéos"
            },
            "test_directory": {
                "type": "text",
                "label": t("test_directory"),
                "default": "/home/joriel/Vidéos/Test"
            },
            "notification_sound": {
                "type": "text",
                "label": t("notification_sound"),
                "default": ""
            },
            "channel_id": {
                "type": "text",
                "label": t("channel_id"),
                "default": ""
            },
            "project_number": {
                "type": "text",
                "label": "YouTube project number",
                "default": ""
            },
            "youtube_api_key": {
                "type": "text",
                "label": "YouTube API Key",
                "default": ""
            },
            "language": {
                "type": "select",
                "label": t("preferred_language"),
                "options": [("fr", "Français"), ("en", "Anglais")],
                "default": "fr"
            },
            "twitter_bearer_token": {
                "type": "text",
                "label": "Twitter Bearer Token",
                "default": ""
            },
            "twitter_api_key": {
                "type": "text",
                "label": "Twitter API Key",
                "default": ""
            },
            "twitter_api_secret": {
                "type": "text",
                "label": "Twitter API Secret",
                "default": ""
            },
            "twitter_access_token": {
                "type": "text",
                "label": "Twitter Access Token",
                "default": ""
            },
            "twitter_access_token_secret": {
                "type": "text",
                "label": "Twitter Access Token Secret",
                "default": ""
            },
            "bluesky_handle": {
                "type": "text",
                "label": "Bluesky Handle",
                "default": ""
            },
            "bluesky_password": {
                "type": "text",
                "label": "Bluesky App Password",
                "default": ""
            },
            "telegram_bot_token": {
                "type": "text",
                "label": "Telegram Bot Token",
                "default": ""
            },
            "telegram_channel_id": {
                "type": "text",
                "label": "Telegram Channel ID",
                "default": ""
            },
            "ghost_url": {
                "type": "text",
                "label": "Ghost URL",
                "default": ""
            },
            "ghost_api_key": {
                "type": "text",
                "label": "Ghost API Key",
                "default": ""
            },
            "twitter_api_v1_enabled": {
                "type": "checkbox",
                "label": "Enable Twitter API v1",
                "default": False
            },
            "twitter_api_v1_consumer_key": {
                "type": "text",
                "label": "Twitter API v1 Consumer Key",
                "default": ""
            },
            "twitter_api_v1_consumer_secret": {
                "type": "text",
                "label": "Twitter API v1 Consumer Secret",
                "default": ""
            },
            "twitter_api_v1_access_token": {
                "type": "text",
                "label": "Twitter API v1 Access Token",
                "default": ""
            },
            "twitter_api_v1_access_token_secret": {
                "type": "text",
                "label": "Twitter API v1 Access Token Secret",
                "default": ""
            },
            "linkedin_client_id": {
                "type": "text",
                "label": "Linkedin client ID",
                "default": ""
            },
            "linkedin_client_secret": {
                "type": "text",
                "label": "Linkedin client secret",
                "default": ""
            },
            "linkedin_access_token": {
                "type": "text",
                "label": "Linkedin access token",
                "default": ""
            },
            "hashnode_api_token": {
                "type": "text",
                "label": "Hashnode API token",
                "default": ""
            },
            "hashnode_default_host": {
                "type": "text",
                "label": "Hashnode default host",
                "default": ""
            },
            "wordpress_client_id": {
                "type": "text",
                "label": "Wordpress client id",
                "default": ""
            },
            "wordpress_client_secret": {
                "type": "text",
                "label": "Wordpress client secret",
                "default": ""
            },
            "wordpress_site_id": {
                "type": "text",
                "label": "Wordpress site id",
                "default": ""
            },
            "substack_email": {
                "type": "text",
                "label": "Substack Email",
                "default": "your_email@example.com"
            },
            "substack_password": {
                "type": "password",
                "label": "Substack Password",
                "default": "your_password"
            },
            "substack_publication_url": {
                "type": "text",
                "label": "Substack Publication URL",
                "default": "https://your-publication.substack.com"
            },
        }

    def get_tabs(self):
        return [{"name": "Commun", "plugin": "common"}]

    def has_tests(self) -> bool:
        return True

    def add_numbers(self, a: int, b: int) -> int:
        """Testable method for simple addition."""
        return a + b

    def run(self, config):
        st.header(t("test_header"))

        num1 = st.number_input(t("test_num1_label"), value=0)
        num2 = st.number_input(t("test_num2_label"), value=0)

        if st.button(t("test_add_button")):
            result = self.add_numbers(num1, num2)
            st.success(t("test_result").format(result=result))

        # Original common plugin functionality
        st.header("Common Plugin")
        st.write(f"Channel: {config['common']['channel_id']}")
        st.write(f"{t('work_directory')}: {config['common']['work_directory']}")
        torch.cuda.empty_cache()
        st.write("CUDA memory reset")
        self.play_notification_sound()

    def play_notification_sound(self):
        """Joue un son MP3 prédéfini si configuré"""
        sound_file = os.path.expanduser(self.get_config('notification_sound'))
        if sound_file and os.path.exists(sound_file):
            st.audio(sound_file, format='audio/mp3', autoplay=True)
        elif sound_file:
            st.warning(f"Sound file not found: {sound_file}")

    @unit_test
    def test_add_basic(self):
        assert self.add_numbers(2, 3) == 5, "2 + 3 should equal 5"

    @unit_test
    def test_add_zero(self):
        assert self.add_numbers(0, 0) == 0, "0 + 0 should equal 0"

SCOPES = [
    'https://www.googleapis.com/auth/youtube.force-ssl',
    'https://www.googleapis.com/auth/yt-analytics.readonly',
    'https://www.googleapis.com/auth/yt-analytics-monetary.readonly',
    'https://www.googleapis.com/auth/cloud-platform.read-only',
    'https://www.googleapis.com/auth/monitoring.read',
]

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
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds

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

    st.success(t('upload_finished')+f" : {response['id']}")
    return response['id']

def list_video_files2(directory, prefix_exclude=None, extensions=('.mkv', '.mp4')):
    def rename_file_without_spaces(file, directory):
        if ' ' in file:
            new_file = file.replace(' ', '_')
            old_path = os.path.join(directory, file)
            new_path = os.path.join(directory, new_file)
            os.rename(old_path, new_path)
            return new_file
        return file

    video_files = []
    for file in os.listdir(directory):
        if file.lower().endswith(tuple(extensions)):
            file = rename_file_without_spaces(file, directory)
            if prefix_exclude:
                if not any(file.startswith(prefix) for prefix in prefix_exclude):
                    full_path = os.path.join(directory, file)
                    mod_time = os.path.getmtime(full_path)
                    video_files.append((file, full_path, mod_time))
            else:
                full_path = os.path.join(directory, file)
                mod_time = os.path.getmtime(full_path)
                video_files.append((file, full_path, mod_time))

    video_files.sort(key=lambda x: x[2], reverse=True)
    return video_files

def list_video_files(directory):
    video_files = []
    outfile_videos = []
    chroma_videos = []
    short_videos = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.mkv', '.mp4', '.mov', '.ogg')):
            if ' ' in file:
                new_file = file.replace(' ', '_')
                old_path = os.path.join(directory, file)
                new_path = os.path.join(directory, new_file)
                os.rename(old_path, new_path)
                file = new_file

            full_path = os.path.join(directory, file)
            mod_time = os.path.getmtime(full_path)
            if file.startswith('outfile_'):
                outfile_videos.append((file, full_path, mod_time))
            elif file.startswith('chroma_'):
                chroma_videos.append((file, full_path, mod_time))
            elif file.startswith('short'):
                short_videos.append((file, full_path, mod_time))
            else:
                video_files.append((file, full_path, mod_time))

    video_files.sort(key=lambda x: x[2], reverse=True)
    outfile_videos.sort(key=lambda x: x[2], reverse=True)
    chroma_videos.sort(key=lambda x: x[2], reverse=True)
    short_videos.sort(key=lambda x: x[2], reverse=True)
    return video_files, outfile_videos, chroma_videos, short_videos

def list_all_video_files(directory):
    l1, l2, l3, l4 = list_video_files(directory)
    return l1+l2+l3+l4

def remove_quotes(s):
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s
