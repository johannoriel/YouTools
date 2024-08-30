from global_vars import t, translations
from app import Plugin
import streamlit as st
import torch

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
        torch.cuda.empty_cache()
        st.write("CUDA memory reset")

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
                'client_secret.json', SCOPES)
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
        if file.lower().endswith(extensions):
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
        if file.lower().endswith(('.mkv', '.mp4')):
            # Check if the file name contains spaces
            if ' ' in file:
                # Create a new file name by replacing spaces with underscores
                new_file = file.replace(' ', '_')
                old_path = os.path.join(directory, file)
                new_path = os.path.join(directory, new_file)

                # Rename the file
                os.rename(old_path, new_path)

                # Use the new file name for further processing
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
