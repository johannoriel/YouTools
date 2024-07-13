from global_vars import t, translations
from app import Plugin
import streamlit as st

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "channel_id": "YouTube Channel ID",
    "work_directory": "Work Directory",
    "preferred_language": "Preferred Language for Transcriptions",
})
translations["fr"].update({
    "channel_id": "ID de la chaîne YouTube",
    "work_directory": "Répertoire de travail",
    "preferred_language": "Langue préférée pour les transcriptions",
})

class CommonPlugin(Plugin):
    def get_config_fields(self):
        global lang
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
        global lang
        st.header("Common Plugin")
        st.write(f"Channel: {config['common']['channel_id']}")
        st.write(f"{t('work_directory')}: {config['common']['work_directory']}")

import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import RefreshError
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
