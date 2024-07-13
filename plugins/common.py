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

