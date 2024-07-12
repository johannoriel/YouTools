from app import Plugin
import streamlit as st

class CommonPlugin(Plugin):
        
    def get_config_fields(self):
        return {
            "channel_id": {
                "type": "text",
                "label": "ID de la chaîne YouTube",
                "default": ""
            },
            "work_directory": {
                "type": "text",
                "label": "Répertoire de travail",
                "default": "/home/joriel/Vidéos"
            },            
            "language": {
                "type": "select",
                "label": "Langue préférée pour les transcriptions",
                "options": [("fr", "Français"), ("en", "Anglais")],
                "default": "fr"
            },
        }

    def get_tabs(self):
        return [{"name": "Commun", "plugin": "common"}]

    def run(self, config):
        st.header("Common Plugin")
        st.write(f"Channel: {config['common']['channel_id']}")
        st.write(f"Répertoire de travail: {config['common']['work_directory']}")

