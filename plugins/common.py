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

    def get_config_ui(self, config):
        updated_config = {}
        #updated_config['separator_common'] = st.header('Common')
        for field, params in self.get_config_fields().items():
            if params['type'] == 'select':
                updated_config[field] = st.selectbox(
                    params['label'],
                    options=[option[0] for option in params['options']],
                    format_func=lambda x: dict(params['options'])[x],
                    index=[option[0] for option in params['options']].index(config.get(field, params['default']))
                )
            elif params['type'] == 'textarea':
                updated_config[field] = st.text_area(
                    params['label'],
                    value=config.get(field, params['default'])
                )
            else:
                updated_config[field] = st.text_input(
                    params['label'],
                    value=config.get(field, params['default']),
                    type="password" if field == "?" else "default"
                )        
        return updated_config

    def get_tabs(self):
        return [{"name": "Commun", "plugin": "common"}]

    def run(self, config):
        st.header("Common Plugin")
        st.write(f"Channel: {config['common']['channel_id']}")
        st.write(f"Répertoire de travail: {config['common']['work_directory']}")

