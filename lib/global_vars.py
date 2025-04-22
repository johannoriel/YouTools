# global_vars.py
import streamlit as st

def get_lang():
    return st.session_state.get('lang', 'en')

def set_lang(new_lang):
    st.session_state['lang'] = new_lang

# Dictionnaire de traduction global
translations = {
    "en": {
        "page_title": "YoutTools : YouTube Channel helpers",
        "navigation": "Navigation",
        "configurations": "Configurations",
        "save_button": "Save Configuration",
        "success_message": "Configuration saved!",
    },
    "fr": {
        "page_title": "YoutTools : Outils pour chaîne YouTube",
        "navigation": "Navigation",
        "configurations": "Configurations",
        "save_button": "Sauvegarder la configuration",
        "success_message": "Configuration sauvegardée!",
    }
}

# Fonction de traduction
def t(key: str) -> str:
    return translations[get_lang()].get(key, key)
