from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import os
from typing import List, Dict, Any
from lib.youtube_api import YoutubeAPI
import pandas as pd


# Traductions existantes conservées
translations["en"].update({
    "promoteyoutube_tab": "Promote YouTube",
    "promoteyoutube_header": "Promote Content on YouTube (video_list.csv)",
    "promoteyoutube_transcript": "Transcript",
    "promoteyoutube_url": "Video URL",
    "promoteyoutube_keywords": "Keywords to Search",
    "promoteyoutube_search": "Search Videos",
    "promoteyoutube_searching": "Searching videos...",
    "promoteyoutube_max_videos_label": "Number of videos to search",
    "promoteyoutube_keywords_warning": "Please enter keywords to search for videos.",
    "promoteyoutube_export_success": "Exported successfully to {}",
    "promoteyoutube_export_error": "Error during export: {}",
    "promoteyoutube_overwrite_checkbox": "Overwrite existing file",
    "promoteyoutube_save_videos": "Save Video List",
    "promoteyoutube_videos_found": "Videos Found",
})

translations["fr"].update({
    "promoteyoutube_tab": "Promotion YouTube",
    "promoteyoutube_header": "Promouvoir le Contenu sur YouTube (video_list.csv)",
    "promoteyoutube_transcript": "Transcription",
    "promoteyoutube_url": "URL de la vidéo",
    "promoteyoutube_keywords": "Mots-clés à rechercher",
    "promoteyoutube_search": "Rechercher des vidéos",
    "promoteyoutube_searching": "Recherche des vidéos...",
    "promoteyoutube_max_videos_label": "Nombre de vidéos à rechercher",
    "promoteyoutube_keywords_warning": "Veuillez entrer des mots-clés pour la recherche.",
    "promoteyoutube_export_success": "Exporté avec succès vers {}",
    "promoteyoutube_export_error": "Erreur lors de l'export : {}",
    "promoteyoutube_overwrite_checkbox": "Écraser le fichier existant",
    "promoteyoutube_save_videos": "Enregistrer la liste des vidéos",
    "promoteyoutube_videos_found": "Vidéos trouvées",
})


class PromoteyoutubePlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)

    def get_config_fields(self):
        return {
            "max_videos": {
                "type": "number",
                "label": "Maximum Number of Videos to Fetch",
                "default": 10
            },
            "response_prompt": {
                "type": "text",
                "label": "LLM Prompt for Responses",
                "default": """Suggère une réponse à ce commentaire de moins de 500 caractères, en lien avec la vidéo dans l'URL {url} (doit être mentionnée). Le ton est direct, réponds comme si tu étais l'utilisateur, et en t'inspirant du transcript suivant : {transcript}"""
            }
        }

    def get_tabs(self):
        return [
            {"name": t("promoteyoutube_tab"), "plugin": "promoteyoutube"},
            {"name": "Get Comments", "widget": "get_comments"},
            {"name": "Generate Responses", "widget": "generate_response"},
            {"name": "Post Responses", "widget": "post_response"}
        ]

    def promote_content(self, config):
        st.header(t("promoteyoutube_header"))
        work_dir = config['common']['work_directory']

        transcript_path = os.path.join(work_dir, "transcript.txt")
        transcript = st.text_area(
            t("promoteyoutube_transcript"),
            value=open(transcript_path, 'r').read() if os.path.exists(transcript_path) else "",
            height=200,
            key="promo_transcript",
            disabled=os.path.exists(transcript_path)
        )

        url_path = os.path.join(work_dir, "url.txt")
        url = st.text_input(
            t("promoteyoutube_url"),
            value=open(url_path, 'r').read().strip() if os.path.exists(url_path) else "",
            key="promo_url",
            disabled=os.path.exists(url_path)
        )

        from widgets.search_youtube import SearchYoutubeWidget
        SearchYoutubeWidget("searchyoutube", "syw", plugin_manager=self.plugin_manager).display(config)

    def get_comments(self, config):
        from widgets.get_comments import GetCommentsWidget
        GetCommentsWidget("promoteyoutube", "gcw",
                          plugin_manager=self.plugin_manager).display()

    def generate_responses(self, config):
        from widgets.generate_response import GenerateResponseWidget
        GenerateResponseWidget("promoteyoutube", "grw",
                               plugin_manager=self.plugin_manager).display()

    def post_responses(self, config):
        from widgets.post_response import PostResponseWidget
        PostResponseWidget("promoteyoutube", "prw",
                           plugin_manager=self.plugin_manager).display()

    def display_responses(self, config):
        from widgets.yt_responses import ResponseDBDisplayWidget
        ResponseDBDisplayWidget("response_db_display",
                                "marketyoutube", plugin_manager=self.plugin_manager).display()

    def run(self, config):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [t("promoteyoutube_tab"), "Get Comments", "Generate Responses", "Post Responses", "Responses (DB)"])
        with tab1:
            from widgets.recentvideos import RecentVideosWidget
            widget = RecentVideosWidget(
                "recentvideos", "rvw", self.plugin_manager)
            widget.display_simple(config)
            self.promote_content(config)
        with tab2:
            self.get_comments(config)
        with tab3:
            self.generate_responses(config)
        with tab4:
            self.post_responses(config)
        with tab5:
            self.display_responses(config)
