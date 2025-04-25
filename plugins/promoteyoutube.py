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

    def search_videos(self, keywords: str, max_videos: int, video_order: str):
        youtube_api = YoutubeAPI(self.plugin_manager.config)
        videos = youtube_api.search_videos(
            keywords,
            max_videos,
            order=video_order,
            language=st.session_state.lang
        )
        return videos

    def export_videos(self, videos, work_dir, overwrite):
        try:
            base_filename = "video_list.csv"
            output_path = os.path.join(work_dir, base_filename)

            if not overwrite and os.path.exists(output_path):
                i = 1
                while True:
                    new_filename = f"video_list_{i:03d}.csv"
                    new_output_path = os.path.join(work_dir, new_filename)
                    if not os.path.exists(new_output_path):
                        output_path = new_output_path
                        break
                    i += 1

            videos_data = [
                {
                    'video_id': video.get('video_id', ''),
                    'title': video.get('title', ''),
                    'url': video.get('url', ''),
                    'channel_id': video.get('channel_id', ''),
                    'channel_title': video.get('channel_title', ''),
                    'view_count': video.get('view_count', 0),
                    'comment_count': video.get('comment_count', 0),
                    'published_at': video.get('published_at', ''),
                    'language': video.get('language', ''),
                    'relevance_score': video.get('relevance_score', 0),
                    'subscriber_count': video.get('subscriber_count', 0),
                    'keywords': st.session_state.get('keywords', '')
                }
                for video in videos
            ]
            df = pd.DataFrame(videos_data)
            df.to_csv(output_path, index=False)
            st.success(t("promoteyoutube_export_success").format(output_path))
        except Exception as e:
            st.error(t("promoteyoutube_export_error").format(str(e)))

    def promote_content(self, config):
        st.header(t("promoteyoutube_header"))
        work_dir = config['common']['work_directory']

        # Entrée pour la transcription
        transcript_path = os.path.join(work_dir, "transcript.txt")
        transcript = st.text_area(
            t("promoteyoutube_transcript"),
            value=open(transcript_path, 'r').read(
            ) if os.path.exists(transcript_path) else "",
            height=200,
            key="promo_transcript",
            disabled=os.path.exists(transcript_path)
        )

        # Entrée pour l'URL
        url_path = os.path.join(work_dir, "url.txt")
        url = st.text_input(
            t("promoteyoutube_url"),
            value=open(url_path, 'r').read().strip(
            ) if os.path.exists(url_path) else "",
            key="promo_url",
            disabled=os.path.exists(url_path)
        )

        # Entrée pour les mots-clés
        keywords = st.text_input(
            t("promoteyoutube_keywords"), key="promo_keywords")

        # Nombre maximum de vidéos
        max_videos = st.number_input(
            t("promoteyoutube_max_videos_label"),
            min_value=1,
            max_value=50,
            value=int(config['promoteyoutube']['max_videos']),
            key="promo_max_videos"
        )

        # Bouton de recherche
        if st.button(t("promoteyoutube_search"), key="promo_search"):
            if keywords:
                with st.spinner(t("promoteyoutube_searching")):
                    # Stocker pour export
                    st.session_state['keywords'] = keywords
                    videos = self.search_videos(
                        keywords, max_videos, "relevance")
                    # Stocker temporairement pour affichage
                    st.session_state['found_videos'] = videos
            else:
                st.warning(t("promoteyoutube_keywords_warning"))

        # Affichage des vidéos trouvées
        if 'found_videos' in st.session_state and st.session_state['found_videos']:
            st.subheader(t("promoteyoutube_videos_found"))
            videos_df = pd.DataFrame([
                {
                    'title': video.get('title', ''),
                    'url': video.get('url', ''),
                    'channel_title': video.get('channel_title', ''),
                    'view_count': video.get('view_count', 0),
                    'comment_count': video.get('comment_count', 0),
                    'language': video.get('language', ''),
                    'relevance_score': video.get('relevance_score', 0),
                    'subscriber_count': video.get('subscriber_count', 0),
                }
                for video in st.session_state['found_videos']
            ])

            column_config = {
                "title": st.column_config.TextColumn("Title", width="large"),
                "url": st.column_config.LinkColumn(
                    "Video URL",
                    help="Click to visit the video",
                    display_text="Visit",
                    width="small"
                ),
                "channel_title": st.column_config.TextColumn("Channel", width="medium"),
                "view_count": st.column_config.NumberColumn("Views", width="small"),
                "comment_count": st.column_config.NumberColumn("Comments", width="small"),
                "language": st.column_config.TextColumn("Language", width="small"),
                "relevance_score": st.column_config.NumberColumn("Relevance", width="small"),
                "subscriber_count": st.column_config.NumberColumn("Subscribers", width="small"),
            }

            st.dataframe(
                videos_df,
                column_config=column_config,
                use_container_width=True,
                height=400,
                key="promo_videos_dataframe"
            )

            # Case à cocher pour écraser le fichier
            overwrite = st.checkbox(
                t("promoteyoutube_overwrite_checkbox"), key="promo_overwrite")

            # Bouton pour sauvegarder
            if st.button(t("promoteyoutube_save_videos"), key="promo_save_videos"):
                self.export_videos(
                    st.session_state['found_videos'], work_dir, overwrite)

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

    def run(self, config):
        tab1, tab2, tab3, tab4 = st.tabs(
            [t("promoteyoutube_tab"), "Get Comments", "Generate Responses", "Post Responses"])
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
