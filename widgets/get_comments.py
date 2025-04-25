from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
import os
from lib.youtube_api import YoutubeAPI

translations["en"].update({
    "get_comments_title": "Get Comments from Videos",
    "no_file_error": "No valid video_list.csv found. Please run a video search first.",
    "fetch_comments": "Fetch Comments",
    "fetching_comments": "Fetching comments for selected videos...",
    "adjust_comments": "Number of comments per video",
    "comment_order": "Comment order",
    "export_comments": "Export Comments to comment_list.csv",
    "export_success": "Comments exported successfully to {filename}",
    "export_error": "Error during export: {error}",
})

translations["fr"].update({
    "get_comments_title": "Récupérer les commentaires des vidéos",
    "no_file_error": "Aucun fichier video_list.csv valide trouvé. Veuillez d'abord effectuer une recherche de vidéos.",
    "fetch_comments": "Récupérer les commentaires",
    "fetching_comments": "Récupération des commentaires pour les vidéos sélectionnées...",
    "adjust_comments": "Nombre de commentaires par vidéo",
    "comment_order": "Ordre des commentaires",
    "export_comments": "Exporter les commentaires vers comment_list.csv",
    "export_success": "Commentaires exportés avec succès vers {filename}",
    "export_error": "Erreur lors de l'export : {error}",
})


class GetCommentsWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def fetch_comments(self, videos, max_comments, comment_order):
        youtube_api = YoutubeAPI(self.plugin_manager.config)
        comments = []
        for video in videos:
            video_comments = youtube_api.get_comments(
                video['video_id'], max_comments, order=comment_order)
            for comment in video_comments:
                comment['video_title'] = video['title']
                comment['channel_title'] = video['channel_title']
                comment['video_id'] = video['video_id']
                comment['channel_id'] = video.get('channel_id', 'unknown')
            comments.extend(video_comments)
        return comments

    def export_comments(self, comments, work_dir):
        try:
            output_path = os.path.join(work_dir, "comment_list.csv")
            comments_data = [
                {
                    'comment_id': comment.get('id', ''),
                    'comment_text': comment.get('text', ''),
                    'author': comment.get('author', ''),
                    'published_at': comment.get('published_at', ''),
                    'like_count': comment.get('like_count', 0),
                    'video_id': comment.get('video_id', ''),
                    'video_title': comment.get('video_title', ''),
                    'channel_id': comment.get('channel_id', ''),
                    'channel_title': comment.get('channel_title', ''),
                    'comment_url': f"https://www.youtube.com/watch?v={comment.get('video_id', '')}&lc={comment.get('id', '')}",
                    'keywords': comment.get('keywords', '')
                }
                for comment in comments
            ]
            df = pd.DataFrame(comments_data)
            df.to_csv(output_path, index=False)
            st.success(t("export_success").format(filename=output_path))
        except Exception as e:
            st.error(t("export_error").format(str(e)))

    def display(self):
        st.title(t("get_comments_title"))
        work_dir = self.plugin_manager.config["common"]["work_directory"]
        video_file = os.path.join(work_dir, "video_list.csv")

        if not os.path.exists(video_file):
            st.error(t("no_file_error"))
            return

        # Charger le fichier video_list.csv
        df = pd.read_csv(video_file)
        required_columns = {'video_id', 'title',
                            'url', 'channel_id', 'channel_title'}
        if not required_columns.issubset(df.columns):
            st.error(t("no_file_error"))
            return

        # Configurer les colonnes pour l'affichage
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

        # Afficher le DataFrame avec sélection multi-lignes
        selected_rows = st.dataframe(
            df,
            column_config=column_config,
            use_container_width=True,
            height=400,
            selection_mode="multi-row",
            on_select="rerun",
            key=f"{self.prefix}_video_dataframe"
        )

        # Paramètres pour la récupération des commentaires
        max_comments_per_video = st.number_input(
            t("adjust_comments"),
            min_value=1,
            max_value=10,
            value=2,
            key=f"{self.prefix}_max_comments_per_video"
        )
        comment_order = st.selectbox(
            t("comment_order"),
            options=["relevance", "time"],
            index=1,
            key=f"{self.prefix}_comment_order"
        )

        # Bouton pour récupérer les commentaires
        if st.button(t("fetch_comments"), key=f"{self.prefix}_fetch_comments") and selected_rows['selection']['rows']:
            with st.spinner(t("fetching_comments")):
                selected_videos = [df.iloc[i].to_dict()
                                   for i in selected_rows['selection']['rows']]
                comments = self.fetch_comments(
                    selected_videos, max_comments_per_video, comment_order)
                self.export_comments(comments, work_dir)
