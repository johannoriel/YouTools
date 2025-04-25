from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
import os
from lib.youtube_api import YoutubeAPI
from datetime import datetime

translations["en"].update({
    "get_comments_title": "Get Comments from Videos (video_list.csv -> comment_list.csv)",
    "no_file_error": "No valid video_list CSV found. Please run a video search first.",
    "fetch_comments": "Fetch Comments",
    "fetching_comments": "Fetching comments for selected videos...",
    "adjust_comments": "Number of comments per video",
    "comment_order": "Comment order",
    "export_comments": "Export Comments",
    "export_success": "Comments exported successfully to {filename}",
    "export_error": "Error during export: {error}",
    "select_video_file": "Select Video List Files",
    "overwrite_comments_checkbox": "Overwrite existing comment file",
    "comments_found": "Comments Found",
})

translations["fr"].update({
    "get_comments_title": "Récupérer les commentaires des vidéos (video_list.csv -> comment_list.csv)",
    "no_file_error": "Aucun fichier video_list CSV valide trouvé. Veuillez d'abord effectuer une recherche de vidéos.",
    "fetch_comments": "Récupérer les commentaires",
    "fetching_comments": "Récupération des commentaires pour les vidéos sélectionnées...",
    "adjust_comments": "Nombre de commentaires par vidéo",
    "comment_order": "Ordre des commentaires",
    "export_comments": "Exporter les commentaires",
    "export_success": "Commentaires exportés avec succès vers {filename}",
    "export_error": "Erreur lors de l'export : {error}",
    "select_video_file": "Sélectionner les fichiers de liste de vidéos",
    "overwrite_comments_checkbox": "Écraser le fichier de commentaires existant",
    "comments_found": "Commentaires trouvés",
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

    def export_comments(self, comments, work_dir, overwrite):
        try:
            base_filename = "comment_list.csv"
            output_path = os.path.join(work_dir, base_filename)

            if not overwrite and os.path.exists(output_path):
                i = 1
                while True:
                    new_filename = f"comment_list_{i:03d}.csv"
                    new_output_path = os.path.join(work_dir, new_filename)
                    if not os.path.exists(new_output_path):
                        output_path = new_output_path
                        break
                    i += 1

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

        # Recherche des fichiers CSV commençant par "video_list"
        video_files = [f for f in os.listdir(work_dir) if f.startswith(
            'video_list') and f.endswith('.csv')]
        if not video_files:
            st.error(t("no_file_error"))
            return

        # Sélection multiple des fichiers vidéo
        selected_video_files = st.multiselect(
            t("select_video_file"),
            options=video_files,
            default=[video_files[0]] if video_files else [],
            key=f"{self.prefix}_select_video_file"
        )

        if not selected_video_files:
            st.warning("Please select at least one video list file.")
            return

        # Charger et concaténer les fichiers sélectionnés
        dfs = []
        required_columns = {'video_id', 'title', 'url',
                            'channel_id', 'channel_title', 'published_at'}
        for video_file in selected_video_files:
            file_path = os.path.join(work_dir, video_file)
            try:
                df = pd.read_csv(file_path)
                if required_columns.issubset(df.columns):
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            st.error(t("no_file_error"))
            return

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset='video_id', keep='first')

        # Calculer l'ancienneté en jours
        current_date = datetime.now()
        combined_df['published_at'] = pd.to_datetime(
            combined_df['published_at']).dt.tz_localize(None)
        combined_df['age_days'] = (
            current_date - combined_df['published_at']).dt.days

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
            "age_days": st.column_config.NumberColumn("Age (days)", width="small"),
        }

        # Afficher le DataFrame avec sélection multi-lignes
        selected_rows = st.dataframe(
            combined_df,
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
                selected_videos = [combined_df.iloc[i].to_dict()
                                   for i in selected_rows['selection']['rows']]
                comments = self.fetch_comments(
                    selected_videos, max_comments_per_video, comment_order)
                # Stocker dans st.session_state
                st.session_state['found_comments'] = comments

        # Affichage des commentaires récupérés
        if 'found_comments' in st.session_state and st.session_state['found_comments']:
            st.subheader(t("comments_found"))
            comments_df = pd.DataFrame([
                {
                    'comment_text': comment.get('text', ''),
                    'author': comment.get('author', ''),
                    'video_title': comment.get('video_title', ''),
                    'channel_title': comment.get('channel_title', ''),
                    'like_count': comment.get('like_count', 0),
                    'published_at': comment.get('published_at', ''),
                    'comment_url': f"https://www.youtube.com/watch?v={comment.get('video_id', '')}&lc={comment.get('id', '')}",
                }
                for comment in st.session_state['found_comments']
            ])

            comment_column_config = {
                "comment_text": st.column_config.TextColumn("Comment", width="large"),
                "author": st.column_config.TextColumn("Author", width="medium"),
                "video_title": st.column_config.TextColumn("Video Title", width="large"),
                "channel_title": st.column_config.TextColumn("Channel", width="medium"),
                "like_count": st.column_config.NumberColumn("Likes", width="small"),
                "published_at": st.column_config.TextColumn("Published At", width="medium"),
                "comment_url": st.column_config.LinkColumn(
                    "Comment URL",
                    help="Click to view comment",
                    display_text="View",
                    width="small"
                ),
            }

            st.dataframe(
                comments_df,
                column_config=comment_column_config,
                use_container_width=True,
                height=400,
                key=f"{self.prefix}_comments_dataframe"
            )

            # Case à cocher pour écraser le fichier
            overwrite_comments = st.checkbox(
                t("overwrite_comments_checkbox"), key=f"{self.prefix}_overwrite_comments")

            # Bouton pour exporter les commentaires
            if st.button(t("export_comments"), key=f"{self.prefix}_export_comments"):
                self.export_comments(
                    st.session_state['found_comments'], work_dir, overwrite_comments)
