from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from lib.youtube_api import YoutubeAPI
import re

translations["en"].update({
    "process_videos_title": "Process Video List (searched_video_list.csv -> video_list.csv)",
    "no_file_error": "No valid CSV file found. Please check the directory.",
    "export_button": "Process and Export Videos",
    "export_success": "Processed videos exported to {filename}",
    "overwrite_checkbox": "Overwrite existing file",
})

translations["fr"].update({
    "process_videos_title": "Traiter la liste des vidéos (searched_video_list.csv -> video_list.csv)",
    "no_file_error": "Aucun fichier CSV valide trouvé. Vérifiez le répertoire.",
    "export_button": "Traiter et exporter les vidéos",
    "export_success": "Vidéos traitées exportées vers {filename}",
    "overwrite_checkbox": "Écraser le fichier existant",
})


def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    youtube_regex = (
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    )
    match = re.search(youtube_regex, url)
    return match.group(1) if match else "N/A"


class ProcessVideosWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.youtube_api = YoutubeAPI(
            plugin_manager.config if plugin_manager else {})

    def extract_video_metadata_yt_dlp(self, video_url, debug=False):
        """Extract video metadata using yt-dlp"""
        try:
            import yt_dlp
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "force_generic_extractor": False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

                channel_id = info.get("channel_id", "N/A")
                channel_title = info.get("channel", "N/A")
                subscriber_count = info.get("channel_follower_count", "N/A")
                view_count = info.get("view_count", "N/A")
                comment_count = info.get("comment_count", "N/A")
                published_at = info.get("upload_date", "N/A")
                description = info.get("description", "N/A")

                if isinstance(subscriber_count, int):
                    subscriber_count = str(subscriber_count)
                else:
                    subscriber_count = ""

                if isinstance(view_count, int):
                    view_count = str(view_count)
                else:
                    view_count = ""

                if isinstance(comment_count, int):
                    comment_count = str(comment_count)
                else:
                    comment_count = ""

                if published_at != "N/A":
                    try:
                        published_at = f"{published_at[:4]}-{published_at[4:6]}-{published_at[6:8]}"
                    except:
                        published_at = ""

                if debug:
                    st.write(f"Extracted from {video_url}: channel_id={channel_id}, channel_title={channel_title}, "
                             f"subscriber_count={subscriber_count}, view_count={view_count}, "
                             f"comment_count={comment_count}, published_at={published_at}, "
                             f"description={description}")

                return {
                    "channel_id": channel_id,
                    "channel_title": channel_title,
                    "subscriber_count": subscriber_count,
                    "view_count": view_count,
                    "comment_count": comment_count,
                    "published_at": published_at,
                    "description": description
                }

        except Exception as e:
            if debug:
                st.write(
                    f"Error extracting video metadata with yt-dlp for {video_url}: {str(e)}")
            return {
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "view_count": "",
                "comment_count": "",
                "published_at": "",
                "description": "N/A"
            }

    def display(self):
        st.title(t("process_videos_title"))
        work_directory = self.work_dir()

        # Recherche des fichiers CSV commençant par "searched_video_list"
        csv_files = [f for f in os.listdir(work_directory) if f.startswith(
            'searched_video_list') and f.endswith('.csv')]

        if not csv_files:
            st.error(t("no_file_error"))
            return

        # Colonnes attendues
        required_columns = {'keyword', 'url', 'video_id', 'title', 'language'}

        # Charger et concaténer les fichiers CSV valides
        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(work_directory, csv_file)
            try:
                df = pd.read_csv(file_path)
                if required_columns.issubset(set(df.columns)):
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            st.error(t("no_file_error"))
            return

        combined_df = pd.concat(dfs, ignore_index=True)

        # Configurer les colonnes pour l'affichage
        column_config = {
            "keyword": st.column_config.TextColumn("Keyword", width="medium"),
            "url": st.column_config.LinkColumn(
                "Video URL",
                help="Click to visit the video",
                display_text="Visit",
                width="small"
            ),
            "video_id": st.column_config.TextColumn("Video ID", width="medium"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "language": st.column_config.TextColumn("Language", width="small"),
        }

        # Afficher le DataFrame avec sélection multi-lignes
        selected_rows = st.dataframe(
            combined_df,
            column_config=column_config,
            width='stretch',
            height=400,
            selection_mode="multi-row",
            on_select="rerun",
            key=f"{self.prefix}_video_selector"
        )

        # Checkbox pour écraser ou renommer
        overwrite = st.checkbox(t("overwrite_checkbox"),
                                value=True, key=f"{self.prefix}_overwrite")

        # Bouton pour traiter et exporter
        if st.button(t("export_button"), key=f"{self.prefix}_export"):
            if not selected_rows['selection']['rows']:
                st.warning("Please select at least one video to process.")
                return

            selected_df = combined_df.iloc[selected_rows['selection']['rows']]
            debug_mode = st.session_state.get("debug_mode", False)

            # Ajouter une barre de progression
            progress_bar = st.progress(0)
            total_videos = len(selected_df)
            processed_videos = 0

            # Ajouter les métadonnées
            processed_data = []
            for _, row in selected_df.iterrows():
                metadata = self.extract_video_metadata_yt_dlp(
                    row['url'], debug=debug_mode)
                published_at = metadata['published_at']
                relevance_score = 0

                if published_at and metadata['subscriber_count'] and metadata['comment_count']:
                    try:
                        if published_at:
                            date_obj = datetime.strptime(
                                published_at, "%Y-%m-%d")
                            published_at_iso = date_obj.strftime(
                                "%Y-%m-%dT00:00:00Z")
                        else:
                            published_at_iso = None

                        if published_at_iso:
                            video_data = {
                                "published_at": published_at_iso,
                                "subscriber_count": int(metadata['subscriber_count'] or 0),
                                "comment_count": int(metadata['comment_count'] or 0)
                            }
                            relevance_score = self.youtube_api.calculate_relevance_score(
                                video_data)
                    except Exception as e:
                        if debug_mode:
                            st.write(
                                f"Error calculating relevance score for {row['url']}: {str(e)}")

                processed_data.append({
                    "keyword": row['keyword'],
                    "url": row['url'],
                    "video_id": row['video_id'],
                    "title": row['title'],
                    "view_count": metadata['view_count'],
                    "language": row['language'],
                    "published_at": metadata['published_at'],
                    "channel_id": metadata['channel_id'],
                    "channel_title": metadata['channel_title'],
                    "subscriber_count": metadata['subscriber_count'],  # Corrigé de 'subscription_count' à 'subscriber_count'
                    "comment_count": metadata['comment_count'],
                    "description": metadata['description'],
                    "relevance_score": relevance_score
                })

                # Mettre à jour la barre de progression
                processed_videos += 1
                progress_bar.progress(processed_videos / total_videos)

            processed_df = pd.DataFrame(processed_data)

            # Sauvegarder le fichier
            base_filename = "video_list.csv"
            export_path = os.path.join(work_directory, base_filename)

            if not overwrite and os.path.exists(export_path):
                i = 1
                while True:
                    new_filename = f"video_list_{i:03d}.csv"
                    new_export_path = os.path.join(
                        work_directory, new_filename)
                    if not os.path.exists(new_export_path):
                        export_path = new_export_path
                        break
                    i += 1

            processed_df.to_csv(export_path, index=False)
            st.success(t("export_success").format(
                filename=os.path.basename(export_path)))
