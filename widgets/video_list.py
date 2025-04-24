from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
import os
from datetime import datetime

translations["en"].update({
    "video_list_title": "Video List",
    "no_file_error": "No CSV file found. Please check the directory.",
})

translations["fr"].update({
    "video_list_title": "Liste des vidéos",
    "no_file_error": "Aucun fichier CSV trouvé. Vérifiez le répertoire.",
})


class VideoListWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def display(self):
        st.title(t("video_list_title"))
        work_directory = self.plugin_manager.config["common"]["work_directory"]

        # Recherche des fichiers CSV dans le répertoire de travail
        csv_files = [f for f in os.listdir(
            work_directory) if f.endswith('.csv')]

        if not csv_files:
            st.error(t("no_file_error"))
            return

        # Charger et concaténer tous les fichiers CSV
        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(work_directory, csv_file)
            df = pd.read_csv(file_path)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        # Calculer l'ancienneté en jours
        current_date = datetime.now()
        combined_df['days_old'] = combined_df['published_at'].apply(
            lambda x: (current_date - pd.to_datetime(x,
                       utc=True).tz_localize(None)).days if pd.notnull(x) else ''
        )

        # Créer une URL pour la chaîne
        combined_df['channel_url'] = combined_df['channel_id'].apply(
            lambda x: f'https://www.youtube.com/channel/{x}' if pd.notnull(
                x) else ''
        )

        # Supprimer les colonnes inutiles
        combined_df = combined_df.drop(
            columns=['video_id', 'title_with_url', 'published_at', 'channel_id'], errors='ignore')

        # Configurer les colonnes pour l'affichage
        column_config = {
            "keyword": st.column_config.TextColumn("Keyword", width="medium"),
            "url": st.column_config.LinkColumn(
                "Video URL",
                help="Click to visit the video",
                display_text="Visit",
                width="small"
            ),
            "title": st.column_config.TextColumn("Title", width="large"),
            "view_count": st.column_config.NumberColumn("Views", width="small"),
            "language": st.column_config.TextColumn("Language", width="small"),
            "days_old": st.column_config.NumberColumn("Days Old", width="small"),
            "channel_title": st.column_config.TextColumn("Channel", width="medium"),
            "channel_url": st.column_config.LinkColumn(
                "Channel URL",
                help="Click to visit the channel",
                display_text="Visit",
                width="small"
            ),
            "subscriber_count": st.column_config.NumberColumn("Subscribers", width="small"),
            "comment_count": st.column_config.NumberColumn("Comments", width="small"),
            "relevance_score": st.column_config.NumberColumn("Relevance", width="small")
        }

        # Afficher le DataFrame
        st.dataframe(
            combined_df,
            column_config=column_config,
            use_container_width=True,
            height=400
        )
