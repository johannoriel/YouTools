from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
import os
from datetime import datetime

translations["en"].update({
    "video_list_title": "Video List (video_list.csv -> filtered_video_list.csv)",
    "no_file_error": "No valid CSV file found. Please check the directory.",
    "language_filter": "Filter by Language",
    "days_old_filter": "Maximum Age (Days)",
    "subscribers_filter": "Minimum Subscribers",
    "export_button": "Export Filtered List",
    "export_success": "Filtered list exported to {filename}",
    "overwrite_checkbox": "Overwrite existing file",
})

translations["fr"].update({
    "video_list_title": "Liste des vidéos (video_list.csv -> filtered_video_list.csv)",
    "no_file_error": "Aucun fichier CSV valide trouvé. Vérifiez le répertoire.",
    "language_filter": "Filtrer par langue",
    "days_old_filter": "Âge maximum (jours)",
    "subscribers_filter": "Abonnés minimum",
    "export_button": "Exporter la liste filtrée",
    "export_success": "Liste filtrée exportée vers {filename}",
    "overwrite_checkbox": "Écraser le fichier existant",
})


class VideoListWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def select_video_list(self):
        work_directory = self.plugin_manager.config["common"]["work_directory"]

        # Recherche des fichiers CSV commençant par "video_list" dans le répertoire de travail
        csv_files = [f for f in os.listdir(work_directory) if f.startswith(
            'video_list') and f.endswith('.csv')]

        if not csv_files:
            st.error(t("no_file_error"))
            return None

        # Colonnes attendues pour un CSV valide
        required_columns = {'keyword', 'url', 'video_id', 'title', 'view_count', 'language',
                            'published_at', 'channel_id', 'channel_title',
                            'subscriber_count', 'comment_count', 'relevance_score'}

        # Charger et concaténer les fichiers CSV valides
        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(work_directory, csv_file)
            try:
                df = pd.read_csv(file_path)
                # Vérifier si le CSV contient les colonnes requises
                if required_columns.issubset(set(df.columns)):
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            st.error(t("no_file_error"))
            return None

        combined_df = pd.concat(dfs, ignore_index=True)

        # Gérer les doublons basés sur video_id, en gardant la ligne avec le plus de vues
        combined_df = combined_df.sort_values(
            by='view_count', ascending=False).drop_duplicates(subset='video_id', keep='first')

        # Calculer l'ancienneté en jours avec type nullable integer
        current_date = datetime.now()
        combined_df['days_old'] = combined_df['published_at'].apply(
            lambda x: (current_date - pd.to_datetime(x,
                       utc=True).tz_localize(None)).days if pd.notnull(x) else None
        ).astype('Int64')

        # Créer une URL pour la chaîne
        combined_df['channel_url'] = combined_df['channel_id'].apply(
            lambda x: f'https://www.youtube.com/channel/{x}' if pd.notnull(
                x) else ''
        )

        # Créer un DataFrame pour l'affichage avec toutes les colonnes nécessaires
        display_df = combined_df.copy()

        # Filtres
        st.subheader("Filtres")
        languages = display_df['language'].unique()
        selected_languages = st.multiselect(
            t("language_filter"), options=languages, default=languages, key=f"{self.prefix}_language_filter")
        max_days_old = st.number_input(
            t("days_old_filter"), min_value=0, value=30, step=1, key=f"{self.prefix}_days_old_filter")
        min_subscribers = st.number_input(
            t("subscribers_filter"), min_value=0, value=1000, step=100, key=f"{self.prefix}_subscribers_filter")

        # Appliquer les filtres
        filtered_df = display_df[
            (display_df['language'].isin(selected_languages)) &
            (display_df['days_old'].apply(lambda x: x <= max_days_old if pd.notnull(x) else True)) &
            (display_df['subscriber_count'].apply(
                lambda x: x >= min_subscribers if pd.notnull(x) else True))
        ]

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
            "view_count": st.column_config.NumberColumn("Views", width="small"),
            "language": st.column_config.TextColumn("Language", width="small"),
            "published_at": st.column_config.TextColumn("Published At", width="medium"),
            "channel_id": st.column_config.TextColumn("Channel ID", width="medium"),
            "channel_title": st.column_config.TextColumn("Channel", width="medium"),
            "channel_url": st.column_config.LinkColumn(
                "Channel URL",
                help="Click to visit the channel",
                display_text="Visit",
                width="small"
            ),
            "subscriber_count": st.column_config.NumberColumn("Subscribers", width="small"),
            "comment_count": st.column_config.NumberColumn("Comments", width="small"),
            "relevance_score": st.column_config.NumberColumn("Relevance", width="small"),
            "days_old": st.column_config.NumberColumn("Days Old", width="small")
        }

        # Afficher le DataFrame avec sélection multi-lignes
        selected_rows = st.dataframe(
            filtered_df,
            column_config=column_config,
            use_container_width=True,
            height=400,
            selection_mode="multi-row",
            on_select="rerun",
            key=f"{self.prefix}_video_dataframe"
        )

        return selected_rows.get('selection', {}).get('rows', []), filtered_df

    def display(self):
        st.title(t("video_list_title"))

        selected_rows, filtered_df = self.select_video_list()

        if selected_rows is None:
            return

        # Checkbox pour écraser ou renommer
        overwrite = st.checkbox(t("overwrite_checkbox"),
                                key=f"{self.prefix}_overwrite_checkbox")

        # Bouton pour exporter la liste filtrée
        if st.button(t("export_button"), key=f"{self.prefix}_export_button"):
            work_directory = self.plugin_manager.config["common"]["work_directory"]
            base_filename = "filtered_video_list.csv"
            export_path = os.path.join(work_directory, base_filename)

            if not overwrite and os.path.exists(export_path):
                # Trouver un nom de fichier unique en ajoutant _xxx
                i = 1
                while True:
                    new_filename = f"filtered_video_list_{i:03d}.csv"
                    new_export_path = os.path.join(
                        work_directory, new_filename)
                    if not os.path.exists(new_export_path):
                        export_path = new_export_path
                        break
                    i += 1

            filtered_df.iloc[selected_rows].to_csv(export_path, index=False)
            st.success(t("export_success").format(
                filename=os.path.basename(export_path)))
