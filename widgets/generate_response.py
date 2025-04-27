from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from lib.youtube_db import cache_campaign_response
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from widgets.utils import generate_responses_for_list, export_responses, remove_quotes

translations["en"].update({
    "generate_response_title": "Generate Responses to Comments (comment_list.csv -> response_list.csv)",
    "no_file_error": "No valid comment_list CSV found. Please fetch comments first.",
    "generate_responses": "Generate Responses",
    "generating": "Generating responses...",
    "responses": "Suggested Responses",
    "char_limit_warning": "⚠️ This response exceeds 500 characters ({} characters). Please shorten it.",
    "export_responses": "Export Responses",
    "export_success": "Responses exported successfully to {filename}",
    "export_error": "Error during export: {error}",
    "select_comment_file": "Select Comment List Files",
    "overwrite_responses_checkbox": "Overwrite existing response file",
})

translations["fr"].update({
    "generate_response_title": "Générer des réponses aux commentaires (comment_list.csv -> response_list.csv)",
    "no_file_error": "Aucun fichier comment_list CSV valide trouvé. Veuillez d'abord récupérer des commentaires.",
    "generate_responses": "Générer des réponses",
    "generating": "Génération des réponses...",
    "responses": "Réponses suggérées",
    "char_limit_warning": "⚠️ Cette réponse dépasse 500 caractères ({} caractères). Veuillez la raccourcir.",
    "export_responses": "Exporter les réponses",
    "export_success": "Réponses exportées avec succès vers {filename}",
    "export_error": "Erreur lors de l'export : {error}",
    "select_comment_file": "Sélectionner les fichiers de liste de commentaires",
    "overwrite_responses_checkbox": "Écraser le fichier de réponses existant",
})

class GenerateResponseWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.work_dir = self.plugin_manager.config["common"]["work_directory"]

    def display(self):
        st.title(t("generate_response_title"))

        # Recherche des fichiers CSV commençant par "comment_list"
        comment_files = [f for f in os.listdir(self.work_dir) if f.startswith(
            'comment_list') and f.endswith('.csv')]
        if not comment_files:
            st.error(t("no_file_error"))
            return

        # Sélection multiple des fichiers de commentaires
        selected_comment_files = st.multiselect(
            t("select_comment_file"),
            options=comment_files,
            default=[comment_files[0]] if comment_files else [],
            key=f"{self.prefix}_select_comment_file"
        )

        if not selected_comment_files:
            st.warning("Please select at least one comment list file.")
            return

        # Charger et concaténer les fichiers sélectionnés
        dfs = []
        required_columns = {'comment_id', 'comment_text',
                           'video_id', 'video_title', 'channel_title', 'author'}
        for comment_file in selected_comment_files:
            file_path = os.path.join(self.work_dir, comment_file)
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
            subset='comment_id', keep='first')

        # Charger la transcription et l'URL
        transcript_path = os.path.join(self.work_dir, "transcript.txt")
        transcript = open(transcript_path, 'r').read(
        ) if os.path.exists(transcript_path) else ""
        url_path = os.path.join(self.work_dir, "url.txt")
        url = open(url_path, 'r').read().strip(
        ) if os.path.exists(url_path) else ""

        # Configurer les colonnes pour l'affichage des commentaires
        comment_column_config = {
            "comment_text": st.column_config.TextColumn("Comment", width="large"),
            "author": st.column_config.TextColumn("Author", width="medium"),
            "video_title": st.column_config.TextColumn("Video Title", width="large"),
            "channel_title": st.column_config.TextColumn("Channel", width="medium"),
            "comment_url": st.column_config.LinkColumn(
                "Comment URL",
                help="Click to view comment",
                display_text="View",
                width="small"
            ),
        }

        # Afficher le DataFrame avec sélection multi-lignes pour les commentaires
        selected_comment_rows = st.dataframe(
            combined_df,
            column_config=comment_column_config,
            use_container_width=True,
            height=400,
            selection_mode="multi-row",
            on_select="rerun",
            key=f"{self.prefix}_comment_dataframe"
        )

        # Bouton pour générer les réponses
        if st.button(t("generate_responses"), key=f"{self.prefix}_generate_responses") and selected_comment_rows['selection']['rows']:
            with st.spinner(t("generating")):
                selected_comments = [
                    {
                        'comment_id': row['comment_id'],
                        'comment_text': row['comment_text'],
                        'video_id': row['video_id'],
                        'channel_id': row['channel_id'],
                        'video_title': row['video_title'],
                        'channel_title': row['channel_title'],
                        'author': row['author'],
                        'keyword': row.get('keyword', '')
                    }
                    for i, row in combined_df.iloc[selected_comment_rows['selection']['rows']].iterrows()
                ]
                responses = generate_responses_for_list(
                    widget=self,
                    config=self.plugin_manager.config,
                    content_list=selected_comments,
                    prompt_template=self.plugin_manager.config['promoteyoutube']['response_prompt'],
                    context=transcript,
                    universitairesurl=url,
                    keyword=selected_comments[0]['keyword']
                )

                # Sauvegarde dans la base de données
                campaign_id = datetime.now().isoformat()
                for resp in responses:
                    cache_campaign_response(
                        campaign_id=campaign_id,
                        comment_id=resp['comment_id'],
                        comment_text=resp['comment_text'],
                        response_text=resp['response'],
                        video_id=resp['target_video_id'],
                        channel_id=resp['channel_id'],
                        author=resp['author'],
                        status="pending"
                    )

                st.session_state['generated_responses'] = responses

        # Affichage des réponses avec AgGrid
        if 'generated_responses' in st.sessionโปรgramme_state and st.session_state['generated_responses']:
            st.subheader(t("responses"))
            responses_df = pd.DataFrame([
                {
                    'content_text': resp['content_text'],
                    'response_text': resp['response'],
                    'author': resp['author'],
                    'video_title': resp['video_title'],
                    'channel_title': resp['channel_title'],
                    'comment_id': resp['comment_id'],
                    'video_id': resp['target_video_id'],
                    'channel_id': resp['channel_id'],
                    'keyword': resp['keyword']
                }
                for resp in st.session_state['generated_responses']
            ])

            # Réorganiser les colonnes
            column_order = ['comment_text', 'response_text', 'author', 'video_title',
                            'channel_title', 'comment_id', 'video_id', 'channel_id', 'keyword']
            responses_df = responses_df[column_order].reset_index(drop=True)

            # Configuration de la grille AgGrid
            gb = GridOptionsBuilder.from_dataframe(responses_df)
            gb.configure_column("content_text", headerName="Comment", width=300, editable=False)
            gb.configure_column("response_text", headerName="Response", width=300, editable=True,
                               cellEditor='agLargeTextCellEditor', cellEditorPopup=True, cellEditorParams={'maxLength': '500'})
            gb.configure_column("author", headerName="Author", width=150, editable=False)
            gb.configure_column("video_title", headerName="Video Title", width=200, editable=False)
            gb.configure_column("channel_title", headerName="Channel", width=150, editable=False)
            gb.configure_column("comment_id", headerName="Comment ID", hide=True)
            gb.configure_column("video_id", headerName="Video ID", hide=True)
            gb.configure_column("channel_id", headerName="Channel ID", hide=True)
            gb.configure_column("keyword", headerName="keyword", hide=True)
            gb.configure_selection(selection_mode="multiple", use_checkbox=True, header_checkbox=True)
            gb.configure_default_column(editable=False, resizable=True)
            grid_options = gb.build()

            # Afficher la grille
            grid_response = AgGrid(
                responses_df,
                gridOptions=grid_options,
                height=400,
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True,
                update_mode=GridUpdateMode.VALUE_CHANGED | GridUpdateMode.SELECTION_CHANGED,
                key=f"{self.prefix}_response_grid"
            )

            # Mettre à jour st.session_state avec les réponses éditées
            updated_df = grid_response['data']
            updated_responses = st.session_state['generated_responses'].copy()
            for idx, row in updated_df.iterrows():
                updated_responses[int(idx)] = {
                    'comment_id': row['comment_id'],
                    'response': row['response_text'],
                    'target_video_id': row['video_id'],
                    'channel_id': row['channel_id'],
                    'keyword': row['keyword'],
                    'comment_text': row['comment_text'],
                    'author': row['author'],
                    'video_title': row['video_title'],
                    'channel_title': row['channel_title']
                }
            st.session_state['generated_responses'] = updated_responses

            # Vérification de la limite de caractères
            for i, row in updated_df.iterrows():
                if len(row['response_text']) > 500:
                    st.warning(t("char_limit_warning").format(len(row['response_text'])))

            # Récupérer les lignes sélectionnées
            selected_rows = grid_response['selected_rows']
            selected_responses = []
            if selected_rows is not None and not selected_rows.empty:
                selected_indices = selected_rows.index.tolist()
                selected_responses = [
                    st.session_state['generated_responses'][int(i)] for i in selected_indices]

            # Case à cocher pour écraser le fichier
            overwrite_responses = st.checkbox(
                t("overwrite_responses_checkbox"), key=f"{self.prefix}_overwrite_responses")

            # Bouton pour exporter les réponses sélectionnées
            if st.button(t("export_responses"), key=f"{self.prefix}_export_responses") and selected_rows is not None and not selected_rows.empty:
                export_responses(selected_responses, self.work_dir, "response_list.csv", overwrite_responses)
