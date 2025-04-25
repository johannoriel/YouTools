from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from plugins.automarket import AutomarketPlugin
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

translations["en"].update({
    "post_response_title": "Post Responses to Comments (response_list.csv)",
    "no_file_error": "No valid response_list CSV found. Please generate responses first.",
    "post_responses": "Post Responses",
    "posting": "Posting responses...",
    "success": "Responses posted successfully!",
    "error": "Error posting responses: ",
    "char_limit_warning": "⚠️ This response exceeds 500 characters ({} characters). Please shorten it.",
    "select_response_file": "Select Response List Files",
})

translations["fr"].update({
    "post_response_title": "Publier des réponses aux commentaires (response_list.csv)",
    "no_file_error": "Aucun fichier response_list CSV valide trouvé. Veuillez d'abord générer des réponses.",
    "post_responses": "Publier les réponses",
    "posting": "Publication des réponses...",
    "success": "Réponses publiées avec succès !",
    "error": "Erreur lors de la publication : ",
    "char_limit_warning": "⚠️ Cette réponse dépasse 500 caractères ({} caractères). Veuillez la raccourcir.",
    "select_response_file": "Sélectionner les fichiers de liste de réponses",
})


class PostResponseWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def display(self):
        st.title(t("post_response_title"))
        work_dir = self.work_dir()

        # Recherche des fichiers CSV commençant par "response_list"
        response_files = [f for f in os.listdir(work_dir) if f.startswith(
            'response_list') and f.endswith('.csv')]
        if not response_files:
            st.error(t("no_file_error"))
            return

        # Sélection multiple des fichiers de réponses
        selected_response_files = st.multiselect(
            t("select_response_file"),
            options=response_files,
            default=[response_files[0]] if response_files else [],
            key=f"{self.prefix}_select_response_file"
        )

        if not selected_response_files:
            st.warning("Please select at least one response list file.")
            return

        # Charger et concaténer les fichiers sélectionnés
        dfs = []
        required_columns = {'comment_id', 'response_text', 'comment_text',
                            'video_id', 'channel_id', 'author', 'video_title', 'channel_title'}
        for response_file in selected_response_files:
            file_path = os.path.join(work_dir, response_file)
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

        # Réorganiser les colonnes
        column_order = ['comment_text', 'response_text', 'author', 'video_title',
                        'channel_title', 'comment_id', 'video_id', 'channel_id', 'keywords']
        available_columns = [
            col for col in column_order if col in combined_df.columns]
        combined_df = combined_df[available_columns].reset_index(drop=True)

        # Configuration de la grille AgGrid
        gb = GridOptionsBuilder.from_dataframe(combined_df)
        gb.configure_column(
            "comment_text", headerName="Comment", width=300, editable=False)
        gb.configure_column(
            "response_text", headerName="Response", width=300, editable=True,
            cellEditor='agLargeTextCellEditor', cellEditorPopup=True, cellEditorParams={'maxLength': '500'}
        )
        gb.configure_column("author", headerName="Author",
                            width=150, editable=False)
        gb.configure_column(
            "video_title", headerName="Video Title", width=200, editable=False)
        gb.configure_column(
            "channel_title", headerName="Channel", width=150, editable=False)
        gb.configure_column("comment_id", headerName="Comment ID", hide=True)
        gb.configure_column("video_id", headerName="Video ID", hide=True)
        gb.configure_column("channel_id", headerName="Channel ID", hide=True)
        gb.configure_column("keywords", headerName="Keywords", hide=True)
        gb.configure_selection(selection_mode="multiple",
                               use_checkbox=True, header_checkbox=True)
        gb.configure_default_column(editable=False, resizable=True)
        grid_options = gb.build()
        grid_options['rowSelection'] = 'multiple'
        grid_options['suppressRowClickSelection'] = True

        # Afficher la grille
        grid_response = AgGrid(
            combined_df,
            gridOptions=grid_options,
            height=400,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            update_mode=GridUpdateMode.VALUE_CHANGED | GridUpdateMode.SELECTION_CHANGED,
            enable_enterprise_modules=True,
            key=f"{self.prefix}_response_grid"
        )

        # Mettre à jour les données avec les réponses éditées
        updated_df = grid_response['data']
        responses = [
            {
                'comment_id': row['comment_id'],
                'response': row['response_text'],
                'target_video_id': row['video_id'],
                'channel_id': row['channel_id'],
                'keyword': row.get('keywords', ''),
                'comment_text': row['comment_text'],
                'author': row['author'],
                'video_title': row['video_title'],
                'channel_title': row['channel_title']
            }
            for _, row in combined_df.iterrows()
        ]
        for idx, row in updated_df.iterrows():
            responses[int(idx)] = {
                'comment_id': row['comment_id'],
                'response': row['response_text'],
                'target_video_id': row['video_id'],
                'channel_id': row['channel_id'],
                'keyword': row.get('keywords', ''),
                'comment_text': row['comment_text'],
                'author': row['author'],
                'video_title': row['video_title'],
                'channel_title': row['channel_title']
            }

        # Vérification de la limite de caractères
        for i, row in updated_df.iterrows():
            if len(row['response_text']) > 500:
                st.warning(t("char_limit_warning").format(
                    len(row['response_text'])))

        # Récupérer les lignes sélectionnées
        selected_rows = grid_response['selected_rows']
        selected_responses = []
        if selected_rows is not None and not selected_rows.empty:
            selected_indices = selected_rows.index.tolist()
            selected_responses = [responses[int(i)] for i in selected_indices]

        # Bouton pour publier les réponses sélectionnées
        if st.button(t("post_responses"), key=f"{self.prefix}_post_responses") and selected_rows is not None and not selected_rows.empty:
            with st.spinner(t("posting")):
                automarket = self.plugin_manager.get_plugin('automarket')
                campaign_id = datetime.now().isoformat()
                automarket.post_responses(
                    self.plugin_manager.config, selected_responses, campaign_id)
                st.success(t("success"))
