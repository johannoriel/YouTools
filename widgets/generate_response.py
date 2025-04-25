from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from lib.youtube_db import cache_campaign_response
from plugins.automarket import AutomarketPlugin

translations["en"].update({
    "generate_response_title": "Generate Responses to Comments (comment_list.csv -> response_list.csv)",
    "no_file_error": "No valid comment_list CSV found. Please fetch comments first.",
    "generate_responses": "Generate Responses",
    "generating": "Generating responses...",
    "responses": "Suggested Responses",
    "post_responses": "Post Responses",
    "posting": "Posting responses...",
    "success": "Responses posted successfully!",
    "error": "Error posting responses: ",
    "char_limit_warning": "⚠️ This response exceeds 500 characters ({} characters). Please shorten it.",
    "export_responses": "Export Responses",
    "export_success": "Responses exported successfully to {filename}",
    "export_error": "Error during export: {error}",
    "response_to_comment": "Response to Comment {}",
    "edit_response": "Edit Response {}",
    "select_comment_file": "Select Comment List Files",
    "overwrite_responses_checkbox": "Overwrite existing response file",
})

translations["fr"].update({
    "generate_response_title": "Générer des réponses aux commentaires (comment_list.csv -> response_list.csv)",
    "no_file_error": "Aucun fichier comment_list CSV valide trouvé. Veuillez d'abord récupérer des commentaires.",
    "generate_responses": "Générer des réponses",
    "generating": "Génération des réponses...",
    "responses": "Réponses suggérées",
    "post_responses": "Poster les réponses",
    "posting": "Publication des réponses...",
    "success": "Réponses publiées avec succès !",
    "error": "Erreur lors de la publication : ",
    "char_limit_warning": "⚠️ Cette réponse dépasse 500 caractères ({} caractères). Veuillez la raccourcir.",
    "export_responses": "Exporter les réponses",
    "export_success": "Réponses exportées avec succès vers {filename}",
    "export_error": "Erreur lors de l'export : {error}",
    "response_to_comment": "Réponse au commentaire {}",
    "edit_response": "Modifier la réponse {}",
    "select_comment_file": "Sélectionner les fichiers de liste de commentaires",
    "overwrite_responses_checkbox": "Écraser le fichier de réponses existant",
})


def remove_quotes(text: str) -> str:
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text


class GenerateResponseWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def generate_responses(self, config, selected_comments, transcript, url, keywords):
        responses = []
        total_comments = len(selected_comments)
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for idx, comment in enumerate(selected_comments):
            progress = (idx + 1) / total_comments
            progress_bar.progress(progress)
            progress_text.text(
                f"Processing comment {idx + 1} of {total_comments}")

            comment_with_context = f"Comment by {comment['author']} on video {comment['video_title']} from channel {comment['channel_title']}:\n{comment['comment_text']}"
            prompt = config['promoteyoutube']['response_prompt'].format(
                url=url, transcript=transcript)
            try:
                llm_response = self.process_with_llm(
                    prompt,
                    config.get('llm', {}).get('llm_sys_prompt', ''),
                    comment_with_context
                )
                clean_response = remove_quotes(llm_response.strip())
            except Exception as e:
                clean_response = f"Error: {str(e)}"
            responses.append({
                'comment_id': comment['comment_id'],
                'response': clean_response,
                'target_video_id': comment['video_id'],
                'channel_id': comment['channel_id'],
                'keyword': keywords,
                'comment_text': comment['comment_text'],
                'author': comment['author'],
                'video_title': comment['video_title'],
                'channel_title': comment['channel_title']
            })

        progress_bar.empty()
        progress_text.empty()
        return responses

    def export_responses(self, responses, work_dir, overwrite):
        try:
            base_filename = "response_list.csv"
            output_path = os.path.join(work_dir, base_filename)

            if not overwrite and os.path.exists(output_path):
                i = 1
                while True:
                    new_filename = f"response_list_{i:03d}.csv"
                    new_output_path = os.path.join(work_dir, new_filename)
                    if not os.path.exists(new_output_path):
                        output_path = new_output_path
                        break
                    i += 1

            responses_data = [
                {
                    'comment_id': resp['comment_id'],
                    'response_text': resp['response'],
                    'comment_text': resp['comment_text'],
                    'video_id': resp['target_video_id'],
                    'channel_id': resp['channel_id'],
                    'author': resp['author'],
                    'video_title': resp['video_title'],
                    'channel_title': resp['channel_title'],
                    'keywords': resp['keyword']
                }
                for resp in responses
            ]
            df = pd.DataFrame(responses_data)
            df.to_csv(output_path, index=False)
            st.success(t("export_success").format(filename=output_path))
        except Exception as e:
            st.error(t("export_error").format(str(e)))

    def display(self):
        st.title(t("generate_response_title"))
        work_dir = self.plugin_manager.config["common"]["work_directory"]

        # Recherche des fichiers CSV commençant par "comment_list"
        comment_files = [f for f in os.listdir(work_dir) if f.startswith(
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
            file_path = os.path.join(work_dir, comment_file)
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
        transcript_path = os.path.join(work_dir, "transcript.txt")
        transcript = open(transcript_path, 'r').read(
        ) if os.path.exists(transcript_path) else ""
        url_path = os.path.join(work_dir, "url.txt")
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
                selected_comments = [combined_df.iloc[i].to_dict(
                ) for i in selected_comment_rows['selection']['rows']]
                keywords = selected_comments[0].get('keywords', '')
                responses = self.generate_responses(
                    self.plugin_manager.config, selected_comments, transcript, url, keywords)

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

                # Stocker les réponses dans st.session_state
                st.session_state['generated_responses'] = responses

        # Affichage des réponses stockées dans st.session_state
        if 'generated_responses' in st.session_state and st.session_state['generated_responses']:
            st.subheader(t("responses"))
            responses_df = pd.DataFrame([
                {
                    'comment_text': resp['comment_text'],
                    'response_text': resp['response'],
                    'author': resp['author'],
                    'video_title': resp['video_title'],
                    'channel_title': resp['channel_title'],
                    'comment_id': resp['comment_id'],
                    'video_id': resp['target_video_id'],
                    'channel_id': resp['channel_id'],
                    'keywords': resp['keyword']
                }
                for resp in st.session_state['generated_responses']
            ])

            response_column_config = {
                "comment_text": st.column_config.TextColumn("Comment", width="large"),
                "response_text": st.column_config.TextColumn("Response", width="large"),
                "author": st.column_config.TextColumn("Author", width="medium"),
                "video_title": st.column_config.TextColumn("Video Title", width="large"),
                "channel_title": st.column_config.TextColumn("Channel", width="medium"),
            }

            # Afficher le DataFrame avec sélection multi-lignes pour les réponses
            selected_response_rows = st.dataframe(
                responses_df,
                column_config=response_column_config,
                use_container_width=True,
                height=400,
                selection_mode="multi-row",
                on_select="rerun",
                key=f"{self.prefix}_response_dataframe"
            )

            # Permettre l'édition des réponses
            for i, row in responses_df.iterrows():
                st.write(t("response_to_comment").format(i + 1))
                edited_response = st.text_area(
                    t("edit_response").format(i + 1),
                    value=row['response_text'],
                    key=f"{self.prefix}_response_{i}",
                    height=100
                )
                responses_df.at[i, 'response_text'] = edited_response

                if len(edited_response) > 500:
                    st.warning(t("char_limit_warning").format(
                        len(edited_response)))

            # Mettre à jour st.session_state avec les réponses éditées
            st.session_state['generated_responses'] = [
                {
                    'comment_id': row['comment_id'],
                    'response': row['response_text'],
                    'target_video_id': row['video_id'],
                    'channel_id': row['channel_id'],
                    'keyword': row['keywords'],
                    'comment_text': row['comment_text'],
                    'author': row['author'],
                    'video_title': row['video_title'],
                    'channel_title': row['channel_title']
                }
                for _, row in responses_df.iterrows()
            ]

            # Case à cocher pour écraser le fichier
            overwrite_responses = st.checkbox(
                t("overwrite_responses_checkbox"), key=f"{self.prefix}_overwrite_responses")

            # Bouton pour exporter les réponses sélectionnées
            if st.button(t("export_responses"), key=f"{self.prefix}_export_responses") and selected_response_rows['selection']['rows']:
                selected_indices = selected_response_rows['selection']['rows']
                selected_responses = [
                    st.session_state['generated_responses'][i] for i in selected_indices]
                self.export_responses(selected_responses,
                                      work_dir, overwrite_responses)

            # Bouton pour publier les réponses sélectionnées
            if st.button(t("post_responses"), key=f"{self.prefix}_post_responses") and selected_response_rows['selection']['rows']:
                with st.spinner(t("posting")):
                    automarket = self.plugin_manager.get_plugin('automarket')
                    campaign_id = datetime.now().isoformat()
                    selected_indices = selected_response_rows['selection']['rows']
                    selected_responses = [
                        st.session_state['generated_responses'][i] for i in selected_indices]
                    automarket.post_responses(
                        self.plugin_manager.config, selected_responses, campaign_id)
                    st.success(t("success"))
