from global_vars import translations, t
from app import Plugin
import streamlit as st
import os
from plugins.ragllm import RagllmPlugin
from typing import List, Dict, Any, Optional
from social_api import YoutubeAPI  # Utilisation de l'API YouTube depuis social_api.py

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "promoteyoutube_tab": "Promote YouTube",
    "promoteyoutube_header": "Promote Content on YouTube",
    "promoteyoutube_transcript": "Transcript",
    "promoteyoutube_url": "Video URL",
    "promoteyoutube_keywords": "Keywords to Search",
    "promoteyoutube_search": "Search Videos and Comments",
    "promoteyoutube_searching": "Searching videos and comments...",
    "promoteyoutube_comments": "Recent Comments",
    "promoteyoutube_select_comments": "Select Comments to Respond",
    "promoteyoutube_generate_responses": "Generate Responses",
    "promoteyoutube_generating": "Generating responses...",
    "promoteyoutube_responses": "Suggested Responses",
    "promoteyoutube_post_responses": "Post Responses",
    "promoteyoutube_posting": "Posting responses...",
    "promoteyoutube_success": "Responses posted successfully!",
    "promoteyoutube_error": "Error posting responses: ",
})

translations["fr"].update({
    "promoteyoutube_tab": "Promotion YouTube",
    "promoteyoutube_header": "Promouvoir le Contenu sur YouTube",
    "promoteyoutube_transcript": "Transcription",
    "promoteyoutube_url": "URL de la vidéo",
    "promoteyoutube_keywords": "Mots-clés à rechercher",
    "promoteyoutube_search": "Rechercher des vidéos et commentaires",
    "promoteyoutube_searching": "Recherche des vidéos et commentaires...",
    "promoteyoutube_comments": "Commentaires récents",
    "promoteyoutube_select_comments": "Sélectionner des commentaires pour répondre",
    "promoteyoutube_generate_responses": "Générer des réponses",
    "promoteyoutube_generating": "Génération des réponses...",
    "promoteyoutube_responses": "Réponses suggérées",
    "promoteyoutube_post_responses": "Poster les réponses",
    "promoteyoutube_posting": "Publication des réponses...",
    "promoteyoutube_success": "Réponses publiées avec succès !",
    "promoteyoutube_error": "Erreur lors de la publication : ",
})

def remove_quotes(text: str) -> str:
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text

class PromoteyoutubePlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'comments' not in st.session_state:
            st.session_state.comments = []
        if 'selected_comments' not in st.session_state:
            st.session_state.selected_comments = {}
        if 'generated_responses' not in st.session_state:
            st.session_state.generated_responses = []
        if 'selected_responses' not in st.session_state:
            st.session_state.selected_responses = {}

    def get_config_fields(self):
        return {
            "max_videos": {
                "type": "number",
                "label": "Maximum Number of Videos to Fetch",
                "default": 5
            },
            "max_comments_per_video": {
                "type": "number",
                "label": "Maximum Comments per Video",
                "default": 2
            },
            "response_prompt": {
                "type": "text",
                "label": "LLM Prompt for Responses",
                "default": """Suggère une réponse à ce commentaire de moins de 500 caractères, en lien avec la vidéo dans l'URL {url} (doit être mentionnée). Le ton est direct, réponds comme si tu étais l'utilisateur, et en t'inspirant du transcript suivant : {transcript}"""
            }
        }

    def get_tabs(self):
        return [{"name": t("promoteyoutube_tab"), "plugin": "promoteyoutube"}]

    def search_videos_and_comments(self, keywords, max_videos, max_comments_per_video):
        youtube_api = YoutubeAPI(self.plugin_manager.config)
        videos = youtube_api.search_videos(keywords, max_videos)
        comments = []

        for video in videos:
            video_comments = youtube_api.get_comments(video['id'], max_comments_per_video)
            comments.extend(video_comments)

        return comments

    def generate_responses(self, config, selected_comments, transcript, url):
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        responses = []

        for comment_index in selected_comments:
            comment = st.session_state.comments[comment_index]  # Récupérer le commentaire complet
            comment_text = comment['text']
            prompt = config['promoteyoutube']['response_prompt'].format(
                url=url,
                transcript=transcript
            )
            llm_response = ragllm_plugin.process_with_llm(
                prompt,
                config.get('llm', {}).get('llm_sys_prompt', ''),
                comment_text
            )
            clean_response = remove_quotes(llm_response.strip())
            responses.append({
                'comment_id': comment['id'],  # Utiliser l'ID du commentaire
                'response': clean_response,
                'comment_index': comment_index  # Ajouter l'index pour référence
            })

        return responses

    def post_responses(self, config, selected_responses):
        youtube_api = YoutubeAPI(self.plugin_manager.config)

        for response in selected_responses:
            comment_id = response['comment_id']
            response_text = response['response']
            youtube_api.post_comment_reply(comment_id, response_text)

    def run(self, config):
        st.header(t("promoteyoutube_header"))

        # Initialiser le dictionnaire des sélections de commentaires
        if 'selected_comments' not in st.session_state:
            st.session_state.selected_comments = {}

        # Initialiser le dictionnaire des sélections de réponses
        if 'selected_responses' not in st.session_state:
            st.session_state.selected_responses = {}

        # Load or input transcript
        work_dir = config['common']['work_directory']
        transcript_path = os.path.join(work_dir, "transcript.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                transcript = f.read()
            st.text_area(t("promoteyoutube_transcript"), transcript, height=100, disabled=True)
        else:
            transcript = st.text_area(t("promoteyoutube_transcript"), height=200)

        # Load or input URL
        url_path = os.path.join(work_dir, "url.txt")
        if os.path.exists(url_path):
            with open(url_path, 'r') as f:
                url = f.read().strip()
            st.text_input(t("promoteyoutube_url"), url, disabled=True)
        else:
            url = st.text_input(t("promoteyoutube_url"))

        # Input keywords to search
        if 'keywords' not in st.session_state:
            st.session_state.keywords = ""

        keywords = st.text_input(
            t("promoteyoutube_keywords"),
            value=st.session_state.keywords,
            key="promoteyoutube_keywords"
        )
        st.session_state.keywords = keywords

        if not keywords:
            st.warning("Please enter keywords to search for videos.")
            return

        # Search videos and comments button
        if st.button(t("promoteyoutube_search")):
            with st.spinner(t("promoteyoutube_searching")):
                max_videos = config['promoteyoutube']['max_videos']
                max_comments_per_video = config['promoteyoutube']['max_comments_per_video']
                youtube_api = YoutubeAPI(self.plugin_manager.config)

                # Rechercher les vidéos
                videos = youtube_api.search_videos(keywords, max_videos)
                st.session_state.videos = videos  # Stocker les vidéos dans session_state

                # Récupérer les commentaires pour chaque vidéo
                comments = []
                for video in videos:
                    video_comments = youtube_api.get_comments(video['id'], max_comments_per_video)
                    for comment in video_comments:
                        comment['video_title'] = video['title']
                        comment['channel_title'] = video['channel_title']  # Ajouter le nom de la chaîne
                    comments.extend(video_comments)

                st.session_state.comments = comments  # Stocker les commentaires dans session_state

        # Afficher les vidéos trouvées
        if 'videos' in st.session_state and st.session_state.videos:
            st.subheader("Vidéos trouvées")
            for video in st.session_state.videos:
                st.write(f"**{video['title']}**")
                st.markdown(f"Chaîne : **[{video['channel_title']}](https://www.youtube.com/channel/{video['channel_id']})**")  # Lien vers la chaîne
                st.markdown(f"[Voir la vidéo]({video['url']})")

        # Afficher les commentaires
        if 'comments' in st.session_state and st.session_state.comments:
            st.subheader(t("promoteyoutube_comments"))
            for i, comment in enumerate(st.session_state.comments):
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <p><strong>{comment['author']}</strong> (sur la vidéo : <em>{comment['video_title']}</em>) :</p>
                        <p>{comment['text']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Mettre à jour la sélection dans st.session_state.selected_comments
                if i not in st.session_state.selected_comments:
                    st.session_state.selected_comments[i] = False

                # Créer la case à cocher
                selected = st.checkbox(
                    f"Select Comment {i+1}",
                    value=st.session_state.selected_comments[i],  # Utiliser la valeur stockée
                    key=f"select_comment_{i}"
                )

                # Mettre à jour la sélection dans st.session_state.selected_comments
                st.session_state.selected_comments[i] = selected

        # Bouton "Select All" pour les commentaires
        if 'comments' in st.session_state and st.session_state.comments:
            if st.button("Select All Comments"):
                for i in range(len(st.session_state.comments)):
                    st.session_state.selected_comments[i] = True
            if st.button("Deselect All Comments"):
                for i in range(len(st.session_state.comments)):
                    st.session_state.selected_comments[i] = False

        # Generate responses button
        if st.button(t("promoteyoutube_generate_responses")) and st.session_state.selected_comments:
            with st.spinner(t("promoteyoutube_generating")):
                selected_comments = [i for i, selected in st.session_state.selected_comments.items() if selected]
                st.session_state.generated_responses = self.generate_responses(
                    config, selected_comments, transcript, url
                )

        # Display generated responses
        if 'generated_responses' in st.session_state and st.session_state.generated_responses:
            st.subheader(t("promoteyoutube_responses"))
            for i, response in enumerate(st.session_state.generated_responses):
                comment_index = response['comment_index']  # Récupérer l'index du commentaire associé
                comment = st.session_state.comments[comment_index]

                st.markdown(
                    f"""
                    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <p><strong>{comment['author']}</strong> (sur la vidéo : <em>{comment['video_title']}</em>) :</p>
                        <p>{comment['text']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(f"**Response to Comment {comment_index+1}**:")
                edited_response = st.text_area(
                    f"Edit Response {i+1}",
                    value=response['response'],  # Utiliser la valeur stockée
                    key=f"response_{i}",
                    height=100
                )
                st.session_state.generated_responses[i]['response'] = edited_response

                # Vérification de la longueur de la réponse
                if len(edited_response) > 500:
                    st.warning(f"⚠️ Cette réponse dépasse 500 caractères ({len(edited_response)} caractères). Veuillez la raccourcir.")

                # Mettre à jour la sélection dans st.session_state.selected_responses
                if i not in st.session_state.selected_responses:
                    st.session_state.selected_responses[i] = False

                selected = st.checkbox(
                    f"Select Response {i+1}",
                    value=st.session_state.selected_responses[i],  # Utiliser la valeur stockée
                    key=f"select_response_{i}"
                )

                # Mettre à jour la sélection dans st.session_state.selected_responses
                st.session_state.selected_responses[i] = selected

        # Bouton "Select All" pour les réponses
        if 'generated_responses' in st.session_state and st.session_state.generated_responses:
            if st.button("Select All Responses"):
                for i in range(len(st.session_state.generated_responses)):
                    st.session_state.selected_responses[i] = True
            if st.button("Deselect All Responses"):
                for i in range(len(st.session_state.generated_responses)):
                    st.session_state.selected_responses[i] = False

        # Post responses button
        if st.button(t("promoteyoutube_post_responses")) and st.session_state.selected_responses:
            with st.spinner(t("promoteyoutube_posting")):
                selected_responses = [
                    response for i, response in enumerate(st.session_state.generated_responses)
                    if st.session_state.selected_responses[i]
                ]
                self.post_responses(config, selected_responses)
                st.success(t("promoteyoutube_success"))
