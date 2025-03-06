from global_vars import translations, t
from app import Plugin
import streamlit as st
import os
from plugins.ragllm import RagllmPlugin
from typing import List, Dict, Any, Optional
from social_api import BlueskyAPI  # Utilisation de l'API Bluesky depuis social_api.py
import pyperclip  # Pour copier le texte en un clic

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "promotebluesky_tab": "Promote Bluesky",
    "promotebluesky_header": "Promote Content",
    "promotebluesky_transcript": "Transcript",
    "promotebluesky_url": "Video URL",
    "promotebluesky_keywords": "Keywords to Search",
    "promotebluesky_search": "Search Posts",
    "promotebluesky_searching": "Searching posts...",
    "promotebluesky_posts": "Recent Posts",
    "promotebluesky_select_posts": "Select Posts to Respond",
    "promotebluesky_generate_responses": "Generate Responses",
    "promotebluesky_generating": "Generating responses...",
    "promotebluesky_responses": "Suggested Responses",
    "promotebluesky_post_responses": "Post Responses",
    "promotebluesky_posting": "Posting responses...",
    "promotebluesky_success": "Responses posted successfully!",
    "promotebluesky_error": "Error posting responses: ",
})

translations["fr"].update({
    "promotebluesky_tab": "Promotion Bluesky",
    "promotebluesky_header": "Promouvoir le Contenu",
    "promotebluesky_transcript": "Transcription",
    "promotebluesky_url": "URL de la vidéo",
    "promotebluesky_keywords": "Mots-clés à rechercher",
    "promotebluesky_search": "Rechercher des posts",
    "promotebluesky_searching": "Recherche des posts...",
    "promotebluesky_posts": "Posts récents",
    "promotebluesky_select_posts": "Sélectionner des posts pour répondre",
    "promotebluesky_generate_responses": "Générer des réponses",
    "promotebluesky_generating": "Génération des réponses...",
    "promotebluesky_responses": "Réponses suggérées",
    "promotebluesky_post_responses": "Poster les réponses",
    "promotebluesky_posting": "Publication des réponses...",
    "promotebluesky_success": "Réponses publiées avec succès !",
    "promotebluesky_error": "Erreur lors de la publication : ",
})

def remove_quotes(text: str) -> str:
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text

def get_post_url(handle: str, post_id: str) -> str:
    return f"https://bsky.app/profile/{handle}/post/{post_id}"

class PromoteblueskyPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'posts' not in st.session_state:
            st.session_state.posts = []
        if 'selected_posts' not in st.session_state:
            st.session_state.selected_posts = {}
        if 'generated_responses' not in st.session_state:
            st.session_state.generated_responses = []
        if 'selected_responses' not in st.session_state:
            st.session_state.selected_responses = {}

    def get_config_fields(self):
        return {
            "max_posts": {
                "type": "number",
                "label": "Maximum Number of Posts to Fetch",
                "default": 10
            },
            "response_prompt": {
                "type": "text",
                "label": "LLM Prompt for Responses",
                "default": """Suggère une réponse à ce post de moins de 300 caractères, en lien avec la vidéo dans l'URL {url} (doit être mentionnée). Le ton est direct, réponds comme si tu étais l'utilisateur, et en t'inspirant du transcript suivant : {transcript}"""
            }
        }

    def get_tabs(self):
        return [{"name": t("promotebluesky_tab"), "plugin": "promotebluesky"}]

    def search_posts(self, keywords, max_posts):
        bluesky_api = BlueskyAPI(self.plugin_manager.config)
        query = " OR ".join(keywords)
        posts = bluesky_api.search_posts(query, max_posts)
        return posts

    def generate_responses(self, config, selected_posts, transcript, url):
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        responses = []

        for post_id in selected_posts:
            post_text = st.session_state.posts[post_id]['text']
            prompt = config['promotebluesky']['response_prompt'].format(
                url=url,
                transcript=transcript
            )
            llm_response = ragllm_plugin.process_with_llm(
                prompt,
                config.get('llm', {}).get('llm_sys_prompt', ''),
                post_text
            )
            clean_response = remove_quotes(llm_response.strip())
            responses.append({
                'post_id': post_id,
                'response': clean_response
            })

        return responses

    def post_responses(self, config, selected_responses):
        bluesky_api = BlueskyAPI(self.plugin_manager.config)

        for response in selected_responses:
            post_id = response['post_id']
            response_text = response['response']
            bluesky_api.create_post(response_text, in_reply_to_post_id=post_id)

    def run(self, config):
        st.header(t("promotebluesky_header"))

        # Load or input transcript
        work_dir = config['common']['work_directory']
        transcript_path = os.path.join(work_dir, "transcript.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                transcript = f.read()
            st.text_area(t("promotebluesky_transcript"), transcript, height=100, disabled=True)
        else:
            transcript = st.text_area(t("promotebluesky_transcript"), height=200)

        # Load or input URL
        url_path = os.path.join(work_dir, "url.txt")
        if os.path.exists(url_path):
            with open(url_path, 'r') as f:
                url = f.read().strip()
            st.text_input(t("promotebluesky_url"), url, disabled=True)
        else:
            url = st.text_input(t("promotebluesky_url"))

        # Input keywords to search
        keywords = st.text_input(t("promotebluesky_keywords"), key="promotebluesky_keywords")
        if not keywords:
            st.warning("Please enter keywords to search for posts.")
            return

        # Search posts button
        if st.button(t("promotebluesky_search")):
            with st.spinner(t("promotebluesky_searching")):
                max_posts = config['promotebluesky']['max_posts']
                st.session_state.posts = self.search_posts(keywords, max_posts)

                # Vérification des mots-clés dans les résultats
                if st.session_state.posts:
                    keyword_list = [kw.strip().lower() for kw in keywords.split(" OR ")]  # Séparer les mots-clés
                    matching_posts = 0
                    total_posts = len(st.session_state.posts)

                    for post in st.session_state.posts:
                        post_text = post['text'].lower()
                        # Vérifier si au moins un mot-clé est présent
                        if any(keyword in post_text for keyword in keyword_list):
                            matching_posts += 1

                    # Calculer le pourcentage
                    match_percentage = (matching_posts / total_posts) * 100 if total_posts > 0 else 0
                    st.info(f"Pourcentage de posts contenant au moins un mot-clé : {match_percentage:.2f}% "
                            f"({matching_posts}/{total_posts})")
                else:
                    st.warning("Aucun post trouvé pour les mots-clés donnés.")

        # Display posts
        if st.session_state.posts:
            st.subheader(t("promotebluesky_posts"))
            for i, post in enumerate(st.session_state.posts):
                st.write(f"**@{post['handle']}**: {post['text']}")
                post_url = post.get('url', get_post_url(post['handle'], post['id']))
                st.markdown(f"[Voir le post]({post_url})")

                selected = st.checkbox(
                    f"Select Post {i+1}",
                    key=f"select_post_{i}"
                )
                st.session_state.selected_posts[i] = selected

        # Generate responses button
        if st.button(t("promotebluesky_generate_responses")) and st.session_state.selected_posts:
            with st.spinner(t("promotebluesky_generating")):
                selected_posts = [i for i, selected in st.session_state.selected_posts.items() if selected]
                st.session_state.generated_responses = self.generate_responses(
                    config, selected_posts, transcript, url
                )

        # Display generated responses
        if st.session_state.generated_responses:
            st.subheader(t("promotebluesky_responses"))
            for i, response in enumerate(st.session_state.generated_responses):
                st.write(f"**Response to Post {response['post_id']+1}**:")
                edited_response = st.text_area(
                    f"Edit Response {i+1}",
                    response['response'],
                    key=f"response_{i}",
                    height=100
                )
                st.session_state.generated_responses[i]['response'] = edited_response

                if len(edited_response) > 300:
                    st.warning(f"⚠️ Cette réponse dépasse 300 caractères ({len(edited_response)} caractères). Veuillez la raccourcir.")

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button(f"Copier la réponse {i+1}", key=f"copy_response_{i}"):
                        pyperclip.copy(edited_response)
                        st.success("Réponse copiée dans le presse-papiers !")
                with col2:
                    post_url = get_post_url(st.session_state.posts[response['post_id']]['handle'], st.session_state.posts[response['post_id']]['id'])
                    st.markdown(f"[Répondre à ce post]({post_url})", unsafe_allow_html=True)
                with col3:
                    selected = st.checkbox(
                        f"Select Response {i+1}",
                        key=f"select_response_{i}"
                    )
                    st.session_state.selected_responses[i] = selected

        # Post responses button
        if st.button(t("promotebluesky_post_responses")) and st.session_state.selected_responses:
            with st.spinner(t("promotebluesky_posting")):
                selected_responses = [
                    response for i, response in enumerate(st.session_state.generated_responses)
                    if st.session_state.selected_responses[i]
                ]
                self.post_responses(config, selected_responses)
                st.success(t("promotebluesky_success"))
