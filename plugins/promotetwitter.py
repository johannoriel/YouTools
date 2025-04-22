from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import os
from typing import List, Dict, Any, Optional
# Utilisation de l'API Twitter depuis social_api.py
from social_api import TwitterAPI
import pyperclip  # Pour copier le texte en un clic

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "promotetwitter_tab": "Promote Twitter",
    "promotetwitter_header": "Promote Content",
    "promotetwitter_transcript": "Transcript",
    "promotetwitter_url": "Video URL",
    "promotetwitter_keywords": "Keywords to Search",
    "promotetwitter_search": "Search Tweets",
    "promotetwitter_searching": "Searching tweets...",
    "promotetwitter_tweets": "Recent Tweets",
    "promotetwitter_select_tweets": "Select Tweets to Respond",
    "promotetwitter_generate_responses": "Generate Responses",
    "promotetwitter_generating": "Generating responses...",
    "promotetwitter_responses": "Suggested Responses",
    "promotetwitter_post_responses": "Post Responses",
    "promotetwitter_posting": "Posting responses...",
    "promotetwitter_success": "Responses posted successfully!",
    "promotetwitter_error": "Error posting responses: ",
})

translations["fr"].update({
    "promotetwitter_tab": "Promotion Twitter",
    "promotetwitter_header": "Promouvoir le Contenu",
    "promotetwitter_transcript": "Transcription",
    "promotetwitter_url": "URL de la vidéo",
    "promotetwitter_keywords": "Mots-clés à rechercher",
    "promotetwitter_search": "Rechercher des tweets",
    "promotetwitter_searching": "Recherche des tweets...",
    "promotetwitter_tweets": "Tweets récents",
    "promotetwitter_select_tweets": "Sélectionner des tweets pour répondre",
    "promotetwitter_generate_responses": "Générer des réponses",
    "promotetwitter_generating": "Génération des réponses...",
    "promotetwitter_responses": "Réponses suggérées",
    "promotetwitter_post_responses": "Poster les réponses",
    "promotetwitter_posting": "Publication des réponses...",
    "promotetwitter_success": "Réponses publiées avec succès !",
    "promotetwitter_error": "Erreur lors de la publication : ",
})


def remove_quotes(text: str) -> str:
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text


def get_tweet_url(username: str, tweet_id: str) -> str:
    return f"https://twitter.com/{username}/status/{tweet_id}"


class PromotetwitterPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self._initialize_session_state()

    def get_tweet_url(self, username: str, tweet_id: str) -> str:
        return f"https://twitter.com/{username}/status/{tweet_id}"

    def _initialize_session_state(self):
        if 'tweets' not in st.session_state:
            st.session_state.tweets = []
        if 'selected_tweets' not in st.session_state:
            st.session_state.selected_tweets = {}
        if 'generated_responses' not in st.session_state:
            st.session_state.generated_responses = []
        if 'selected_responses' not in st.session_state:
            st.session_state.selected_responses = {}

    def get_config_fields(self):
        return {
            "max_tweets": {
                "type": "number",
                "label": "Maximum Number of Tweets to Fetch",
                "default": 10
            },
            "response_prompt": {
                "type": "text",
                "label": "LLM Prompt for Responses",
                "default": """Suggère une réponse à ce tweet de moins de 280 caractères, en lien avec la vidéo dans l'URL {url} (doit être mentionnée). Le ton est direct, réponds comme si tu étais l'utilisateur, et en t'inspirant du transcript suivant : {transcript}"""
            }
        }

    def get_tabs(self):
        return [{"name": t("promotetwitter_tab"), "plugin": "promotetwitter"}]

    def search_tweets(self, query: str, max_tweets: int, api_version: str) -> List[Dict[str, Any]]:
        twitter_api = TwitterAPI(self.plugin_manager.config)
        if api_version == "v1":
            return twitter_api.search_v1(query, max_tweets)
        elif api_version == "v2":
            # Récupérer la langue depuis st.session_state.lang, avec 'fr' comme valeur par défaut
            language = st.session_state.get('lang', 'fr')
            return twitter_api.search_v2(query, max_tweets, language=language)
        else:
            st.error("Invalid API version selected.")
            return []

    def generate_responses(self, config, selected_tweets, transcript, url):
        responses = []

        for tweet_id in selected_tweets:
            tweet_text = st.session_state.tweets[tweet_id]['text']
            prompt = config['promotetwitter']['response_prompt'].format(
                url=url,
                transcript=transcript
            )
            llm_response = self.process_with_llm(
                prompt,
                config.get('llm', {}).get('llm_sys_prompt', ''),
                tweet_text
            )
            clean_response = remove_quotes(llm_response.strip())
            responses.append({
                'tweet_id': tweet_id,
                'response': clean_response
            })

        return responses

    def post_responses(self, config, selected_responses):
        twitter_api = TwitterAPI(self.plugin_manager.config)

        for response in selected_responses:
            tweet_id = response['tweet_id']
            response_text = response['response']
            twitter_api.create_tweet(
                response_text, in_reply_to_tweet_id=tweet_id)

    def has_llm_error(self, response_text: str) -> bool:
        return "litellm.APIError" in response_text

    def run(self, config):
        st.header(t("promotetwitter_header"))

        # Load or input transcript
        work_dir = config['common']['work_directory']
        transcript_path = os.path.join(work_dir, "transcript.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                transcript = f.read()
            st.text_area(t("promotetwitter_transcript"), transcript,
                         height=100, disabled=True, key="promotetwitter_transcript")
        else:
            transcript = st.text_area(
                t("promotetwitter_transcript"), height=200, key="promotetwitter_transcript")

        # Load or input URL
        url_path = os.path.join(work_dir, "url.txt")
        if os.path.exists(url_path):
            with open(url_path, 'r') as f:
                url = f.read().strip()
            st.text_input(t("promotetwitter_url"), url,
                          disabled=True, key="promotetwitter_url")
        else:
            url = st.text_input(t("promotetwitter_url"),
                                key="promotetwitter_url")

        # Input keywords to search
        keywords = st.text_input(
            t("promotetwitter_keywords"), key="promotetwitter_keywords")
        if not keywords:
            st.warning("Please enter keywords to search for tweets.")
            return

        # Select API version
        api_version = st.selectbox(
            "Select Twitter API Version",
            options=["v1", "v2"],
            index=1  # Default to v2
        )

        # Search tweets button
        if st.button(t("promotetwitter_search")):
            with st.spinner(t("promotetwitter_searching")):
                max_tweets = config['promotetwitter']['max_tweets']
                st.session_state.tweets = self.search_tweets(
                    keywords, max_tweets, api_version)

        # Display tweets
        if st.session_state.tweets:
            st.subheader(t("promotetwitter_tweets"))
            for i, tweet in enumerate(st.session_state.tweets):
                st.write(f"**@{tweet['user']}**: {tweet['text']}")

                # Utilisation de l'URL reconstruite si tweet['url'] est vide
                tweet_url = tweet.get(
                    'url', get_tweet_url(tweet['user'], tweet['id']))
                # Ajout du lien vers le tweet
                st.markdown(f"[Voir le tweet]({tweet_url})")

                selected = st.checkbox(
                    f"Select Tweet {i+1}",
                    key=f"select_tweet_{i}"
                )
                st.session_state.selected_tweets[i] = selected

        # Generate responses button
        if st.button(t("promotetwitter_generate_responses")) and st.session_state.selected_tweets:
            with st.spinner(t("promotetwitter_generating")):
                selected_tweets = [
                    i for i, selected in st.session_state.selected_tweets.items() if selected]
                st.session_state.generated_responses = self.generate_responses(
                    config, selected_tweets, transcript, url
                )

        # Display generated responses
        if st.session_state.generated_responses:
            st.subheader(t("promotetwitter_responses"))
            error_count = 0  # Compteur d'erreurs

            for i, response in enumerate(st.session_state.generated_responses):
                # Récupérer le tweet original
                tweet_id = response['tweet_id']
                tweet = st.session_state.tweets[tweet_id]
                tweet_text = tweet['text']
                tweet_user = tweet['user']

                # Afficher le tweet original
                st.write(f"**Tweet original de @{tweet_user}**:")
                st.write(tweet_text)
                # Lien vers le tweet
                st.markdown(
                    f"[Voir le tweet]({self.get_tweet_url(tweet_user, tweet['id'])})")

                # Afficher la réponse générée
                st.write(f"**Réponse générée pour ce tweet**:")
                edited_response = st.text_area(
                    f"Edit Response {i+1}",
                    response['response'],
                    key=f"response_{i}",
                    height=100
                )
                st.session_state.generated_responses[i]['response'] = edited_response

                # Vérification de la longueur de la réponse
                if len(edited_response) > 280:
                    st.warning(
                        f"⚠️ Cette réponse dépasse 280 caractères ({len(edited_response)} caractères). Veuillez la raccourcir.")

                # Vérification des erreurs LLM
                if self.has_llm_error(edited_response):
                    error_count += 1
                    st.error("⚠️ Cette réponse contient une erreur LLM.")

                # Bouton pour copier la réponse et lien pour répondre au tweet
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button(f"Copier la réponse {i+1}", key=f"copy_response_{i}"):
                        pyperclip.copy(edited_response)
                        st.success("Réponse copiée dans le presse-papiers !")
                with col2:
                    tweet_url = self.get_tweet_url(tweet_user, tweet['id'])
                    st.markdown(
                        f"[Répondre à ce tweet]({tweet_url})", unsafe_allow_html=True)
                with col3:
                    selected = st.checkbox(
                        f"Select Response {i+1}",
                        key=f"select_response_{i}"
                    )
                    st.session_state.selected_responses[i] = selected

            # Afficher le décompte des erreurs
            st.write(f"**Erreurs LLM détectées : {error_count}**")

            # Bouton pour regénérer les réponses en erreur
            if error_count > 0:
                if st.button("Regénérer les réponses en erreur"):
                    with st.spinner("Regénération des réponses en erreur..."):
                        for i, response in enumerate(st.session_state.generated_responses):
                            if self.has_llm_error(response['response']):
                                # Regénérer la réponse pour ce tweet
                                tweet_id = response['tweet_id']
                                tweet_text = st.session_state.tweets[tweet_id]['text']
                                prompt = config['promotetwitter']['response_prompt'].format(
                                    url=url,
                                    transcript=transcript
                                )
                                llm_response = self.process_with_llm(
                                    prompt,
                                    config.get('llm', {}).get('llm_sys_prompt', ''),
                                    tweet_text
                                )
                                clean_response = remove_quotes(
                                    llm_response.strip())
                                st.session_state.generated_responses[i]['response'] = clean_response

                        st.success("Réponses regénérées avec succès !")
