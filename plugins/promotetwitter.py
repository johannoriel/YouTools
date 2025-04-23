from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import os
from typing import List, Dict, Any, Optional
# Utilisation de l'API Twitter depuis social_api.py
from lib.social_api import TwitterAPI
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
            },
            "user_id": {
                "type": "text",
                "label": "Twitter User ID",
                "default": ""
            }
        }

    def get_tabs(self):
        return [
            {"name": t("promotetwitter_tab"), "plugin": "promotetwitter"},
            {"name": "Timeline", "plugin": "promotetwitter_timeline"}
        ]

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

    def run_timeline(self, config):
        st.header("Timeline des abonnements")

        # Vérifier la présence du user_id
        user_id = config['promotetwitter'].get('user_id', '')
        if not user_id:
            st.error("Veuillez configurer votre User ID dans les paramètres.")
            return

        # Case à cocher pour choisir l'API
        use_api_v1 = st.checkbox(
            "Utiliser l'API v1 (limité à 20 tweets)", value=False)

        # Afficher les limites de taux
        if use_api_v1:
            twitter_api = TwitterAPI(self.plugin_manager.config)
            rate_limit = twitter_api.get_rate_limit_status()
            st.subheader("Statut des limites de taux")
            st.write("**Timeline (home_timeline)** :")
            if rate_limit['timeline']['remaining'] is not None:
                st.write(
                    f"Requêtes restantes : {rate_limit['timeline']['remaining']} sur {rate_limit['timeline']['limit']}")
                st.write(
                    f"Réinitialisation : {rate_limit['timeline']['reset']}")
            else:
                st.warning(
                    "Impossible de récupérer les limites de taux pour la timeline.")
            st.write("**Publication de tweets (update)** :")
            if rate_limit['update']['remaining'] is not None:
                st.write(
                    f"Requêtes restantes : {rate_limit['update']['remaining']} sur {rate_limit['update']['limit']}")
                st.write(f"Réinitialisation : {rate_limit['update']['reset']}")
            else:
                st.warning(
                    "Impossible de récupérer les limites de taux pour la publication.")

        # Bouton pour récupérer la timeline
        if st.button("Récupérer la timeline"):
            with st.spinner("Récupération des tweets..."):
                if use_api_v1:
                    st.session_state.tweets = twitter_api.get_following_timeline_v1(
                        max_results=20)
                else:
                    st.session_state.tweets = twitter_api.get_following_timeline(
                        user_id, max_results=100)

        # Afficher les threads
        if st.session_state.tweets:
            st.subheader("Threads récents des abonnements")
            threads = twitter_api.organize_tweets_into_threads(
                st.session_state.tweets)
            selected_tweet = None
            for i, thread in enumerate(threads):
                root_tweet = thread['root_tweet']
                with st.expander(f"Thread de @{root_tweet['user']} - {root_tweet['created_at']}"):
                    # Afficher le tweet racine
                    st.image(root_tweet['profile_image_url'], width=50)
                    st.write(
                        f"**{root_tweet['name']} (@{root_tweet['user']})**")
                    st.write(root_tweet['text'])
                    st.write(f"**Langue** : {root_tweet['lang']}")
                    st.write(f"**Source** : {root_tweet['source']}")
                    st.write(f"**Métriques** : {root_tweet['public_metrics']['like_count']} likes, "
                             f"{root_tweet['public_metrics']['retweet_count']} retweets, "
                             f"{root_tweet['public_metrics']['reply_count']} réponses, "
                             f"{root_tweet['public_metrics']['quote_count']} citations")
                    st.markdown(f"[Voir le tweet]({root_tweet['url']})")
                    if st.button(f"Sélectionner pour répondre au tweet racine", key=f"select_root_tweet_{i}"):
                        selected_tweet = root_tweet
                        st.session_state.selected_tweet = root_tweet

                    # Afficher les réponses dans des sous-expanders
                    if thread['replies']:
                        st.write("**Réponses dans ce thread** :")
                        for j, reply in enumerate(thread['replies']):
                            with st.expander(f"Réponse de @{reply['user']} - {reply['created_at']}"):
                                st.image(reply['profile_image_url'], width=50)
                                st.write(
                                    f"**{reply['name']} (@{reply['user']})**")
                                st.write(reply['text'])
                                st.write(f"**Langue** : {reply['lang']}")
                                st.write(f"**Source** : {reply['source']}")
                                st.write(f"**Métriques** : {reply['public_metrics']['like_count']} likes, "
                                         f"{reply['public_metrics']['retweet_count']} retweets, "
                                         f"{reply['public_metrics']['reply_count']} réponses, "
                                         f"{reply['public_metrics']['quote_count']} citations")
                                st.markdown(f"[Voir le tweet]({reply['url']})")
                                if st.button(f"Sélectionner pour répondre", key=f"select_reply_tweet_{i}_{j}"):
                                    selected_tweet = reply
                                    st.session_state.selected_tweet = reply

        # Section pour répondre manuellement
        if 'selected_tweet' in st.session_state and st.session_state.selected_tweet:
            st.subheader("Répondre au tweet sélectionné")
            tweet = st.session_state.selected_tweet
            st.write(
                f"**Tweet sélectionné de @{tweet['user']}** : {tweet['text']}")
            response_text = st.text_area(
                "Votre réponse (max 280 caractères)", max_chars=280, key="manual_response")
            if len(response_text) > 280:
                st.warning(
                    f"La réponse dépasse 280 caractères ({len(response_text)}).")
            if st.button("Poster la réponse"):
                if response_text:
                    with st.spinner("Publication de la réponse..."):
                        if use_api_v1:
                            response = twitter_api.create_tweet_v1(
                                text=response_text,
                                in_reply_to_tweet_id=tweet['id']
                            )
                        else:
                            response = twitter_api.create_tweet(
                                text=response_text,
                                in_reply_to_tweet_id=tweet['id']
                            )
                        if response:
                            st.success("Réponse publiée avec succès !")
                            st.session_state.selected_tweet = None  # Réinitialiser la sélection
                        else:
                            st.error(
                                "Erreur lors de la publication de la réponse.")
                else:
                    st.warning("Veuillez entrer une réponse.")

    def run_post(self, config):
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
                                    config.get('llm', {}).get(
                                        'llm_sys_prompt', ''),
                                    tweet_text
                                )
                                clean_response = remove_quotes(
                                    llm_response.strip())
                                st.session_state.generated_responses[i]['response'] = clean_response

                        st.success("Réponses regénérées avec succès !")

    def run(self, config):
        """Main plugin logic"""
        tab1, tab2 = st.tabs(["Poster", "Répondre"])
        with tab1:
            self.run_post(config)
        with tab2:
            self.run_timeline(config)
