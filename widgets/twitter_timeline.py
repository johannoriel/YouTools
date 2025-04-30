# widgets/twittertimeline.py
from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from lib.social_api import TwitterAPI

# Ajout des traductions spécifiques au widget
translations["en"].update({
    "promotetwitter_timeline_header": "Timeline des abonnements",
    "promotetwitter_error_user_id": "Please configure your User ID in the settings.",
    "promotetwitter_use_api_v1": "Use API v1 (limited to 20 tweets)",
    "promotetwitter_rate_limit": "Rate Limit Status",
    "promotetwitter_timeline_label": "**Timeline (home_timeline)**:",
    "promotetwitter_requests_remaining": "Remaining requests: {remaining} out of {limit}",
    "promotetwitter_reset": "Reset: {reset}",
    "promotetwitter_rate_limit_warning": "Unable to retrieve rate limits for the timeline.",
    "promotetwitter_update_label": "**Tweet posting (update)**:",
    "promotetwitter_rate_limit_warning_update": "Unable to retrieve rate limits for posting.",
    "promotetwitter_fetch_timeline": "Fetch Timeline",
    "promotetwitter_fetching": "Fetching tweets...",
    "promotetwitter_recent_threads": "Recent Threads from Followings",
    "promotetwitter_language": "Language",
    "promotetwitter_source": "Source",
    "promotetwitter_metrics": "Metrics",
    "promotetwitter_select_root": "Select to reply to the root tweet",
    "promotetwitter_thread_replies": "**Replies in this thread**:",
    "promotetwitter_select_reply": "Select to reply",
    "promotetwitter_reply_selected": "Reply to the selected tweet",
    "promotetwitter_your_response": "Your response (max 280 characters)",
    "promotetwitter_response_too_long": "The response exceeds 280 characters ({length}).",
    "promotetwitter_post_response": "Post the response",
    "promotetwitter_posting_response": "Posting the response...",
    "promotetwitter_success_response": "Response posted successfully!",
    "promotetwitter_error_response": "Error posting the response.",
    "promotetwitter_empty_response": "Please enter a response."
})

translations["fr"].update({
    "promotetwitter_timeline_header": "Timeline des abonnements",
    "promotetwitter_error_user_id": "Veuillez configurer votre User ID dans les paramètres.",
    "promotetwitter_use_api_v1": "Utiliser l'API v1 (limité à 20 tweets)",
    "promotetwitter_rate_limit": "Statut des limites de taux",
    "promotetwitter_timeline_label": "**Timeline (home_timeline)** :",
    "promotetwitter_requests_remaining": "Requêtes restantes : {remaining} sur {limit}",
    "promotetwitter_reset": "Réinitialisation : {reset}",
    "promotetwitter_rate_limit_warning": "Impossible de récupérer les limites de taux pour la timeline.",
    "promotetwitter_update_label": "**Publication de tweets (update)** :",
    "promotetwitter_rate_limit_warning_update": "Impossible de récupérer les limites de taux pour la publication.",
    "promotetwitter_fetch_timeline": "Récupérer la timeline",
    "promotetwitter_fetching": "Récupération des tweets...",
    "promotetwitter_recent_threads": "Threads récents des abonnements",
    "promotetwitter_language": "Langue",
    "promotetwitter_source": "Source",
    "promotetwitter_metrics": "Métriques",
    "promotetwitter_select_root": "Sélectionner pour répondre au tweet racine",
    "promotetwitter_thread_replies": "**Réponses dans ce thread** :",
    "promotetwitter_select_reply": "Sélectionner pour répondre",
    "promotetwitter_reply_selected": "Répondre au tweet sélectionné",
    "promotetwitter_your_response": "Votre réponse (max 280 caractères)",
    "promotetwitter_response_too_long": "La réponse dépasse 280 caractères ({length}).",
    "promotetwitter_post_response": "Poster la réponse",
    "promotetwitter_posting_response": "Publication de la réponse...",
    "promotetwitter_success_response": "Réponse publiée avec succès !",
    "promotetwitter_error_response": "Erreur lors de la publication de la réponse.",
    "promotetwitter_empty_response": "Veuillez entrer une réponse."
})

class TwitterTimelineWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'tweets' not in st.session_state:
            st.session_state.tweets = []
        if 'selected_tweet' not in st.session_state:
            st.session_state.selected_tweet = None

    def display(self, config):
        st.header(t("promotetwitter_timeline_header"))

        user_id = config['promotetwitter'].get('user_id', '')
        if not user_id:
            st.error(t("promotetwitter_error_user_id"))
            return

        use_api_v1 = st.checkbox(
            t("promotetwitter_use_api_v1"),
            value=False,
            key=f"{self.prefix}_use_api_v1"
        )

        twitter_api = TwitterAPI(self.plugin_manager.config)
        if use_api_v1:
            rate_limit = twitter_api.get_rate_limit_status()
            st.subheader(t("promotetwitter_rate_limit"))
            st.write(t("promotetwitter_timeline_label"))
            if rate_limit['timeline']['remaining'] is not None:
                st.write(
                    t("promotetwitter_requests_remaining").format(
                        remaining=rate_limit['timeline']['remaining'],
                        limit=rate_limit['timeline']['limit']
                    ))
                st.write(t("promotetwitter_reset").format(reset=rate_limit['timeline']['reset']))
            else:
                st.warning(t("promotetwitter_price_limit_warning"))
            st.write(t("promotetwitter_update_label"))
            if rate_limit['update']['remaining'] is not None:
                st.write(
                    t("promotetwitter_requests_remaining").format(
                        remaining=rate_limit['update']['remaining'],
                        limit=rate_limit['update']['limit']
                    ))
                st.write(t("promotetwitter_reset").format(reset=rate_limit['update']['reset']))
            else:
                st.warning(t("promotetwitter_rate_limit_warning_update"))

        if st.button(t("promotetwitter_fetch_timeline"), key=f"{self.prefix}_fetch_timeline"):
            with st.spinner(t("promotetwitter_fetching")):
                if use_api_v1:
                    st.session_state.tweets = twitter_api.get_following_timeline_v1(max_results=20)
                else:
                    st.session_state.tweets = twitter_api.get_following_timeline(user_id, max_results=100)

        if st.session_state.tweets:
            st.subheader(t("promotetwitter_recent_threads"))
            threads = twitter_api.organize_tweets_into_threads(st.session_state.tweets)
            for i, thread in enumerate(threads):
                root_tweet = thread['root_tweet']
                with st.expander(f"Thread de @{root_tweet['user']} - {root_tweet['created_at']}"):
                    st.image(root_tweet['profile_image_url'], width=50)
                    st.write(f"**{root_tweet['name']} (@{root_tweet['user']})**")
                    st.write(root_tweet['text'])
                    st.write(f"**{t('promotetwitter_language')}** : {root_tweet['lang']}")
                    st.write(f"**{t('promotetwitter_source')}** : {root_tweet['source']}")
                    st.write(f"**{t('promotetwitter_metrics')}** : {root_tweet['public_metrics']['like_count']} likes, "
                             f"{root_tweet['public_metrics']['retweet_count']} retweets, "
                             f"{root_tweet['public_metrics']['reply_count']} réponses, "
                             f"{root_tweet['public_metrics']['quote_count']} citations")
                    st.markdown(f"[Voir le tweet]({root_tweet['url']})")
                    if st.button(
                        t("promotetwitter_select_root"),
                        key=f"{self.prefix}_select_root_tweet_{i}"
                    ):
                        st.session_state.selected_tweet = root_tweet

                    if thread['replies']:
                        st.write(t("promotetwitter_thread_replies"))
                        for j, reply in enumerate(thread['replies']):
                            with st.expander(f"Réponse de @{reply['user']} - {reply['created_at']}"):
                                st.image(reply['profile_image_url'], width=50)
                                st.write(f"**{reply['name']} (@{reply['user']})**")
                                st.write(reply['text'])
                                st.write(f"**{t('promotetwitter_language')}** : {reply['lang']}")
                                st.write(f"**{t('promotetwitter_source')}** : {reply['source']}")
                                st.write(f"**{t('promotetwitter_metrics')}** : {reply['public_metrics']['like_count']} likes, "
                                         f"{reply['public_metrics']['retweet_count']} retweets, "
                                         f"{reply['public_metrics']['reply_count']} réponses, "
                                         f"{reply['public_metrics']['quote_count']} citations")
                                st.markdown(f"[Voir le tweet]({reply['url']})")
                                if st.button(
                                    t("promotetwitter_select_reply"),
                                    key=f"{self.prefix}_select_reply_tweet_{i}_{j}"
                                ):
                                    st.session_state.selected_tweet = reply

        if st.session_state.selected_tweet:
            st.subheader(t("promotetwitter_reply_selected"))
            tweet = st.session_state.selected_tweet
            st.write(f"**Tweet sélectionné de @{tweet['user']}** : {tweet['text']}")
            response_text = st.text_area(
                t("promotetwitter_your_response"),
                max_chars=280,
                key=f"{self.prefix}_manual_response"
            )
            if len(response_text) > 280:
                st.warning(t("promotetwitter_response_too_long").format(length=len(response_text)))
            if st.button(t("promotetwitter_post_response"), key=f"{self.prefix}_post_response"):
                if response_text:
                    with st.spinner(t("promotetwitter_posting_response")):
                        if use_api_v1:
                            response = twitter_api.create_tweet_v1(text=response_text, in_reply_to_tweet_id=tweet['id'])
                        else:
                            response = twitter_api.create_tweet(text=response_text, in_reply_to_tweet_id=tweet['id'])
                        if response:
                            st.success(t("promotetwitter_success_response"))
                            st.session_state.selected_tweet = None
                        else:
                            st.error(t("promotetwitter_error_response"))
                else:
                    st.warning(t("promotetwitter_empty_response"))
