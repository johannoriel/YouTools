from global_vars import translations, t
from app import Plugin
import streamlit as st
import os
import tweepy
from atproto import Client as AtprotoClient, models
from atproto import client_utils
from plugins.ragllm import RagllmPlugin
import telegram
import json
import re

translations["en"].update({
    "social_tab": "Social Networks",
    "social_header": "Generate & Post Content",
    "social_transcript": "Transcript",
    "social_url": "Video URL",
    "social_prompt": "LLM Prompt",
    "social_generate": "Generate Posts",
    "social_generating": "Generating posts...",
    "social_preview": "Preview and Edit Posts",
    "social_manual_post": "Add Manual Post",
    "social_post": "Post Selected Content",
    "social_posting": "Posting content...",
    "social_success": "Content posted successfully!",
    "social_error": "Error posting content: ",
    "social_twitter": "Twitter",
    "social_bluesky": "Bluesky",
    "social_telegram": "Telegram",
    "social_add_post": "Add Post",
})

translations["fr"].update({
    "social_tab": "Réseaux Sociaux",
    "social_header": "Générer & Poster du Contenu",
    "social_transcript": "Transcription",
    "social_url": "URL de la vidéo",
    "social_prompt": "Prompt LLM",
    "social_generate": "Générer les posts",
    "social_generating": "Génération des posts...",
    "social_preview": "Prévisualiser et Éditer",
    "social_manual_post": "Ajouter un post manuel",
    "social_post": "Poster la sélection",
    "social_posting": "Publication en cours...",
    "social_success": "Contenu publié avec succès !",
    "social_error": "Erreur lors de la publication : ",
    "social_twitter": "Twitter",
    "social_bluesky": "Bluesky",
    "social_telegram": "Telegram",
    "social_add_post": "Ajouter un post",
})

class SocialPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        if 'generated_posts' not in st.session_state:
            st.session_state.generated_posts = []
        if 'selected_platforms' not in st.session_state:
            st.session_state.selected_platforms = {}
        if 'manual_post_start' not in st.session_state:
            st.session_state.manual_post_start = ""
        if 'manual_post_end' not in st.session_state:
            st.session_state.manual_post_end = ""

    def get_config_fields(self):
        return {
            "default_prompt": {
                "type": "text",
                "label": "Default LLM Prompt",
                "default": """A partir de ce transcript, agis en tant qu'infopreneur et crée une série de tweets pour teaser la vidéo en donnant les key insights.

Format de sortie attendu :
TWEET:
[Contenu du premier tweet]
---
TWEET:
[Contenu du deuxième tweet]
---
TWEET:
[Contenu du troisième tweet]
etc...

Chaque tweet doit faire maximum 280 caractères."""
            },
            "url_suffix_template": {
                "type": "text",
                "label": "URL Suffix Template",
                "default": "Voir plus dans la vidéo : {url}"
            },
            "twitter_bearer_token": {
                "type": "text",
                "label": "Twitter Bearer Token",
                "default": ""
            },
            "twitter_api_key": {
                "type": "text",
                "label": "Twitter API Key",
                "default": ""
            },
            "twitter_api_secret": {
                "type": "text",
                "label": "Twitter API Secret",
                "default": ""
            },
            "twitter_access_token": {
                "type": "text",
                "label": "Twitter Access Token",
                "default": ""
            },
            "twitter_access_token_secret": {
                "type": "text",
                "label": "Twitter Access Token Secret",
                "default": ""
            },
            "bluesky_handle": {
                "type": "text",
                "label": "Bluesky Handle",
                "default": ""
            },
            "bluesky_password": {
                "type": "text",
                "label": "Bluesky App Password",
                "default": ""
            },
            "telegram_bot_token": {
                "type": "text",
                "label": "Telegram Bot Token",
                "default": ""
            },
            "telegram_channel_id": {
                "type": "text",
                "label": "Telegram Channel ID",
                "default": ""
            }
        }

    def get_tabs(self):
        return [{"name": t("social_tab"), "plugin": "social"}]

    def parse_posts(self, llm_response):
        posts = []
        for post in llm_response.split('TWEET:')[1:]:
            clean_post = post.strip().split('---')[0].strip()
            if clean_post:
                posts.append(clean_post)
        return posts

    def simple_post_to_twitter(self, text, config):
        try:
            client = tweepy.Client(
                bearer_token=config['social']['twitter_bearer_token'],
                consumer_key=config['social']['twitter_api_key'],
                consumer_secret=config['social']['twitter_api_secret'],
                access_token=config['social']['twitter_access_token'],
                access_token_secret=config['social']['twitter_access_token_secret']
            )
            response = client.create_tweet(text=text)
            return response
        except Exception as e:
            st.error(f"Twitter: {str(e)}")
            return None

    def create_twitter_thread(self, posts, config):
        try:
            client = tweepy.Client(
                bearer_token=config['social']['twitter_bearer_token'],
                consumer_key=config['social']['twitter_api_key'],
                consumer_secret=config['social']['twitter_api_secret'],
                access_token=config['social']['twitter_access_token'],
                access_token_secret=config['social']['twitter_access_token_secret']
            )

            previous_tweet_id = None
            responses = []

            for post in posts:
                if previous_tweet_id:
                    response = client.create_tweet(
                        text=post,
                        in_reply_to_tweet_id=previous_tweet_id
                    )
                else:
                    response = client.create_tweet(text=post)

                previous_tweet_id = response.data['id']
                responses.append(response)

            return responses
        except Exception as e:
            st.error(f"Twitter: {str(e)}")
            return None

    def simple_post_to_bluesky(self, text, config):
        try:
            client = AtprotoClient()
            client.login(config['social']['bluesky_handle'], config['social']['bluesky_password'])
            response = client.send_post(text=text)
            return response
        except Exception as e:
            st.error(f"Bluesky: {str(e)}")
            return None

    def prepare_bluesky_post(self, text):
        # URL pattern matching
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        # Find all URLs in the text
        urls = re.findall(url_pattern, text)

        if not urls:
            return text

        # Use TextBuilder to properly handle URLs
        builder = client_utils.TextBuilder()

        # Split text by URLs and rebuild with proper formatting
        segments = re.split(url_pattern, text)
        for i, segment in enumerate(segments):
            if segment:
                builder.text(segment)
            if i < len(urls):
                builder.link(urls[i], urls[i])

        return builder

    def create_bluesky_thread(self, posts, config):
        try:
            client = AtprotoClient()
            client.login(config['social']['bluesky_handle'], config['social']['bluesky_password'])

            responses = []
            root_ref = None
            parent_ref = None

            for i, post in enumerate(posts):
                # Prepare post content with proper URL handling
                prepared_text = self.prepare_bluesky_post(post)

                # Create the post
                if root_ref is None:
                    # First post in thread
                    if isinstance(prepared_text, client_utils.TextBuilder):
                        response = client.send_post(text_builder=prepared_text)
                    else:
                        response = client.send_post(text=prepared_text)
                    root_ref = models.create_strong_ref(response)
                    parent_ref = root_ref
                else:
                    # Reply posts
                    reply_ref = models.AppBskyFeedPost.ReplyRef(
                        root=root_ref,
                        parent=parent_ref
                    )

                    if isinstance(prepared_text, client_utils.TextBuilder):
                        response = client.send_post(text=prepared_text, reply_to=reply_ref)
                    else:
                        response = client.send_post(text=prepared_text, reply_to=reply_ref)
                    parent_ref = models.create_strong_ref(response)

                responses.append(response)

            return responses
        except Exception as e:
            st.error(f"Bluesky: {str(e)}")
            return None

    def post_to_telegram(self, text, config):
        try:
            bot = telegram.Bot(token=config['social']['telegram_bot_token'])
            response = bot.send_message(chat_id=config['social']['telegram_channel_id'], text=text)
            return response
        except Exception as e:
            st.error(f"Telegram: {str(e)}")
            return None

    def run(self, config):
        st.header(t("social_header"))

        # Load or input transcript
        work_dir = config['common']['work_directory']
        transcript_path = os.path.join(work_dir, "transcript.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                transcript = f.read()
            st.text_area(t("social_transcript"), transcript, height=100, disabled=True)
        else:
            transcript = st.text_area(t("social_transcript"), height=200)

        # Load or input URL
        url_path = os.path.join(work_dir, "url.txt")
        if os.path.exists(url_path):
            with open(url_path, 'r') as f:
                url = f.read().strip()
            st.text_input(t("social_url"), url, disabled=True)
        else:
            url = st.text_input(t("social_url"))


        # Generate posts
        prompt = st.text_area(t("social_prompt"), value=config['social']['default_prompt'])


        # Manual post at start
        st.subheader("Post manuel de début")
        manual_post_start = st.text_area("", key="manual_post_start", height=100)
        cols = st.columns(3)
        with cols[0]:
            start_twitter = st.checkbox(t("social_twitter"), key="start_twitter")
        with cols[1]:
            start_bluesky = st.checkbox(t("social_bluesky"), key="start_bluesky")
        with cols[2]:
            start_telegram = st.checkbox(t("social_telegram"), key="start_telegram")

        # Manual post at end
        st.subheader("Post manuel de fin")
        manual_post_end = st.text_area("", key="manual_post_end", height=100)
        cols = st.columns(3)
        with cols[0]:
            end_twitter = st.checkbox(t("social_twitter"), key="end_twitter")
        with cols[1]:
            end_bluesky = st.checkbox(t("social_bluesky"), key="end_bluesky")
        with cols[2]:
            end_telegram = st.checkbox(t("social_telegram"), key="end_telegram")


        if st.button(t("social_generate")) and transcript:
            with st.spinner(t("social_generating")):
                ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
                llm_response = ragllm_plugin.process_with_llm(
                    prompt,
                    config.get('llm', {}).get('llm_sys_prompt', ''),
                    transcript
                )

                # Reset and rebuild posts list
                st.session_state.generated_posts = []

                # Add manual start post if present
                if manual_post_start:
                    st.session_state.generated_posts.append(manual_post_start)

                # Add generated posts
                st.session_state.generated_posts.extend(self.parse_posts(llm_response))

                # Add URL suffix if present
                if url:
                    url_suffix = config['social']['url_suffix_template'].format(url=url)
                    st.session_state.generated_posts.append(url_suffix)

                # Add manual end post if present
                if st.session_state.manual_post_end:
                    st.session_state.generated_posts.append(st.session_state.manual_post_end)

                st.session_state.selected_platforms = {}


        # Display and edit posts
        if st.session_state.generated_posts:
            st.subheader(t("social_preview"))
            for i, post in enumerate(st.session_state.generated_posts):
                st.text_area(f"Post {i+1}", post, key=f"post_{i}", height=100)

                cols = st.columns(3)
                is_manual_start = i == 0 and manual_post_start
                is_manual_end = i == len(st.session_state.generated_posts) - 1 and manual_post_end

                with cols[0]:
                    twitter = st.checkbox(t("social_twitter"),
                                       key=f"twitter_{i}",
                                       value=start_twitter if is_manual_start else (
                                           end_twitter if is_manual_end else False))
                with cols[1]:
                    bluesky = st.checkbox(t("social_bluesky"),
                                        key=f"bluesky_{i}",
                                        value=start_bluesky if is_manual_start else (
                                            end_bluesky if is_manual_end else False))
                with cols[2]:
                    telegram = st.checkbox(t("social_telegram"),
                                         key=f"telegram_{i}",
                                         value=start_telegram if is_manual_start else (
                                             end_telegram if is_manual_end else False))

                st.session_state.selected_platforms[i] = {
                    'twitter': twitter,
                    'bluesky': bluesky,
                    'telegram': telegram
                }


        # Posting
        if st.button(t("social_post")):
            with st.spinner(t("social_posting")):
                # Collect posts by platform
                twitter_posts = []
                bluesky_posts = []

                for i, post in enumerate(st.session_state.generated_posts):
                    platforms = st.session_state.selected_platforms.get(i, {})
                    if platforms.get('twitter'):
                        twitter_posts.append(post)
                    if platforms.get('bluesky'):
                        bluesky_posts.append(post)
                    if platforms.get('telegram'):
                        self.post_to_telegram(post, config)

                # Send threaded posts
                if twitter_posts:
                    self.create_twitter_thread(twitter_posts, config)
                if bluesky_posts:
                    self.create_bluesky_thread(bluesky_posts, config)

                st.success(t("social_success"))
