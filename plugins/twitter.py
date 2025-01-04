from global_vars import translations, t
from app import Plugin
from plugins.common import list_all_video_files
from plugins.ragllm import RagllmPlugin
import streamlit as st
import os
import tweepy
import tempfile
import subprocess

translations["en"].update({
    "twitter_tab": "Twitter",
    "twitter_header": "Generate & Post Tweets",
    "twitter_input_method": "Input Method",
    "twitter_video": "Video Transcription",
    "twitter_text": "Direct Text Input",
    "twitter_file": "Text File",
    "twitter_select_video": "Select video to transcribe",
    "twitter_enter_text": "Enter or paste text",
    "twitter_select_file": "Select text file",
    "twitter_prompt": "LLM Prompt",
    "twitter_generate": "Generate Tweets",
    "twitter_generating": "Generating tweets...",
    "twitter_preview": "Preview and Edit Tweets",
    "twitter_select_tweets": "Select",
    "twitter_post": "Post Selected Tweets",
    "twitter_posting": "Posting tweets...",
    "twitter_success": "Tweets posted successfully!",
    "twitter_error": "Error posting tweets: ",
})

translations["fr"].update({
    "twitter_tab": "Twitter",
    "twitter_header": "Générer & Poster des Tweets",
    "twitter_input_method": "Méthode d'entrée",
    "twitter_video": "Transcription vidéo",
    "twitter_text": "Texte direct",
    "twitter_file": "Fichier texte",
    "twitter_select_video": "Sélectionner la vidéo à transcrire",
    "twitter_enter_text": "Entrez ou collez le texte",
    "twitter_select_file": "Sélectionner le fichier texte",
    "twitter_prompt": "Prompt LLM",
    "twitter_generate": "Générer les Tweets",
    "twitter_generating": "Génération des tweets...",
    "twitter_preview": "Prévisualiser et Éditer les Tweets",
    "twitter_select_tweets": "Sélectionner",
    "twitter_post": "Poster les Tweets Sélectionnés",
    "twitter_posting": "Publication des tweets...",
    "twitter_success": "Tweets publiés avec succès !",
    "twitter_error": "Erreur lors de la publication : ",
})

class TwitterPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.default_prompt = """A partir de ce transcript, agis en tant qu'infopreneur et crée une série de tweets pour teaser la vidéo en donnant les key insights.

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

    def get_config_fields(self):
        return {
            "bearer_token": {
                "type": "text",
                "label": "Twitter Bearer Token",
                "default": ""
            },
            "api_key": {
                "type": "text",
                "label": "Twitter API Key",
                "default": ""
            },
            "api_secret": {
                "type": "text",
                "label": "Twitter API Secret",
                "default": ""
            },
            "access_token": {
                "type": "text",
                "label": "Twitter Access Token",
                "default": ""
            },
            "access_token_secret": {
                "type": "text",
                "label": "Twitter Access Token Secret",
                "default": ""
            }
        }

    def get_tabs(self):
        return [{"name": t("twitter_tab"), "plugin": "twitter"}]

    def transcribe_video(self, video_path, whisper_path, whisper_model, ffmpeg_path, lang):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            try:
                ffmpeg_command = [
                    ffmpeg_path, '-y',
                    '-i', video_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    temp_audio_path
                ]
                subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_output:
                    output_file = temp_output.name

                expanded_whisper_path = os.path.expanduser(whisper_path)
                whisper_command = [
                    expanded_whisper_path,
                    "-m", f"{os.path.dirname(expanded_whisper_path)}/models/ggml-{whisper_model}.bin",
                    "-f", temp_audio_path,
                    "-l", lang,
                    "-of", os.path.splitext(output_file)[0],
                    "-otxt"
                ]
                subprocess.run(whisper_command, check=True, capture_output=True, text=True)

                with open(output_file, 'r') as f:
                    transcript = f.read()

                os.remove(output_file)
                return transcript

            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

    def parse_tweets(self, llm_response):
        tweets = []
        for tweet in llm_response.split('TWEET:')[1:]:
            clean_tweet = tweet.strip().split('---')[0].strip()
            if clean_tweet:
                tweets.append(clean_tweet)
        return tweets

    def post_tweets(self, tweets, bearer_token, api_key, api_secret, access_token, access_token_secret):
        client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )

        posted_tweets = []
        for tweet in tweets:
            try:
                response = client.create_tweet(text=tweet)
                posted_tweets.append(response)
            except Exception as e:
                st.error(f"{t('twitter_error')}{str(e)}")
                break
        return posted_tweets


        posted_tweets = []
        for tweet in tweets:
            try:
                #response = api.update_status(tweet)
                response = client.create_tweet(text=tweet)
                posted_tweets.append(response)
            except Exception as e:
                st.error(f"{t('twitter_error')}{str(e)}")
                break
        return posted_tweets

    def run(self, config):
        st.header(t("twitter_header"))

        if 'generated_tweets' not in st.session_state:
            st.session_state.generated_tweets = []
            st.session_state.selected_tweets = set()

        input_method = st.radio(t("twitter_input_method"),
            [t("twitter_video"), t("twitter_text"), t("twitter_file")])

        text_content = None

        if input_method == t("twitter_video"):
            videos = list_all_video_files(config['common']['work_directory'])
            if videos:
                selected_video = st.selectbox(t("twitter_select_video"),
                    options=[v[0] for v in videos])
                selected_video_path = next(v[1] for v in videos if v[0] == selected_video)
                if st.button(t("twitter_generate")):
                    with st.spinner(t("twitter_generating")):
                        text_content = self.transcribe_video(
                            selected_video_path,
                            config['transcript']['whisper_path'],
                            config['transcript']['whisper_model'],
                            config['transcript']['ffmpeg_path'],
                            config['common']['language']
                        )

        elif input_method == t("twitter_text"):
            text_content = st.text_area(t("twitter_enter_text"))

        else:  # Text file
            work_dir = config['common']['work_directory']
            files = [f for f in os.listdir(work_dir) if f.endswith('.txt')]
            if files:
                selected_file = st.selectbox(t("twitter_select_file"), options=files)
                with open(os.path.join(work_dir, selected_file), 'r') as f:
                    text_content = f.read()

        if text_content:
            prompt = st.text_area(t("twitter_prompt"), value=self.default_prompt)

            if st.button(t("twitter_generate")):
                ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
                llm_response = ragllm_plugin.process_with_llm(
                    prompt,
                    config.get('llm', {}).get('llm_sys_prompt', ''),
                    text_content
                )
                st.session_state.generated_tweets = self.parse_tweets(llm_response)
                st.session_state.selected_tweets = set()

            if st.session_state.generated_tweets:
                st.subheader(t("twitter_preview"))
                for i, tweet in enumerate(st.session_state.generated_tweets):
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        selected = st.checkbox(t("twitter_select_tweets"), key=f"select_{i}",
                                            value=i in st.session_state.selected_tweets)
                        if selected:
                            st.session_state.selected_tweets.add(i)
                        elif i in st.session_state.selected_tweets:
                            st.session_state.selected_tweets.remove(i)
                    with col2:
                        edited_tweet = st.text_area(f"Tweet {i+1}", tweet, key=f"tweet_{i}",
                                                  height=100)
                        st.session_state.generated_tweets[i] = edited_tweet

                selected_tweets = [st.session_state.generated_tweets[i]
                                 for i in st.session_state.selected_tweets]

                if selected_tweets and st.button(t("twitter_post")):
                    with st.spinner(t("twitter_posting")):
                        self.post_tweets(
                            selected_tweets,
                            config['twitter']['bearer_token'],
                            config['twitter']['api_key'],
                            config['twitter']['api_secret'],
                            config['twitter']['access_token'],
                            config['twitter']['access_token_secret']
                        )
                        st.success(t("twitter_success"))
