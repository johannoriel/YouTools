from global_vars import translations, t
from app import Plugin
import streamlit as st
import os
from plugins.ragllm import RagllmPlugin
from typing import List, Dict, Any, Optional
from social_api import TwitterAPI, BlueskyAPI, TelegramAPI, GhostAPI

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
    "social_manual_start": "Manual Start Post",
    "social_manual_end": "Manual End Post",
    "social_select_all": "Select All",
    "social_deselect_all": "Deselect All",
    "social_validate": "Validate Character Count",
    "social_ghost": "Ghost",
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
    "social_manual_start": "Post manuel de début",
    "social_manual_end": "Post manuel de fin",
    "social_select_all": "Tout sélectionner",
    "social_deselect_all": "Tout désélectionner",
    "social_validate": "Vérifier le nombre de caractères",
    "social_ghost": "Ghost",
})


class SocialNetwork:
    def __init__(self, name: str, api_class: Any, config_fields: Dict[str, Dict[str, Any]],
                 post_method: str = 'post', max_chars: Optional[int] = None):
        self.name = name
        self.api_class = api_class
        self.config_fields = config_fields
        self.post_method = post_method
        self.max_chars = max_chars

class SocialPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self._initialize_session_state()
        self._setup_social_networks()

    def _initialize_session_state(self):
        if 'generated_posts' not in st.session_state:
            st.session_state.generated_posts = []
        if 'selected_platforms' not in st.session_state:
            st.session_state.selected_platforms = {}
        if 'manual_post_start' not in st.session_state:
            st.session_state.manual_post_start = ""
        if 'manual_post_end' not in st.session_state:
            st.session_state.manual_post_end = ""
        if 'platform_select_all' not in st.session_state:
            st.session_state.platform_select_all = {}
        if 'has_generated' not in st.session_state:
            st.session_state.has_generated = False

    def _setup_social_networks(self):
        self.social_networks = [
            SocialNetwork(
                name="twitter",
                api_class=TwitterAPI,
                config_fields={
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
                    }
                },
                post_method='create_thread',
                max_chars=280
            ),
            SocialNetwork(
                name="bluesky",
                api_class=BlueskyAPI,
                config_fields={
                    "bluesky_handle": {
                        "type": "text",
                        "label": "Bluesky Handle",
                        "default": ""
                    },
                    "bluesky_password": {
                        "type": "text",
                        "label": "Bluesky App Password",
                        "default": ""
                    }
                },
                post_method='create_thread',
                max_chars=280
            ),
            SocialNetwork(
                name="telegram",
                api_class=TelegramAPI,
                config_fields={
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
            ),
            SocialNetwork(
                name="ghost",
                api_class=GhostAPI,
                config_fields={
                    "ghost_url": {
                        "type": "text",
                        "label": "Ghost URL",
                        "default": ""
                    },
                    "ghost_api_key": {
                        "type": "text",
                        "label": "Ghost API Key",
                        "default": ""
                    }
                }
            )
        ]

        # Initialize platform_select_all for each network
        for network in self.social_networks:
            if network.name not in st.session_state.platform_select_all:
                st.session_state.platform_select_all[network.name] = False

    def get_config_fields(self):
        config_fields = {
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
            }
        }

        # Add config fields for each social network
        for network in self.social_networks:
            for field_name, field_config in network.config_fields.items():
                config_fields[field_name] = field_config

        return config_fields

    def get_tabs(self):
        return [{"name": t("social_tab"), "plugin": "social"}]

    def parse_posts(self, llm_response):
        posts = []
        for post in llm_response.split('TWEET:')[1:]:
            clean_post = post.strip().split('---')[0].strip()
            if clean_post:
                posts.append(clean_post)
        return posts

    def create_platform_columns(self):
        num_networks = len(self.social_networks)
        return st.columns(num_networks)

    def render_platform_checkboxes(self, cols, post_index: int, is_manual: bool = False):
        platforms = {}
        for col, network in zip(cols, self.social_networks):
            with col:
                if is_manual:
                    key = f"{network.name}_manual_{post_index}"
                else:
                    key = f"{network.name}_{post_index}"
                checked = st.checkbox(
                    t(f"social_{network.name}"),
                    key=key,
                    value=st.session_state.selected_platforms.get(post_index, {}).get(network.name, False)
                )
                platforms[network.name] = checked
        return platforms

    def render_select_all_buttons(self, cols):
        for col, network in zip(cols, self.social_networks):
            with col:
                select_all = st.checkbox(
                    t("social_select_all"),
                    key=f"{network.name}_select_all"
                )
                if select_all != st.session_state.platform_select_all[network.name]:
                    st.session_state.platform_select_all[network.name] = select_all
                    for i in range(len(st.session_state.generated_posts)):
                        if st.session_state.generated_posts[i].strip():
                            if i not in st.session_state.selected_platforms:
                                st.session_state.selected_platforms[i] = {}
                            st.session_state.selected_platforms[i][network.name] = select_all

    def post_content(self, config):
        with st.spinner(t("social_posting")):
            # Initialize API instances
            api_instances = {}
            for network in self.social_networks:
                api_config = {
                    key.replace(f"{network.name}_", ""): value
                    for key, value in config['social'].items()
                    if key.startswith(f"{network.name}_")
                }
                api_instances[network.name] = network.api_class(config)

            # Collect posts by platform
            platform_posts = {network.name: [] for network in self.social_networks}

            for i, post in enumerate(st.session_state.generated_posts):
                platforms = st.session_state.selected_platforms.get(i, {})
                for network in self.social_networks:
                    if platforms.get(network.name, False):
                        platform_posts[network.name].append(post)

            # Send posts to each platform
            for network in self.social_networks:
                posts = platform_posts[network.name]
                if posts:
                    api = api_instances[network.name]
                    if network.post_method == 'create_thread':
                        getattr(api, network.post_method)(posts)
                    else:
                        for post in posts:
                            if network.name == 'ghost':
                                api.post("Generated Post", post)
                            else:
                                api.post(post)

            st.success(t("social_success"))

    def validate_posts(self):
        problematic_posts = []
        for i, post in enumerate(st.session_state.generated_posts):
            platforms = st.session_state.selected_platforms.get(i, {})
            for network in self.social_networks:
                if platforms.get(network.name) and network.max_chars:
                    if len(post) > network.max_chars:
                        problematic_posts.append((i+1, network.name))
                        st.warning(f"{post} length: {len(post)}")

        if problematic_posts:
            error_msg = "The following posts exceed character limits:\n"
            for post_num, network in problematic_posts:
                error_msg += f"Post {post_num} exceeds {network} limit\n"
            st.error(error_msg)
        else:
            st.success("All selected posts respect character limits.")

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
        st.subheader(t("social_manual_start"))
        manual_post_start = st.text_area("", key="manual_post_start", height=100)
        cols = self.create_platform_columns()
        start_platforms = self.render_platform_checkboxes(cols, "start", True)

        # Handle manual start post
        if not st.session_state.has_generated:
            if manual_post_start:
                st.session_state.generated_posts = [manual_post_start]
                st.session_state.selected_platforms = {0: start_platforms}

        # Generate button
        if st.button(t("social_generate")) and transcript:
            st.session_state.has_generated = True
            with st.spinner(t("social_generating")):
                ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
                llm_response = ragllm_plugin.process_with_llm(
                    prompt,
                    config.get('llm', {}).get('llm_sys_prompt', ''),
                    transcript
                )

                # Reset and rebuild posts list
                st.session_state.generated_posts = []
                if manual_post_start:
                    st.session_state.generated_posts.append(manual_post_start)
                    st.session_state.selected_platforms[0] = start_platforms

                # Add generated posts
                st.session_state.generated_posts.extend(self.parse_posts(llm_response))

                # Add URL suffix if present
                if url:
                    url_suffix = config['social']['url_suffix_template'].format(url=url)
                    st.session_state.generated_posts.append(url_suffix)

        # Display posts
        if st.session_state.generated_posts:
            st.subheader(t("social_preview"))

            # Select All / Deselect All buttons
            if st.session_state.has_generated:
                cols = self.create_platform_columns()
                self.render_select_all_buttons(cols)

            # Display posts
            for i, post in enumerate(st.session_state.generated_posts):
                if st.session_state.has_generated and not (manual_post_start and i == 0):
                    post_label = "" if (not st.session_state.has_generated and i == 0) else f"Post {i+1}"
                    edited_post = st.text_area(post_label, post, key=f"post_{i}", height=100)
                    st.session_state.generated_posts[i] = edited_post

                    cols = self.create_platform_columns()
                    is_manual_start = i == 0 and manual_post_start
                    platforms = self.render_platform_checkboxes(cols, i)
                    st.session_state.selected_platforms[i] = platforms

        # Manual post at end
        st.subheader(t("social_manual_end"))
        manual_post_end = st.text_area("", key="manual_post_end", height=100)
        cols = self.create_platform_columns()
        end_platforms = self.render_platform_checkboxes(cols, "end", True)

        # Add manual end post if present
        if manual_post_end and st.session_state.generated_posts:
            st.session_state.generated_posts.append(manual_post_end)
            last_index = len(st.session_state.generated_posts) - 1
            st.session_state.selected_platforms[last_index] = end_platforms

        # Validate character count
        if st.button(t("social_validate")):
            self.validate_posts()

        # Debug button
        if st.button("Debug"):
            for i, post in enumerate(st.session_state.generated_posts):
                st.info(post)

        # Post button
        if st.button(t("social_post")):
            self.post_content(config)
