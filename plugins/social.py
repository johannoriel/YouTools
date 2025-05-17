from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import os
from typing import List, Dict, Any, Optional
from lib.social_api import TwitterAPI, BlueskyAPI, TelegramAPI, GhostAPI
from lib.youtube_api import YoutubeAPI

# Traductions spécifiques au plugin
translations["en"].update({
    "social_tab": "Social Networks",
    "social_preview": "Preview and Edit Posts",
    "social_post": "Post Selected Content",
    "social_posting": "Posting content...",
    "social_success": "Content posted successfully!",
    "social_error": "Error posting content: ",
    "social_twitter": "Twitter",
    "social_bluesky": "Bluesky",
    "social_telegram": "Telegram",
    "social_ghost": "Ghost",
    "social_youtube": "YouTube",
    "social_select_all": "Select All",
    "social_validate": "Validate Character Count",
    "select_meme_for_first_post": "Select Meme for First Post (Optional)",
})

translations["fr"].update({
    "social_tab": "Réseaux Sociaux",
    "social_preview": "Prévisualiser et Éditer",
    "social_post": "Poster la sélection",
    "social_posting": "Publication en cours...",
    "social_success": "Contenu publié avec succès !",
    "social_error": "Erreur lors de la publication : ",
    "social_twitter": "Twitter",
    "social_bluesky": "Bluesky",
    "social_telegram": "Telegram",
    "social_ghost": "Ghost",
    "social_youtube": "YouTube",
    "social_select_all": "Tout sélectionner",
    "social_validate": "Vérifier le nombre de caractères",
    "select_meme_for_first_post": "Sélectionner un mème pour le premier post (facultatif)",
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
        if 'platform_select_all' not in st.session_state:
            st.session_state.platform_select_all = {}
        if 'selected_meme' not in st.session_state:
            st.session_state.selected_meme = None

    def _get_image_files(self, work_dir):
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif')
        return [f for f in os.listdir(work_dir)
                if os.path.isfile(os.path.join(work_dir, f))
                and f.lower().endswith(image_extensions)]

    def _setup_social_networks(self):
        self.social_networks = [
            SocialNetwork(
                name="twitter",
                api_class=TwitterAPI,
                config_fields={},
                post_method='create_thread',
                max_chars=280
            ),
            SocialNetwork(
                name="bluesky",
                api_class=BlueskyAPI,
                config_fields={},
                post_method='create_thread',
                max_chars=280
            ),
            SocialNetwork(
                name="telegram",
                api_class=TelegramAPI,
                config_fields={}
            ),
            SocialNetwork(
                name="ghost",
                api_class=GhostAPI,
                config_fields={}
            ),
            SocialNetwork(
                name="youtube",
                api_class=YoutubeAPI,
                config_fields={},
                post_method='post',
                max_chars=10000
            ),
        ]

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

        for network in self.social_networks:
            for field_name, field_config in network.config_fields.items():
                config_fields[field_name] = field_config

        return config_fields

    def get_tabs(self):
        return [
            {"name": "Generate Thread", "plugin": "social"},
            {"name": "Edit Thread", "plugin": "social"},
            {"name": "Post Thread", "plugin": "social"}
        ]

    def create_platform_columns(self):
        num_networks = len(self.social_networks)
        return st.columns(num_networks)

    def render_platform_checkboxes(self, cols, post_index: int):
        platforms = {}
        for col, network in zip(cols, self.social_networks):  # Correction de la syntaxe
            with col:
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
            api_instances = {}
            work_dir = config['common']['work_directory']

            for network in self.social_networks:
                api_config = {
                    key.replace(f"{network.name}_", ""): value
                    for key, value in config['social'].items()
                    if key.startswith(f"{network.name}_")
                }
                api_instances[network.name] = network.api_class(config)

            platform_posts = {network.name: [] for network in self.social_networks}
            meme_path = None
            if st.session_state.selected_meme:
                meme_path = os.path.join(work_dir, st.session_state.selected_meme)

            for i, post in enumerate(st.session_state.generated_posts):
                platforms = st.session_state.selected_platforms.get(i, {})
                for network in self.social_networks:
                    if platforms.get(network.name, False):
                        if i == 0 and meme_path and network.name in ['twitter', 'bluesky', 'telegram']:  # Correction de la syntaxe
                            platform_posts[network.name].append((post, meme_path))
                        else:
                            platform_posts[network.name].append(post)

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
        tab1, tab2, tab3 = st.tabs(["Generate Thread", "Edit Thread", "Post Thread"])

        with tab1:
            from widgets.recentvideos import RecentVideosWidget
            widget = RecentVideosWidget("treahdgen", "rvw", self.plugin_manager)
            widget.display_simple(config)
            from widgets.thread_generator import ThreadGeneratorWidget
            ThreadGeneratorWidget("threadgen", "tgw", self.plugin_manager).display(config)

        with tab2:
            from widgets.thread_editor import ThreadEditorWidget
            ThreadEditorWidget("threadedit", "tew", self.plugin_manager).display(config)

        with tab3:
            st.subheader(t("social_preview"))
            work_dir = config['common']['work_directory']
            thread_path = os.path.join(work_dir, "thread_edited.txt")
            if not os.path.exists(thread_path):
                thread_path = os.path.join(work_dir, "thread.txt")

            if os.path.exists(thread_path):
                with open(thread_path, 'r') as f:
                    st.session_state.generated_posts = [post.strip() for post in f.read().split('---') if post.strip()]

                cols = self.create_platform_columns()
                self.render_select_all_buttons(cols)

                for i, post in enumerate(st.session_state.generated_posts):
                    st.text_area(f"Post {i+1}", post, key=f"post_{i}", height=100, disabled=True)
                    cols = self.create_platform_columns()
                    platforms = self.render_platform_checkboxes(cols, i)
                    st.session_state.selected_platforms[i] = platforms

                st.subheader(t("select_meme_for_first_post"))
                image_files = self._get_image_files(work_dir)
                meme_options = ["None"] + image_files
                default_index = meme_options.index(st.session_state.selected_meme) if st.session_state.selected_meme in meme_options else 0
                selected_meme = st.selectbox("Choose an image file", options=meme_options, index=default_index)
                if selected_meme != st.session_state.selected_meme:
                    st.session_state.selected_meme = selected_meme if selected_meme != "None" else None
                if st.session_state.selected_meme and selected_meme != "None":
                    st.image(os.path.join(work_dir, st.session_state.selected_meme), caption="Selected Meme")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(t("social_validate")):
                        self.validate_posts()
                with col2:
                    if st.button(t("social_post")):
                        self.post_content(config)
