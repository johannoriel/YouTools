from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
import os, re
from datetime import datetime as date
import requests
import jwt
import markdown2
from lib.social_api import GhostAPI, LinkedinAPI, WordPressAPI, SubstackAPI
from widgets.random_article import RandomArticleWidget
from widgets.article_editor import ArticleEditorWidget

translations["en"].update({
    "ghost_tab": "Ghost Publisher",
    "ghost_header": "Publish to Ghost",
    "ghost_publish_button": "Publish to Ghost",
    "ghost_processing": "Publishing to Ghost...",
    "ghost_success": "Published successfully! Post ID: {result}",
    "ghost_error": "Publishing failed: {error}",
    "ghost_config_api_key": "Ghost API Key",
    "ghost_config_url": "Ghost Admin URL",
    "ghost_config_api_key_default": "your_api_key",
    "ghost_config_url_default": "https://your-site.com/ghost/api/admin/",
    "linkedin_tab": "LinkedIn Publisher",
    "linkedin_header": "Publish to LinkedIn",
    "linkedin_publish_button": "Publish to LinkedIn",
    "linkedin_processing": "Publishing to LinkedIn...",
    "linkedin_success": "Published successfully! Post ID: {result}",
    "linkedin_error": "Publishing failed: {error}",
    "wordpress_tab": "WordPress Publisher",
    "wordpress_header": "Publish to WordPress",
    "wordpress_publish_button": "Publish to WordPress",
    "wordpress_processing": "Publishing to WordPress...",
    "wordpress_success": "Published successfully! Post ID: {result}",
    "wordpress_error": "Publishing failed: {error}",
    "wordpress_get_token_button": "Get WordPress Access Token",
    "wordpress_auth_code_label": "Authorization Code",
    "wordpress_token_success": "Access token retrieved successfully!",
    "wordpress_token_error": "Failed to retrieve access token: {error}",
    "substack_tab": "Substack Publisher",
    "substack_header": "Publish to Substack",
    "substack_publish_button": "Publish to Substack",
    "substack_processing": "Publishing to Substack...",
    "substack_success": "Published successfully! Post ID: {result}",
    "substack_error": "Publishing failed: {error}",
    "substack_auth_status": "Substack Authentication Status",
    "substack_authenticated": "Substack is authenticated. You can now publish posts.",
    "substack_not_authenticated": "Substack is not authenticated. Please check your credentials.",
    "substack_reauth_button": "Re-authenticate Substack",
    "substack_reauth_processing": "Re-authenticating Substack...",
    "substack_drafts_label": "Select Draft Post",
    "substack_publish_draft_button": "Publish Selected Draft",
    "substack_refresh_drafts_button": "Refresh Drafts List",
    "substack_no_drafts": "No draft posts available."
})

translations["fr"].update({
    "ghost_tab": "Publicateur Ghost",
    "ghost_header": "Publier sur Ghost",
    "ghost_publish_button": "Publier sur Ghost",
    "ghost_processing": "Publication sur Ghost...",
    "ghost_success": "Publié avec succès ! ID du post : {result}",
    "ghost_error": "Échec de la publication : {error}",
    "ghost_config_api_key": "Clé API Ghost",
    "ghost_config_url": "URL Admin Ghost",
    "ghost_config_api_key_default": "votre_clé_api",
    "ghost_config_url_default": "https://votre-site.com/ghost/api/admin/",
    "linkedin_tab": "Publicateur LinkedIn",
    "linkedin_header": "Publier sur LinkedIn",
    "linkedin_publish_button": "Publier sur LinkedIn",
    "linkedin_processing": "Publication sur LinkedIn...",
    "linkedin_success": "Publié avec succès ! ID du post : {result}",
    "linkedin_error": "Échec de la publication : {error}",
    "wordpress_tab": "Publicateur WordPress",
    "wordpress_header": "Publier sur WordPress",
    "wordpress_publish_button": "Publier sur WordPress",
    "wordpress_processing": "Publication sur WordPress...",
    "wordpress_success": "Publié avec succès ! ID du post : {result}",
    "wordpress_error": "Échec de la publication : {error}",
    "wordpress_get_token_button": "Obtenir le jeton d'accès WordPress",
    "wordpress_auth_code_label": "Code d'autorisation",
    "wordpress_token_success": "Jeton d'accès récupéré avec succès !",
    "wordpress_token_error": "Échec de la récupération du jeton : {error}",
    "substack_tab": "Publicateur Substack",
    "substack_header": "Publier sur Substack",
    "substack_publish_button": "Publier sur Substack",
    "substack_processing": "Publication sur Substack...",
    "substack_success": "Publié avec succès ! ID du post : {result}",
    "substack_error": "Échec de la publication : {error}",
    "substack_auth_status": "État de l'authentification Substack",
    "substack_authenticated": "Substack est authentifié. Vous pouvez maintenant publier des posts.",
    "substack_not_authenticated": "Substack n'est pas authentifié. Veuillez vérifier vos identifiants.",
    "substack_reauth_button": "Ré-authentifier Substack",
    "substack_reauth_processing": "Ré-authentification sur Substack...",
    "substack_drafts_label": "Sélectionner un brouillon",
    "substack_publish_draft_button": "Publier le brouillon sélectionné",
    "substack_refresh_drafts_button": "Rafraîchir la liste des brouillons",
    "substack_no_drafts": "Aucun brouillon disponible."
})

class PostarticlePlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.ghost_api = GhostAPI(plugin_manager.config)
        self.linkedin_api = LinkedinAPI(plugin_manager.config)
        self.wordpress_api = WordPressAPI(plugin_manager.config)
        self.substack_api = SubstackAPI(plugin_manager.config)
        self.work_dir = self.work_dir()
        # Initialize Substack authentication status in session state
        if "substack_authenticated" not in st.session_state:
            # Use the first publication URL as default for cookie validation
            publication_urls = self.plugin_manager.config['common'].get('substack_publication_url', '')
            if isinstance(publication_urls, str):
                publication_urls = [url.strip() for url in publication_urls.split(';') if url.strip()]
            default_publication_url = publication_urls[0] if publication_urls else None
            if default_publication_url:
                st.session_state["substack_authenticated"] = self.substack_api._is_cookie_valid(default_publication_url)
            else:
                st.session_state["substack_authenticated"] = False
        if "substack_drafts" not in st.session_state:
            st.session_state["substack_drafts"] = []

    def get_config_fields(self):
        return {
            "randompost_prompt": {
                "type": "textarea",
                "label": t("randomarticle_prompt"),
                "default": """Génère un article de blog en Markdown basé sur le produit suivant :
            Titre : {title}
            Contenu : {content}
            Mots-clés : {keywords}

            L'article doit être structuré avec une introduction, 3 sections principales et une conclusion. Utilise un ton professionnel et intègre les mots-clés naturellement."""
            },
        }

    def get_tabs(self):
        return [
            {"name": t("ghost_tab"), "plugin": "ghostplugin"},
            {"name": t("randomarticle_tab"), "plugin": "randomarticle"},
            {"name": t("linkedin_tab"), "plugin": "linkedinplugin"},
            {"name": t("wordpress_tab"), "plugin": "wordpressplugin"},
            {"name": t("substack_tab"), "plugin": "substackplugin"}
        ]

    def _markdown_to_text(self, markdown_content: str) -> str:
        """
        Converts markdown content to plain text, stripping formatting and headers.
        :param markdown_content: Markdown string.
        :return: Plain text string.
        """
        # Convert markdown to HTML using markdown2
        html = markdown2.markdown(markdown_content)
        # Strip HTML tags and clean up
        text = re.sub(r'<[^>]+>', '', html)  # Remove HTML tags
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove extra newlines
        text = text.strip()  # Remove leading/trailing whitespace
        return text

    def run(self, config):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([t("randomarticle_tab"), t("ghost_tab"), t("linkedin_tab"), t("wordpress_tab"), t("substack_tab")])

        # Ghost Publisher Tab
        with tab1:
            from widgets.random_article import RandomArticleWidget
            RandomArticleWidget("randomarticle", "randomarticle", self.plugin_manager).display(config)

        # Random Article Tab
        with tab2:
            st.header(t("ghost_header"))
            editor = ArticleEditorWidget("ghosteditor", "ghost", self.plugin_manager)
            result = editor.display()

            if st.button(t("ghost_publish_button"), key="ghost_publish"):
                with st.spinner(t("ghost_processing")):
                    try:
                        response = self.ghost_api.post(
                            result["title"],
                            result["html_content"],
                            result["publish_immediately"],
                            feature_image=result["image_path"]
                        )
                        if response:
                            post_id = response.get('posts', [{}])[0].get('id', 'N/A')
                            st.success(t("ghost_success").format(result=post_id))
                            for key in ["ghost_title", "ghost_content", "ghost_image_path", "ghost_url"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                        else:
                            st.error(t("ghost_error").format(error="Unknown error"))
                    except Exception as e:
                        st.error(t("ghost_error").format(error=str(e)))

        # LinkedIn Publisher Tab
        with tab3:
            st.header(t("linkedin_header"))
            editor = ArticleEditorWidget("linkedineditor", "linkedin", self.plugin_manager)
            result = editor.display()

            source_url = st.session_state.get('linkedin_url', '')

            if st.button(t("linkedin_publish_button"), key="linkedin_publish"):
                with st.spinner(t("linkedin_processing")):
                    try:
                        plain_content = self._markdown_to_text(result["markdown_content"])
                        response = self.linkedin_api.post_article(
                            result["title"],
                            plain_content,
                            source_url=source_url,
                            feature_image=result["image_path"]
                        )
                        if response:
                            post_id = response.get('id', 'N/A')
                            st.success(t("linkedin_success").format(result=post_id))
                            for key in ["linkedin_title", "linkedin_content", "linkedin_image_path", "linkedin_url"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                        else:
                            st.error(t("linkedin_error").format(error="Unknown error"))
                    except Exception as e:
                        st.error(t("linkedin_error").format(error=str(e)))

        # WordPress Publisher Tab
        with tab4:
            st.header(t("wordpress_header"))

            # Check for authorization code in query parameters
            query_params = st.query_params
            auth_code = query_params.get("code")
            state = query_params.get("state")

            # Handle OAuth redirect
            if auth_code and state:
                if st.button("Retrieve Token"):
                    with st.spinner("Retrieving WordPress access token..."):
                        try:
                            token = self.wordpress_api._get_access_token(code=auth_code, state=state)
                            if token:
                                self.plugin_manager.config['common']['wordpress_access_token'] = token
                                self.plugin_manager.save_config(config)
                                st.success(t("wordpress_token_success"))
                            else:
                                st.error(t("wordpress_token_error").format(error="Failed to retrieve token"))
                        except Exception as e:
                            st.error(t("wordpress_token_error").format(error=str(e)))

            # Token retrieval section
            st.subheader("WordPress Authentication")
            if not self.wordpress_api.access_token or st.button("Refresh Token"):
                with st.spinner("Generating WordPress authorization URL..."):
                    try:
                        auth_url = self.wordpress_api._get_access_token()
                        if auth_url:
                            st.markdown(f"[Click here to authorize WordPress]({auth_url})")
                        else:
                            st.error(t("wordpress_token_error").format(error="Failed to generate authorization URL"))
                    except Exception as e:
                        st.error(t("wordpress_token_error").format(error=str(e)))
                        st.write("Go to https://developer.wordpress.com/apps and https://developer.wordpress.com/docs/oauth2/ to get ids")
            else:
                st.success("WordPress is authenticated. You can now publish posts.")

            editor = ArticleEditorWidget("wordpresseditor", "wordpress", self.plugin_manager)
            result = editor.display()

            if st.button(t("wordpress_publish_button"), key="wordpress_publish"):
                with st.spinner(t("wordpress_processing")):
                    try:
                        response = self.wordpress_api.post(
                            result["title"],
                            result["html_content"],
                            result["publish_immediately"],
                            feature_image=result["image_path"]
                        )
                        if response:
                            post_id = response.get('id', 'N/A')
                            st.success(t("wordpress_success").format(result=post_id))
                            for key in ["wordpress_title", "wordpress_content", "wordpress_image_path", "wordpress_url"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                        else:
                            st.error(t("wordpress_error").format(error="Unknown error"))
                    except Exception as e:
                        st.error(t("wordpress_error").format(error=str(e)))

        # Substack Publisher Tab
        with tab5:
            st.header(t("substack_header"))

            # Récupérer la liste des URLs de publication depuis la configuration
            publication_urls = self.plugin_manager.config['common'].get('substack_publication_url', '')
            if isinstance(publication_urls, str):
                publication_urls = [url.strip() for url in publication_urls.split(';') if url.strip()]
            if not publication_urls:
                st.error("No Substack publication URLs configured.")
                return

            # Sélecteur pour choisir la publication
            st.subheader("Select Publication")
            selected_publication = st.selectbox(
                "Choose a Substack publication",
                options=publication_urls,
                index=0,  # Par défaut, sélectionner la première URL
                key="substack_publication_select"
            )

            # Authentication status
            st.subheader(t("substack_auth_status"))
            if st.session_state.get("substack_authenticated", False):
                st.success(t("substack_authenticated"))
            else:
                st.error(t("substack_not_authenticated"))

            # Re-authentication button
            if st.button(t("substack_reauth_button"), key="substack_reauth"):
                with st.spinner(t("substack_reauth_processing")):
                    try:
                        #self.substack_api._renew_cookie(self.substack_api.email, self.substack_api.password, selected_publication)
                        self.substack_api._initialize_api(publication_url=selected_publication, force=True)
                        st.session_state["substack_authenticated"] = self.substack_api._is_cookie_valid(selected_publication)
                        if st.session_state["substack_authenticated"]:
                            st.success("Substack re-authenticated successfully!")
                        else:
                            st.error("Substack re-authentication failed.")
                    except Exception as e:
                        st.error(f"Substack re-authentication error: {str(e)}")
                        st.session_state["substack_authenticated"] = False

            # Draft posts section
            st.subheader(t("substack_drafts_label"))
            if st.button(t("substack_refresh_drafts_button"), key="substack_refresh_drafts"):
                with st.spinner("Refreshing Substack drafts..."):
                    try:
                        st.session_state["substack_drafts"] = self.substack_api.list_drafts(publication_url=selected_publication)
                    except Exception as e:
                        st.error(f"Failed to refresh drafts: {str(e)}")

            drafts = st.session_state.get("substack_drafts", [])
            draft_options = {f"{d['title']} (ID: {d['id']})": d['id'] for d in drafts} if drafts else {}
            if draft_options:
                selected_draft = st.selectbox(t("substack_drafts_label"), options=[""] + list(draft_options.keys()), key="substack_draft_select")
                if selected_draft and st.button(t("substack_publish_draft_button"), key="substack_publish_draft"):
                    with st.spinner(t("substack_processing")):
                        try:
                            draft_id = draft_options[selected_draft]
                            response = self.substack_api.publish_draft(draft_id, publication_url=selected_publication)
                            if response:
                                st.success(t("substack_success").format(result=response.get('id', 'N/A')))
                                st.session_state["substack_drafts"] = self.substack_api.list_drafts(publication_url=selected_publication)
                            else:
                                st.error(t("substack_error").format(error="Unknown error"))
                        except Exception as e:
                            st.error(t("substack_error").format(error=str(e)))
            else:
                st.info(t("substack_no_drafts"))

            # New post section
            editor = ArticleEditorWidget("substackeditor", "substack", self.plugin_manager)
            result = editor.display()

            if st.button(t("substack_publish_button"), key="substack_publish"):
                with st.spinner(t("substack_processing")):
                    try:
                        response = self.substack_api.post(
                            result["title"],
                            result["markdown_content"],
                            result["publish_immediately"],
                            feature_image=result["image_path"],
                            publication_url=selected_publication
                        )
                        if response:
                            post_id = response.get('id', 'N/A')
                            st.success(t("substack_success").format(result=post_id))
                            st.session_state["substack_drafts"] = self.substack_api.list_drafts(publication_url=selected_publication)
                            for key in ["substack_title", "substack_content", "substack_image_path", "substack_url"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                        else:
                            st.error(t("substack_error").format(error="Unknown error"))
                    except Exception as e:
                        st.error(t("substack_error").format(error=str(e)))
