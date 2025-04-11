from global_vars import translations, t
from app import Plugin
import streamlit as st
import os
from plugins.ragllm import RagllmPlugin
from typing import List, Dict, Any, Optional
# Utilisation de l'API YouTube depuis social_api.py
from youtube_api import YoutubeAPI
from datetime import datetime
import pytz
from youtube_db import *
from plugins.automarket import AutomarketPlugin

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
    "promoteyoutube_select_videos": "Select Videos to Analyze",
    "promoteyoutube_fetch_comments": "Fetch Comments",
    "promoteyoutube_fetching_comments": "Fetching comments for selected videos...",
    "promoteyoutube_select_videos": "Select Videos to Analyze",
    "promoteyoutube_adjust_comments": "Number of comments per video",
    "promoteyoutube_select_all_videos": "Select All Videos",
    "promoteyoutube_deselect_all_videos": "Deselect All Videos",
    "promoteyoutube_select_all_comments": "Select All Comments",
    "promoteyoutube_deselect_all_comments": "Deselect All Comments",
    "promoteyoutube_select_all_responses": "Select All Responses",
    "promoteyoutube_deselect_all_responses": "Deselect All Responses",
    "promoteyoutube_subscribers": "Subscribers",
    "promoteyoutube_views": "Views",
    "promoteyoutube_comments_count": "Comments",
    "promoteyoutube_watch_video": "Watch video",
    "promoteyoutube_select_video": "Select",
    "promoteyoutube_video_order": "Video order",
    "promoteyoutube_comment_order": "Comment order",
    "promoteyoutube_comment_context": "Comment by {} on video {} from channel {}:",
    "promoteyoutube_view_context": "View comment in context",
    "promoteyoutube_response_to_comment": "Response to Comment {}",
    "promoteyoutube_edit_response": "Edit Response {}",
    "promoteyoutube_max_videos_label": "Number of videos to search",
    "promoteyoutube_keywords_warning": "Please enter keywords to search for videos.",
    "promoteyoutube_getting_comments": "Fetching comments...",
    "promoteyoutube_char_limit_warning": "⚠️ This response exceeds 500 characters ({} characters). Please shorten it.",
    "promoteyoutube_regenerate_errors": "Regenerate Failed Responses",
    "promoteyoutube_regenerating": "Regenerating failed responses...",
    "promoteyoutube_no_errors": "No failed responses to regenerate.",
    "promoteyoutube_progress": "Processing comment {} of {}",
    "promoteyoutube_error_count": "{} responses contain errors",
    "promoteyoutube_sort_by": "Sort results by",
    "promoteyoutube_sort_api": "API (default)",
    "promoteyoutube_sort_relevance": "Relevance score",
    "promoteyoutube_days_old": "days old",
    "promoteyoutube_relevance_score": "Relevance score",
    "promoteyoutube_language": "Language",
    "promoteyoutube_filter_language": "Filter by language",
    "promoteyoutube_show_all_languages": "All languages",
    "promoteyoutube_language": "Langue",
    "promoteyoutube_filter_language": "Filtrer par langue",
    "promoteyoutube_show_all_languages": "Toutes les langues",
    "promoteyoutube_export_channels": "Export Selected Channels List",
    "promoteyoutube_export_videos": "Export Selected Videos List",
    "promoteyoutube_export_comments": "Export Selected Comments List",
    "promoteyoutube_export_success": "Exported successfully to {}",
    "promoteyoutube_export_error": "Error during export: {}",
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
    "promoteyoutube_select_videos": "Sélectionner les vidéos à analyser",
    "promoteyoutube_fetch_comments": "Récupérer les commentaires",
    "promoteyoutube_fetching_comments": "Récupération des commentaires pour les vidéos sélectionnées...",
    "promoteyoutube_select_videos": "Sélectionner les vidéos à analyser",
    "promoteyoutube_adjust_comments": "Nombre de commentaires par vidéo",
    "promoteyoutube_select_all_videos": "Sélectionner toutes les vidéos",
    "promoteyoutube_deselect_all_videos": "Désélectionner toutes les vidéos",
    "promoteyoutube_select_all_comments": "Sélectionner tous les commentaires",
    "promoteyoutube_deselect_all_comments": "Désélectionner tous les commentaires",
    "promoteyoutube_select_all_responses": "Sélectionner toutes les réponses",
    "promoteyoutube_deselect_all_responses": "Désélectionner toutes les réponses",
    "promoteyoutube_subscribers": "Abonnés",
    "promoteyoutube_views": "Vues",
    "promoteyoutube_comments_count": "Commentaires",
    "promoteyoutube_watch_video": "Voir la vidéo",
    "promoteyoutube_select_video": "Sélectionner",
    "promoteyoutube_video_order": "Ordre des vidéos",
    "promoteyoutube_comment_order": "Ordre des commentaires",
    "promoteyoutube_comment_context": "Commentaire de {} sur la vidéo {} de la chaîne {} :",
    "promoteyoutube_view_context": "Voir le commentaire en contexte",
    "promoteyoutube_response_to_comment": "Réponse au commentaire {}",
    "promoteyoutube_edit_response": "Modifier la réponse {}",
    "promoteyoutube_max_videos_label": "Nombre de vidéos à rechercher",
    "promoteyoutube_keywords_warning": "Veuillez entrer des mots-clés pour la recherche.",
    "promoteyoutube_getting_comments": "Récupération des commentaires...",
    "promoteyoutube_char_limit_warning": "⚠️ Cette réponse dépasse 500 caractères ({} caractères). Veuillez la raccourcir.",
    "promoteyoutube_regenerate_errors": "Regénérer les réponses en erreur",
    "promoteyoutube_regenerating": "Regénération des réponses en erreur...",
    "promoteyoutube_no_errors": "Aucune réponse en erreur à regénérer.",
    "promoteyoutube_progress": "Traitement du commentaire {} sur {}",
    "promoteyoutube_error_count": "{} réponses contiennent des erreurs",
    "promoteyoutube_sort_by": "Trier les résultats par",
    "promoteyoutube_sort_api": "API (défaut)",
    "promoteyoutube_sort_relevance": "Score de pertinence",
    "promoteyoutube_days_old": "jours",
    "promoteyoutube_relevance_score": "Score de pertinence",
    "promoteyoutube_language": "Langue",
    "promoteyoutube_filter_language": "Filtrer par langue",
    "promoteyoutube_show_all_languages": "Toutes les langues",
    "promoteyoutube_export_channels": "Exporter la liste des chaînes sélectionnées",
    "promoteyoutube_export_videos": "Exporter la liste des vidéos sélectionnées",
    "promoteyoutube_export_comments": "Exporter la liste des commentaires sélectionnés",
    "promoteyoutube_export_success": "Exporté avec succès vers {}",
    "promoteyoutube_export_error": "Erreur lors de l'export : {}",
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
        if 'videos' not in st.session_state:
            st.session_state.videos = []
        if 'selected_videos' not in st.session_state:
            st.session_state.selected_videos = {}
        if 'show_comments_section' not in st.session_state:
            st.session_state.show_comments_section = False

    def search_videos(self, keywords: str, max_videos: int, video_order: str):
        youtube_api = YoutubeAPI(self.plugin_manager.config)
        videos = youtube_api.search_videos(
            keywords,
            max_videos,
            order=video_order,
            language=st.session_state.lang
        )
        return videos

    def get_config_fields(self):
        return {
            "max_videos": {
                "type": "number",
                "label": "Maximum Number of Videos to Fetch",
                "default": 10
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

    def fetch_comments(self, target_videos, max_comments, comment_order):
        """Récupère les commentaires pour les vidéos cibles."""
        youtube_api = YoutubeAPI(self.plugin_manager.config)
        comments = []
        for video in target_videos:
            video_comments = youtube_api.get_comments(
                video['video_id'], max_comments, order=comment_order)
            for comment in video_comments:
                comment['video_title'] = video['title']
                comment['channel_title'] = video['channel_title']
                comment['video_id'] = video['video_id']
                comment['channel_id'] = video.get('channel_id', 'unknown')
            comments.extend(video_comments)
        return comments

    def generate_responses(self, config, selected_comments, transcript, url, prefix="promo_", keyword=""):
        """Génère les réponses pour les commentaires sélectionnés."""
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        responses = []
        total_comments = len(selected_comments)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        if keyword == "" or keyword is None:
            raise ValueError("Keywords cannot be empty or None")

        for idx, comment_index in enumerate(selected_comments):
            progress = (idx + 1) / total_comments
            progress_bar.progress(progress)
            progress_text.text(
                t("promoteyoutube_progress").format(idx + 1, total_comments))

            comment = st.session_state[f"{prefix}comments"][comment_index]
            comment_with_context = t("promoteyoutube_comment_context").format(
                comment['author'], comment['video_title'], comment['channel_title']
            ) + f"\n{comment['text']}"

            prompt = config['promoteyoutube']['response_prompt'].format(
                url=url, transcript=transcript)
            try:
                llm_response = ragllm_plugin.process_with_llm(
                    prompt,
                    config.get('llm', {}).get('llm_sys_prompt', ''),
                    comment_with_context
                )
                clean_response = remove_quotes(llm_response.strip())
            except Exception as e:
                clean_response = f"Error: {str(e)}"
            responses.append({
                'comment_id': comment['id'],
                'response': clean_response,
                'response_text': clean_response,
                'target_video_id': comment['video_id'],
                'comment_index': comment_index,
                'channel_id': comment['channel_id'],
                'keyword': keyword,
                'comment_text': comment['text'],
            })

        progress_bar.empty()
        progress_text.empty()
        return responses

    def regenerate_error_responses(self, config, transcript, url):
        if not st.session_state.generated_responses:
            return

        error_indices = [i for i, response in enumerate(st.session_state.generated_responses)
                         if "litellm.APIError" in response['response']]

        if not error_indices:
            st.warning(t("promoteyoutube_no_errors"))
            return

        # Créer une barre de progression
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_errors = len(error_indices)

        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)

        for idx, i in enumerate(error_indices):
            # Mise à jour de la progression
            progress = (idx + 1) / total_errors
            progress_bar.progress(progress)
            progress_text.text(
                t("promoteyoutube_progress").format(idx + 1, total_errors))

            response = st.session_state.generated_responses[i]
            comment = st.session_state.comments[response['comment_index']]
            comment_with_context = t("promoteyoutube_comment_context").format(
                comment['author'],
                comment['video_title'],
                comment['channel_title']
            ) + f"\n{comment['text']}"

            prompt = config['promoteyoutube']['response_prompt'].format(
                url=url,
                transcript=transcript
            )

            try:
                llm_response = ragllm_plugin.process_with_llm(
                    prompt,
                    config.get('llm', {}).get('llm_sys_prompt', ''),
                    comment_with_context
                )
                clean_response = remove_quotes(llm_response.strip())
                st.session_state.generated_responses[i]['response'] = clean_response
            except Exception as e:
                continue

        # Nettoyer la barre de progression à la fin
        progress_bar.empty()
        progress_text.empty()

    def post_responses(self, config, selected_responses, campaign_id):
        """Poste les réponses et met à jour la base."""
        youtube_api = YoutubeAPI(self.plugin_manager.config)
        for response in selected_responses:
            comment_id = response['comment_id']
            response_text = response['response']
            try:
                youtube_api.post_comment_reply(comment_id, response_text)
                update_campaign_response_status(
                    comment_id, "posted", campaign_id)
            except Exception as e:
                update_campaign_response_status(
                    comment_id, f"error: {str(e)}", campaign_id)

    def fetch_comments_for_selected_videos(self, selected_video_indices, max_comments_per_video, comment_order):
        youtube_api = YoutubeAPI(self.plugin_manager.config)
        comments = []

        for idx in selected_video_indices:
            video = st.session_state.videos[idx]
            # st.info(video)
            video_comments = youtube_api.get_comments(
                video['video_id'], max_comments_per_video, order=comment_order)
            for comment in video_comments:
                comment['video_title'] = video['title']
                comment['channel_title'] = video['channel_title']
            comments.extend(video_comments)

        return comments

    def _sort_videos(self, videos, sort_by):
        """
        Trie la liste des vidéos selon le critère choisi
        """
        if sort_by == t("promoteyoutube_sort_relevance"):
            return sorted(videos, key=lambda x: x['relevance_score'], reverse=True)
        return videos

    def display_and_select_videos(self, config, target_videos, prefix="promo_"):
        youtube_api = YoutubeAPI(self.plugin_manager.config)

        # Initialiser les variables de session
        if f"{prefix}videos" not in st.session_state:
            st.session_state[f"{prefix}videos"] = target_videos
            st.session_state[f"{prefix}original_order"] = target_videos.copy()
            st.session_state[f"{prefix}selected_videos"] = {
                i: False for i in range(len(target_videos))}
        if f"{prefix}sort_by" not in st.session_state:
            st.session_state[f"{prefix}sort_by"] = t("promoteyoutube_sort_api")

        st.subheader(t("promoteyoutube_select_videos"))

        # Filtre de langue
        all_languages = list(set(
            video['language'] for video in st.session_state[f"{prefix}videos"] if video['language'] != 'unknown'))
        selected_language = st.selectbox(
            t("promoteyoutube_filter_language"),
            [t("promoteyoutube_show_all_languages")] + all_languages,
            index=0,
            key=f"{prefix}filter_language"
        )

        filtered_videos = st.session_state[f"{prefix}videos"]
        filtered_indices = list(
            range(len(st.session_state[f"{prefix}videos"])))
        if selected_language != t("promoteyoutube_show_all_languages"):
            filtered_indices = [i for i, v in enumerate(
                st.session_state[f"{prefix}videos"]) if v['language'] == selected_language]
            filtered_videos = [
                st.session_state[f"{prefix}videos"][i] for i in filtered_indices]

        # Tri des vidéos
        sort_by = st.selectbox(
            t("promoteyoutube_sort_by"),
            options=[t("promoteyoutube_sort_api"), t(
                "promoteyoutube_sort_relevance")],
            index=0 if st.session_state[f"{prefix}sort_by"] == t(
                "promoteyoutube_sort_api") else 1,
            key=f"{prefix}sort_by",
            on_change=lambda: setattr(
                st.session_state, f"{prefix}sort_by", st.session_state[f"{prefix}sort_by"])
        )
        if sort_by != st.session_state[f"{prefix}sort_by"]:
            st.session_state[f"{prefix}sort_by"] = sort_by
            st.session_state[f"{prefix}videos"] = self._sort_videos(
                st.session_state[f"{prefix}original_order"].copy(), sort_by)

        # Boutons Select All/Deselect All
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("promoteyoutube_select_all_videos"), key=f"{prefix}select_all_videos"):
                for i in filtered_indices:
                    st.session_state[f"{prefix}selected_videos"][i] = True
        with col2:
            if st.button(t("promoteyoutube_deselect_all_videos"), key=f"{prefix}deselect_all_videos"):
                for i in filtered_indices:
                    st.session_state[f"{prefix}selected_videos"][i] = False

        # Affichage des vidéos
        for display_index, original_index in enumerate(filtered_indices):
            video = st.session_state[f"{prefix}videos"][original_index]
            published_at = video.get('published_at', '')
            if published_at and 'T' in published_at and 'Z' in published_at:
                published_display = published_at.split('T')[0]
                # Parse the string back to a datetime object for the days_ago calculation
                published_at_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
            else:
                published_display = str(published_at)[:10] if published_at else "--"
                # Fallback: assume a default date or handle as needed
                published_at_dt = datetime.strptime("1970-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ") if not published_at else datetime.strptime(published_at[:10] + "T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ")

            # Now calculate days_ago with the datetime object
            days_ago = (datetime.now(pytz.UTC) - published_at_dt.replace(tzinfo=pytz.UTC)).days

            st.markdown(f"[**{video['title']}**]({video['url']})")
            st.markdown(
                f"Chaîne : **[{video['channel_title']}](https://www.youtube.com/channel/{video['channel_id']})**")

            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            col1.markdown(
                f"{t('promoteyoutube_subscribers')} : **{youtube_api.format_count(video['subscriber_count'])}**")
            col2.markdown(
                f"{t('promoteyoutube_views')} : **{youtube_api.format_count(video['view_count'])}**")
            col3.markdown(
                f"{t('promoteyoutube_comments_count')} : **{video['comment_count']}**")
            col4.markdown(f"**{days_ago}** jours")
            col5.markdown(f"Score : **{video['relevance_score']}**/100")
            col6.markdown(
                f"{t('promoteyoutube_language')} : **{video['language']}**")
            st.session_state[f"{prefix}selected_videos"][original_index] = col7.checkbox(
                t("promoteyoutube_select_video"),
                key=f"{prefix}video_{original_index}",
                value=st.session_state[f"{prefix}selected_videos"].get(
                    original_index, False)
            )

        return [i for i, selected in st.session_state[f"{prefix}selected_videos"].items() if selected]

    def select_and_process_comments(self, config, selected_video_indices, campaign_video, max_comments, prefix="promo_", keywords=""):
        """Gère la sélection des commentaires et le traitement des réponses."""
        transcript = campaign_video.get('transcript', '')
        url = campaign_video['url']

        # Initialiser les variables de session
        if f"{prefix}comments" not in st.session_state:
            st.session_state[f"{prefix}comments"] = []
        if f"{prefix}selected_comments" not in st.session_state:
            st.session_state[f"{prefix}selected_comments"] = {}
        if f"{prefix}generated_responses" not in st.session_state:
            st.session_state[f"{prefix}generated_responses"] = []
        if f"{prefix}selected_responses" not in st.session_state:
            st.session_state[f"{prefix}selected_responses"] = {}

        # Paramètres pour les commentaires
        st.subheader("Recherche des commentaires")

        if selected_video_indices:
            if st.button(t("promoteyoutube_export_channels"), key=f"{prefix}export_channels"):
                self.export_selected_channels(config, selected_video_indices, prefix)

            # Ajout du bouton d'export des vidéos sélectionnées
            if st.button(t("promoteyoutube_export_videos"), key=f"{prefix}export_videos"):
                self.export_selected_videos(config, selected_video_indices, prefix)

        max_comments_per_video = st.number_input(
            t("promoteyoutube_adjust_comments"),
            min_value=1,
            max_value=10,
            value=max_comments,
            key=f"{prefix}max_comments_per_video"
        )
        comment_order = st.selectbox(
            t("promoteyoutube_comment_order"),
            options=["relevance", "time"],
            index=1,
            key=f"{prefix}comment_order"
        )

        if st.button(t("promoteyoutube_fetch_comments"), key=f"{prefix}fetch_comments"):
            with st.spinner(t("promoteyoutube_getting_comments")):
                selected_videos = [
                    st.session_state[f"{prefix}videos"][i] for i in selected_video_indices]
                st.session_state[f"{prefix}comments"] = self.fetch_comments(
                    selected_videos, max_comments_per_video, comment_order)
                st.session_state[f"{prefix}selected_comments"] = {
                    i: False for i in range(len(st.session_state[f"{prefix}comments"]))}

        # Affichage et sélection des commentaires
        if st.session_state[f"{prefix}comments"]:
            st.subheader(t("promoteyoutube_comments"))

            col1, col2 = st.columns(2)
            with col1:
                if st.button(t("promoteyoutube_select_all_comments"), key=f"{prefix}select_all_comments"):
                    st.session_state[f"{prefix}selected_comments"] = {
                        i: True for i in range(len(st.session_state[f"{prefix}comments"]))}
            with col2:
                if st.button(t("promoteyoutube_deselect_all_comments"), key=f"{prefix}deselect_all_comments"):
                    st.session_state[f"{prefix}selected_comments"] = {
                        i: False for i in range(len(st.session_state[f"{prefix}comments"]))}

            for i, comment in enumerate(st.session_state[f"{prefix}comments"]):
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <p>{t("promoteyoutube_comment_context").format(comment['author'], comment['video_title'], comment['channel_title'])}</p>
                        <p>{comment['text']}</p>
                        <p><a href="https://www.youtube.com/watch?v={comment['video_id']}&lc={comment['id']}" target="_blank">{t("promoteyoutube_view_context")}</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.session_state[f"{prefix}selected_comments"][i] = st.checkbox(
                    f"Select Comment {i+1}",
                    value=st.session_state[f"{prefix}selected_comments"].get(
                        i, False),
                    key=f"{prefix}select_comment_{i}"
                )

            # Génération des réponses
            if st.button(t("promoteyoutube_generate_responses"), key=f"{prefix}generate_responses"):
                with st.spinner(t("promoteyoutube_generating")):
                    selected_comments = [
                        i for i, sel in st.session_state[f"{prefix}selected_comments"].items() if sel]
                    responses = self.generate_responses(
                        config, selected_comments, transcript, url, prefix, keywords)
                    st.session_state[f"{prefix}generated_responses"] = responses
                    st.session_state[f"{prefix}selected_responses"] = {
                        i: False for i in range(len(responses))}

                    campaign_id = datetime.now(pytz.UTC).isoformat()
                    for resp in responses:
                        comment = st.session_state[f"{prefix}comments"][resp['comment_index']]
                        cache_campaign_response(
                            campaign_id=campaign_id,
                            comment_id=comment['id'],
                            comment_text=comment['text'],
                            response_text=resp['response'],
                            video_id=comment['video_id'],
                            channel_id=comment['channel_id'],
                            author=comment['author'],
                            status="pending"
                        )
                    st.session_state[f"{prefix}campaign_id"] = campaign_id

            selected_comment_indices = [i for i, sel in st.session_state[f"{prefix}selected_comments"].items() if sel]
            if selected_comment_indices:
                if st.button(t("promoteyoutube_export_comments"), key=f"{prefix}export_comments"):
                    self.export_selected_comments(config, selected_comment_indices, prefix)

            # Affichage et publication des réponses
            if st.session_state.get(f"{prefix}generated_responses"):
                st.subheader(t("promoteyoutube_responses"))

                for i, response in enumerate(st.session_state[f"{prefix}generated_responses"]):
                    comment = st.session_state[f"{prefix}comments"][response['comment_index']]
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <p><strong>{comment['author']}</strong> sur <strong>{comment['channel_title']}</strong> (vidéo : <em>{comment['video_title']}</em>) :</p>
                            <p>{comment['text']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write(t("promoteyoutube_response_to_comment").format(
                        response['comment_index']+1))
                    edited_response = st.text_area(
                        t("promoteyoutube_edit_response").format(i+1),
                        value=response['response'],
                        key=f"{prefix}response_{i}",
                        height=100
                    )
                    st.session_state[f"{prefix}generated_responses"][i]['response'] = edited_response

                    if len(edited_response) > 500:
                        st.warning(t("promoteyoutube_char_limit_warning").format(
                            len(edited_response)))

                    st.session_state[f"{prefix}selected_responses"][i] = st.checkbox(
                        f"Select Response {i+1}",
                        value=st.session_state[f"{prefix}selected_responses"].get(
                            i, False),
                        key=f"{prefix}select_response_{i}"
                    )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(t("promoteyoutube_select_all_responses"), key=f"{prefix}select_all_responses"):
                        st.session_state[f"{prefix}selected_responses"] = {
                            i: True for i in range(len(st.session_state[f"{prefix}generated_responses"]))}
                with col2:
                    if st.button(t("promoteyoutube_deselect_all_responses"), key=f"{prefix}deselect_all_responses"):
                        st.session_state[f"{prefix}selected_responses"] = {
                            i: False for i in range(len(st.session_state[f"{prefix}generated_responses"]))}

                if st.button(t("promoteyoutube_post_responses"), key=f"{prefix}post_responses"):
                    with st.spinner(t("promoteyoutube_posting")):
                        selected_responses = [
                            resp for i, resp in enumerate(st.session_state[f"{prefix}generated_responses"])
                            if st.session_state[f"{prefix}selected_responses"][i]
                        ]
                        campaign_id = st.session_state.get(
                            f"{prefix}campaign_id", datetime.now(pytz.UTC).isoformat())
                        automarket = self.plugin_manager.get_plugin('automarket')
                        automarket.post_responses(
                            config, selected_responses, campaign_id)
                        st.success(t("promoteyoutube_success"))

    # Ajouter ces nouvelles méthodes dans la classe PromoteyoutubePlugin
    def export_selected_channels(self, config, selected_video_indices, prefix="promo_"):
        """Exporte les chaînes sélectionnées dans un fichier CSV."""
        try:
            work_dir = config['common']['work_directory']
            output_path = os.path.join(work_dir, "channel_list.csv")

            # Récupérer les vidéos sélectionnées
            selected_videos = [st.session_state[f"{prefix}videos"][i] for i in selected_video_indices]

            # Créer un dictionnaire pour éliminer les doublons (par channel_id)
            unique_channels = {}
            for video in selected_videos:
                channel_id = video.get('channel_id', 'unknown')
                if channel_id not in unique_channels:
                    unique_channels[channel_id] = {
                        'channel_id': channel_id,
                        'channel_title': video.get('channel_title', ''),
                        'subscriber_count': video.get('subscriber_count', 0),
                        'video_count': video.get('video_count', 0),
                        'keywords': st.session_state.get('keywords', '')
                    }

            # Créer le DataFrame et exporter
            import pandas as pd
            df = pd.DataFrame(list(unique_channels.values()))
            df.to_csv(output_path, index=False)

            st.success(t("promoteyoutube_export_success").format(output_path))
        except Exception as e:
            st.error(t("promoteyoutube_export_error").format(str(e)))

    def export_selected_videos(self, config, selected_video_indices, prefix="promo_"):
        """Exporte les vidéos sélectionnées dans un fichier CSV."""
        try:
            work_dir = config['common']['work_directory']
            output_path = os.path.join(work_dir, "video_list.csv")

            # Récupérer les vidéos sélectionnées
            selected_videos = [st.session_state[f"{prefix}videos"][i] for i in selected_video_indices]

            # Préparer les données
            videos_data = []
            for video in selected_videos:
                videos_data.append({
                    'video_id': video.get('video_id', ''),
                    'title': video.get('title', ''),
                    'url': video.get('url', ''),
                    'channel_id': video.get('channel_id', ''),
                    'channel_title': video.get('channel_title', ''),
                    'view_count': video.get('view_count', 0),
                    'comment_count': video.get('comment_count', 0),
                    'published_at': video.get('published_at', ''),
                    'language': video.get('language', ''),
                    'relevance_score': video.get('relevance_score', 0),
                    'keywords': st.session_state.get('keywords', '')
                })

            # Créer le DataFrame et exporter
            import pandas as pd
            df = pd.DataFrame(videos_data)
            df.to_csv(output_path, index=False)

            st.success(t("promoteyoutube_export_success").format(output_path))
        except Exception as e:
            st.error(t("promoteyoutube_export_error").format(str(e)))

    def export_selected_comments(self, config, selected_comment_indices, prefix="promo_"):
        """Exporte les commentaires sélectionnés dans un fichier CSV."""
        try:
            work_dir = config['common']['work_directory']
            output_path = os.path.join(work_dir, "comment_list.csv")

            # Récupérer les commentaires sélectionnés
            selected_comments = [st.session_state[f"{prefix}comments"][i] for i in selected_comment_indices]

            # Préparer les données
            comments_data = []
            for comment in selected_comments:
                comments_data.append({
                    'comment_id': comment.get('id', ''),
                    'comment_text': comment.get('text', ''),
                    'author': comment.get('author', ''),
                    'published_at': comment.get('published_at', ''),
                    'like_count': comment.get('like_count', 0),
                    'video_id': comment.get('video_id', ''),
                    'video_title': comment.get('video_title', ''),
                    'channel_id': comment.get('channel_id', ''),
                    'channel_title': comment.get('channel_title', ''),
                    'comment_url': f"https://www.youtube.com/watch?v={comment.get('video_id', '')}&lc={comment.get('id', '')}",
                    'keywords': st.session_state.get('keywords', '')
                })

            # Créer le DataFrame et exporter
            import pandas as pd
            df = pd.DataFrame(comments_data)
            df.to_csv(output_path, index=False)

            st.success(t("promoteyoutube_export_success").format(output_path))
        except Exception as e:
            st.error(t("promoteyoutube_export_error").format(str(e)))

    def run_campaign(self, config, target_videos, campaign_video, max_comments, prefix="promo_", keywords=""):
        """Exécute une campagne en deux étapes : sélection des vidéos puis des commentaires."""
        if f"{prefix}selected_video_indices" not in st.session_state:
            st.session_state[f"{prefix}selected_video_indices"] = []

        # Étape 1 : Affichage et sélection des vidéos
        selected_video_indices = self.display_and_select_videos(
            config, target_videos, prefix)
        if selected_video_indices != st.session_state[f"{prefix}selected_video_indices"]:
            st.session_state[f"{prefix}selected_video_indices"] = selected_video_indices
            # Réinitialiser les commentaires si la sélection change
            if f"{prefix}comments" in st.session_state:
                del st.session_state[f"{prefix}comments"]
            if f"{prefix}selected_comments" in st.session_state:
                del st.session_state[f"{prefix}selected_comments"]
            if f"{prefix}generated_responses" in st.session_state:
                del st.session_state[f"{prefix}generated_responses"]
            if f"{prefix}selected_responses" in st.session_state:
                del st.session_state[f"{prefix}selected_responses"]

        # Étape 2 : Gestion des commentaires si des vidéos sont sélectionnées
        if st.session_state[f"{prefix}selected_video_indices"]:
            self.select_and_process_comments(
                config, st.session_state[f"{prefix}selected_video_indices"], campaign_video, max_comments, prefix, keywords)

    def run(self, config):
        st.header(t("promoteyoutube_header"))
        work_dir = config['common']['work_directory']

        transcript_path = os.path.join(work_dir, "transcript.txt")
        transcript = st.text_area(
            t("promoteyoutube_transcript"),
            value=open(transcript_path, 'r').read(
            ) if os.path.exists(transcript_path) else "",
            height=200,
            key="promo_transcript",
            disabled=os.path.exists(transcript_path)
        )

        url_path = os.path.join(work_dir, "url.txt")
        url = st.text_input(
            t("promoteyoutube_url"),
            value=open(url_path, 'r').read().strip(
            ) if os.path.exists(url_path) else "",
            key="promo_url",
            disabled=os.path.exists(url_path)
        )
        if not 'keywords' in st.session_state:
            st.session_state['keywords'] = ''
        st.session_state.keywords = st.text_input(
            t("promoteyoutube_keywords"), key="promo_keywords", value=st.session_state.keywords)

        max_videos = st.number_input(
            t("promoteyoutube_max_videos_label"),
            min_value=1,
            max_value=50,
            value=int(config['promoteyoutube']['max_videos']),
            key="promo_max_videos"
        )

        # Persister target_videos dans session_state
        if "promo_target_videos" not in st.session_state:
            st.session_state["promo_target_videos"] = []

        if st.button(t("promoteyoutube_search"), key="promo_search"):
            with st.spinner(t("promoteyoutube_searching")):
                target_videos = self.search_videos(
                    st.session_state.keywords, max_videos, "relevance")
                st.session_state["promo_target_videos"] = target_videos

        # Utiliser target_videos depuis session_state
        if st.session_state["promo_target_videos"]:
            self.run_campaign(config, st.session_state["promo_target_videos"], {
                              'url': url, 'transcript': transcript}, 2, prefix="promo_", keywords=st.session_state.keywords)
