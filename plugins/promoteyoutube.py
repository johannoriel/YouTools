from global_vars import translations, t
from app import Plugin
import streamlit as st
import os
from plugins.ragllm import RagllmPlugin
from typing import List, Dict, Any, Optional
from youtube_api import YoutubeAPI  # Utilisation de l'API YouTube depuis social_api.py
from datetime import datetime
import pytz

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

    def generate_responses(self, config, selected_comments, transcript, url):
        ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)
        responses = []

        # Créer une barre de progression
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_comments = len(selected_comments)

        for idx, comment_index in enumerate(selected_comments):
            # Mise à jour de la progression
            progress = (idx + 1) / total_comments
            progress_bar.progress(progress)
            progress_text.text(t("promoteyoutube_progress").format(idx + 1, total_comments))

            comment = st.session_state.comments[comment_index]
            comment_with_context = t("promoteyoutube_comment_context").format(
                comment['author'],
                comment['video_title'],
                comment['channel_title']
            ) + f"\n{comment['text']}"

            prompt = config['promoteyoutube']['response_prompt'].format(
                url=url,
                transcript=transcript
            )

            llm_response = ragllm_plugin.process_with_llm(
                prompt,
                config.get('llm', {}).get('llm_sys_prompt', ''),
                comment_with_context
            )
            clean_response = remove_quotes(llm_response.strip())
            responses.append({
                'comment_id': comment['id'],
                'response': clean_response,
                'comment_index': comment_index
            })

        # Nettoyer la barre de progression à la fin
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
            progress_text.text(t("promoteyoutube_progress").format(idx + 1, total_errors))

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

    def post_responses(self, config, selected_responses):
        youtube_api = YoutubeAPI(self.plugin_manager.config)

        for response in selected_responses:
            comment_id = response['comment_id']
            response_text = response['response']
            youtube_api.post_comment_reply(comment_id, response_text)

    def fetch_comments_for_selected_videos(self, selected_video_indices, max_comments_per_video, comment_order):
        youtube_api = YoutubeAPI(self.plugin_manager.config)
        comments = []

        for idx in selected_video_indices:
            video = st.session_state.videos[idx]
            #st.info(video)
            video_comments = youtube_api.get_comments(video['video_id'], max_comments_per_video, order=comment_order)
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

    def run(self, config):
        st.header(t("promoteyoutube_header"))

        # Section 1: Configuration initiale et recherche de vidéos
        work_dir = config['common']['work_directory']

        # [Code existant pour transcript et URL]
        transcript_path = os.path.join(work_dir, "transcript.txt")
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r') as f:
                transcript = f.read()
            st.text_area(t("promoteyoutube_transcript"), transcript, height=100, disabled=True)
        else:
            transcript = st.text_area(t("promoteyoutube_transcript"), height=200)

        url_path = os.path.join(work_dir, "url.txt")
        if os.path.exists(url_path):
            with open(url_path, 'r') as f:
                url = f.read().strip()
            st.text_input(t("promoteyoutube_url"), url, disabled=True)
        else:
            url = st.text_input(t("promoteyoutube_url"))

        # Ajout du sélecteur de catégorie
        categories = {
            0: "All Categories", 1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals",
            17: "Sports", 18: "Short Movies", 19: "Travel & Events", 20: "Gaming", 22: "People & Blogs", 23: "Comedy",
            24: "Entertainment", 25: "News & Politics", 26: "Howto & Style", 27: "Education", 28: "Science & Technology",
            29: "Nonprofits & Activism"
        }
        selected_category = st.selectbox("Select Category", options=list(categories.keys()), format_func=lambda x: categories[x])

        if st.button("Get Trending"):
            with st.spinner("Fetching trending videos..."):
                youtube_api = YoutubeAPI(self.plugin_manager.config)
                trending_videos = youtube_api.get_trending_videos(language=st.session_state.lang, category_id=selected_category)

                if trending_videos:
                    for video in trending_videos:
                        st.markdown(f"**[{video['title']}]({video['url']})** - {video['channel_title']} ({video['view_count']} views)")
                else:
                    st.warning("No trending videos found.")

        # Configuration de la recherche de vidéos
        if 'keywords' not in st.session_state:
            st.session_state.keywords = ""

        keywords = st.text_input(
            t("promoteyoutube_keywords"),
            value=st.session_state.keywords,
            key="promoteyoutube_keywords"
        )
        st.session_state.keywords = keywords

        if not keywords:
            st.warning(t("promoteyoutube_keywords_warning"))
            return

        max_videos = st.number_input(
            t("promoteyoutube_max_videos_label"),
            min_value=1,
            max_value=50,
            value=int(config['promoteyoutube']['max_videos']),
            key="max_videos"
        )

        # L'ordre de l'API reste le même
        video_order = st.selectbox(
            t("promoteyoutube_video_order"),
            options=["date", "relevance", "viewCount", "rating"],
            index=1,
            key="video_order"
        )

        youtube_api = YoutubeAPI(self.plugin_manager.config)
        # Bouton de recherche des vidéos
        if st.button(t("promoteyoutube_search")):
            with st.spinner(t("promoteyoutube_searching")):
                youtube_api = YoutubeAPI(self.plugin_manager.config)
                videos = youtube_api.search_videos(keywords, max_videos, order=video_order, language=st.session_state.lang)
                st.session_state.videos = videos
                st.session_state.original_order = videos.copy()  # Sauvegarder l'ordre original
                st.session_state.selected_videos = {i: False for i in range(len(videos))}
                st.session_state.show_comments_section = False
                st.session_state.comments = []

        # Section 2: Affichage et sélection des vidéos
        if st.session_state.videos:
            st.subheader(t("promoteyoutube_select_videos"))

            # Ajout du filtre de langue
            all_languages = list(set(video['language'] for video in st.session_state.videos if video['language'] != 'unknown'))
            selected_language = st.selectbox(
                t("promoteyoutube_filter_language"),
                [t("promoteyoutube_show_all_languages")] + all_languages,
                index=0
            )

            # Création d'une map des indices pour maintenir la correspondance
            filtered_videos = st.session_state.videos
            filtered_indices = list(range(len(st.session_state.videos)))
            if selected_language != t("promoteyoutube_show_all_languages"):
                filtered_indices = [i for i, v in enumerate(st.session_state.videos) if v['language'] == selected_language]
                filtered_videos = [st.session_state.videos[i] for i in filtered_indices]

            sort_by = st.selectbox(
                t("promoteyoutube_sort_by"),
                options=[t("promoteyoutube_sort_api"), t("promoteyoutube_sort_relevance")],
                index=0,
                key="sort_by",
                on_change=lambda: setattr(st.session_state, 'videos',
                    self._sort_videos(st.session_state.original_order.copy(), st.session_state.sort_by))
            )

            # Boutons Select All/Deselect All pour les vidéos
            col1, col2 = st.columns(2)
            if col1.button(t("promoteyoutube_select_all_videos")):
                for i in filtered_indices:
                    st.session_state.selected_videos[i] = True
            if col2.button(t("promoteyoutube_deselect_all_videos")):
                for i in filtered_indices:
                    st.session_state.selected_videos[i] = False

            # Affichage des vidéos filtrées avec les indices originaux
            for display_index, original_index in enumerate(filtered_indices):
                video = st.session_state.videos[original_index]
                published_at = datetime.strptime(video['published_at'], "%Y-%m-%dT%H:%M:%SZ")
                days_ago = (datetime.now(pytz.UTC) - published_at.replace(tzinfo=pytz.UTC)).days

                st.markdown(f"[**{video['title']}**]({video['url']})")
                st.markdown(f"Chaîne : **[{video['channel_title']}](https://www.youtube.com/channel/{video['channel_id']})**")

                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                col1.markdown(f"{t('promoteyoutube_subscribers')} : **{youtube_api.format_count(video['subscriber_count'])}**")
                col2.markdown(f"{t('promoteyoutube_views')} : **{youtube_api.format_count(video['view_count'])}**")
                col3.markdown(f"{t('promoteyoutube_comments_count')} : **{video['comment_count']}**")
                col4.markdown(f"**{days_ago}** jours")
                col5.markdown(f"Score : **{video['relevance_score']}**/100")
                col6.markdown(f"{t('promoteyoutube_language')} : **{video['language']}**")

                # Utilisation de l'indice original pour la checkbox
                st.session_state.selected_videos[original_index] = col7.checkbox(
                    t("promoteyoutube_select_video"),
                    key=f"video_{original_index}",
                    value=st.session_state.selected_videos.get(original_index, False)
                )

            # Utilisation des indices originaux pour la sélection des vidéos
            selected_video_indices = [i for i, selected in st.session_state.selected_videos.items() if selected]

            if selected_video_indices:
                st.subheader("Recherche des commentaires")

                max_comments_per_video = st.number_input(
                    t("promoteyoutube_adjust_comments"),
                    min_value=1,
                    max_value=10,
                    value=int(config['promoteyoutube']['max_comments_per_video']),
                    key="max_comments_per_video_adjusted"
                )

                comment_order = st.selectbox(
                    t("promoteyoutube_comment_order"),
                    options=["relevance", "time"],
                    index=1,
                    key="comment_order"
                )

                if st.button(t("promoteyoutube_fetch_comments")):
                    with st.spinner(t("promoteyoutube_getting_comments")):
                        st.session_state.comments = self.fetch_comments_for_selected_videos(
                            selected_video_indices,
                            max_comments_per_video,
                            comment_order
                        )
                        st.session_state.show_comments_section = True
                        st.session_state.selected_comments = {i: False for i in range(len(st.session_state.comments))}

            # Section 4: Affichage des commentaires et suite du processus
            if st.session_state.show_comments_section and st.session_state.comments:
                st.subheader(t("promoteyoutube_comments"))

                # Boutons Select All/Deselect All pour les commentaires
                col1, col2 = st.columns(2)
                if col1.button("Select All Comments"):
                    st.session_state.selected_comments = {i: True for i in range(len(st.session_state.comments))}
                if col2.button("Deselect All Comments"):
                    st.session_state.selected_comments = {i: False for i in range(len(st.session_state.comments))}

                for i, comment in enumerate(st.session_state.comments):
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

                    selected = st.checkbox(
                        f"Select Comment {i+1}",
                        value=st.session_state.selected_comments.get(i, False),
                        key=f"select_comment_{i}"
                    )
                    st.session_state.selected_comments[i] = selected

                # [Code existant pour la génération et l'envoi des réponses]
                if st.button(t("promoteyoutube_generate_responses")) and any(st.session_state.selected_comments.values()):
                    with st.spinner(t("promoteyoutube_generating")):
                        selected_comments = [i for i, selected in st.session_state.selected_comments.items() if selected]
                        st.session_state.generated_responses = self.generate_responses(
                            config, selected_comments, transcript, url
                        )

                # [Rest of the existing code for displaying and posting responses]
                if 'generated_responses' in st.session_state and st.session_state.generated_responses:

                    st.subheader(t("promoteyoutube_responses"))
                    # [Code existant pour l'affichage et la gestion des réponses]
                    for i, response in enumerate(st.session_state.generated_responses):
                        comment_index = response['comment_index']
                        comment = st.session_state.comments[comment_index]

                        st.markdown(
                            f"""
                            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                <p><strong>{comment['author']}</strong> sur la chaîne <strong>{comment['channel_title']}</strong> (vidéo : <em>{comment['video_title']}</em>) :</p>
                                <p>{comment['text']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.write(t("promoteyoutube_response_to_comment").format(comment_index+1))
                        edited_response = st.text_area(
                            t("promoteyoutube_edit_response").format(i+1),
                            value=response['response'],
                            key=f"response_{i}",
                            height=100
                        )
                        st.session_state.generated_responses[i]['response'] = edited_response

                        if len(edited_response) > 500:
                            st.warning(t("promoteyoutube_char_limit_warning").format(len(edited_response)))

                        if i not in st.session_state.selected_responses:
                            st.session_state.selected_responses[i] = False

                        selected = st.checkbox(
                            f"Select Response {i+1}",
                            value=st.session_state.selected_responses.get(i, False),
                            key=f"select_response_{i}"
                        )
                        st.session_state.selected_responses[i] = selected

                    # Boutons Select All/Deselect All pour les réponses
                    col1, col2 = st.columns(2)
                    if col1.button(t("promoteyoutube_select_all_responses")):
                        st.session_state.selected_responses = {i: True for i in range(len(st.session_state.generated_responses))}
                    if col2.button(t("promoteyoutube_deselect_all_responses")):
                        st.session_state.selected_responses = {i: False for i in range(len(st.session_state.generated_responses))}

                    col1, col2 = st.columns(2)
                    if col1.button(t("promoteyoutube_regenerate_errors")):
                        with st.spinner(t("promoteyoutube_regenerating")):
                            self.regenerate_error_responses(config, transcript, url)
                    # Post responses button
                    if col2.button(t("promoteyoutube_post_responses")) and any(st.session_state.selected_responses.values()):
                        with st.spinner(t("promoteyoutube_posting")):
                            selected_responses = [
                                response for i, response in enumerate(st.session_state.generated_responses)
                                if st.session_state.selected_responses[i]
                            ]
                            self.post_responses(config, selected_responses)
                            st.success(t("promoteyoutube_success"))
                    # Compter les erreurs
                    error_count = sum(1 for response in st.session_state.generated_responses
                                        if "litellm.APIError" in response['response'])
                    # Afficher le compteur d'erreurs s'il y en a
                    if error_count > 0:
                        st.warning(t("promoteyoutube_error_count").format(error_count))
