from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import pandas as pd
from widgets.video_list import VideoListWidget
from widgets.theme_selector import ThemeSelectorWidget
from widgets.product_matcher import VideoProductMatchWidget
from widgets.utils import export_responses, remove_quotes
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from datetime import datetime
from lib.youtube_db import cache_campaign_response

translations["en"].update({
    "video_product_promotion_title": "Video-Product Promotion",
    "no_videos_selected": "No videos selected. Please select at least one video.",
    "no_products_matched": "No products matched for the selected videos.",
    "generate_promotions": "Generate Promotional Messages",
    "generating_promotions": "Generating promotional messages...",
    "promotional_messages": "Promotional Messages",
    "char_limit_warning": "⚠️ This message exceeds 500 characters ({} characters). Please shorten it.",
    "export_promotions": "Export Promotional Messages",
    "prompt_label": "LLM Prompt for Promotional Messages",
    "prompo_default_prompt": "Generate a concise promotional message (<500 chars) for the product at {url} (mandatory in result), tailored to the video with keywords. Use a direct tone, as if you're a viewer",
    'promo_sys_prompt': "You are an expert in creating promotional messages for products. Your task is to generate a concise promotional message for the product.",
    "theme_mapping": "Map Keywords to Themes",
    "no_themes_matched": "No themes matched for the selected videos' keywords.",
    "matched_themes": "Matched Themes for Video Keywords",
    "score_threshold": "Score Threshold for Product Matching",
    "products_list": "Available Products",
    "calculate_pairs": "Calculate Video-Product Pairs",
    "video_product_pairs": "Matched Video-Product Pairs",
    "overwrite_responses_checkbox": "Overwrite Existing Responses"
})

translations["fr"].update({
    "video_product_promotion_title": "Promotion Vidéo-Produit",
    "no_videos_selected": "Aucune vidéo sélectionnée. Veuillez sélectionner au moins une vidéo.",
    "no_products_matched": "Aucun produit correspondant pour les vidéos sélectionnées.",
    "generate_promotions": "Générer des messages promotionnels",
    "generating_promotions": "Génération des messages promotionnels...",
    "promotional_messages": "Messages promotionnels",
    "char_limit_warning": "⚠️ Ce message dépasse 500 caractères ({} caractères). Veuillez le raccourcir.",
    "export_promotions": "Exporter les messages promotionnels",
    "prompt_label": "Prompt LLM pour les messages promotionnels",
    "promo_default_prompt": "Générer un message promotionnel concis (<500 chars) pour le produit '{product}' à l'url '{url}' (l'URL doit obligatoirement être mentionnée dans le résultat), qui est une réponse de l'auteur du produit à la vidéo '{video}' d'une chaîne youtube externe ou l'on veut acquérir des prospect.",
    "promo_sys_prompt": "Vous êtes un assistant de marketing qui est l'auteur qui vient promouvoir ses produits. Votre message doit être concis et donner envie de consulter le produit.",
    "theme_mapping": "Associer les mots-clés aux thèmes",
    "no_themes_matched": "Aucun thème correspondant aux mots-clés des vidéos sélectionnées.",
    "matched_themes": "Thèmes correspondants pour les mots-clés des vidéos",
    "score_threshold": "Seuil de score pour la correspondance des produits",
    "products_list": "Produits disponibles",
    "calculate_pairs": "Calculer les paires vidéo-produit",
    "video_product_pairs": "Paires vidéo-produit correspondantes",
    "overwrite_responses_checkbox": "Écraser le fichier de réponses existant",
})

@st.dialog("Alert")
def alert(message):
    st.write(message)

class VideoProductPromotionWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.work_dir = self.plugin_manager.config["common"]["work_directory"]

    def map_keywords_to_themes(self, keywords, theme_selector):
        """
        Associe les mots-clés des vidéos aux thématiques disponibles.

        Args:
            keywords: Liste de mots-clés extraits des vidéos
            theme_selector: Instance de ThemeSelectorWidget

        Returns:
            Liste de thématiques correspondantes
        """
        matched_themes = []
        for keyword in keywords:
            theme = theme_selector.get_theme_for_keyword(keyword)
            if theme and theme not in matched_themes:
                matched_themes.append(theme)

        # Si aucune thématique n'est trouvée, récupérer toutes les thématiques disponibles
        if not matched_themes:
            keywords_config = self.plugin_manager.config.get("trendwatcher", {}).get(
                "trendwatcher_keywords", t("trendwatcher_keywords_default"))
            themes = []
            for line in keywords_config.split("\n"):
                line = line.strip()
                if line:
                    parts = line.split(":", 2)
                    theme = parts[0].strip() if len(parts) >= 2 else "Default"
                    if theme not in themes:
                        themes.append(theme)
            return themes
        return matched_themes

    def generate_response_for_content(self, content_dict, prompt_template, sys_prompt):
        """
        Génère une réponse pour un contenu donné (commentaire, vidéo, etc.) en utilisant un prompt LLM.
        Args:
            content_dict: Dictionnaire contenant les informations du contenu
            prompt_template: Modèle de prompt pour le LLM
            sys_prompt: Prompt système pour le LLM

        Returns:
            Dictionnaire contenant la réponse générée et les métadonnées
        """
        video_rag = f"Description of video {content_dict.get('video_title', '')} to be responded:\n {content_dict.get('video_description', '')}"
        keywords_rag = f"Keywords describing the link between the video and the product:\n {content_dict.get('keywords', '')}"
        product_rag = f"Description of the product {content_dict.get('product_title', '')} of type {content_dict.get('product_type', '')} to promote:\n {content_dict.get('product_content', '')}"
        prompt = prompt_template.format(url=content_dict['product_url'], product=content_dict['product_title'], video=content_dict['video_title'])
        prompts = [video_rag,keywords_rag,product_rag,prompt]
        try:
            llm_response = self.process_with_llm(prompts,sys_prompt)
            clean_response = remove_quotes(llm_response.strip())
        except Exception as e:
            clean_response = f"Error: {str(e)}"

        return {
            'comment_id': content_dict.get('comment_id', ''),
            'response': clean_response,
            'target_video_id': content_dict.get('video_id', ''),
            'channel_id': content_dict.get('channel_id', ''),
            'keyword': content_dict.get('keyword',''),
            'author': content_dict.get('author', ''),
            'video_title': content_dict.get('video_title', ''),
            'video_description': content_dict.get('video_description', ''),
            'channel_title': content_dict.get('channel_title', ''),
            'product_title': content_dict.get('product_title', ''),
            'product_type': content_dict.get('product_type', ''),
        }

    def generate_responses_for_list(self, content_list, prompt_template, sys_prompt):
        """
        Génère des réponses pour une liste de contenus.

        Args:
            content_list: Liste de dictionnaires de contenus
            prompt_template: Modèle de prompt pour le LLM
            sys_prompt: Prompt système pour le LLM

        Returns:
            Liste de réponses générées
        """
        responses = []
        total_items = len(content_list)
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for idx, content in enumerate(content_list):
            progress = (idx + 1) / total_items
            progress_bar.progress(progress)
            progress_text.text(f"Processing item {idx + 1} of {total_items}")

            response = self.generate_response_for_content(content, prompt_template, sys_prompt)
            responses.append(response)

        progress_bar.empty()
        progress_text.empty()
        return responses

    def display(self):
        st.title(t("video_product_promotion_title"))

        # Étape 1 : Sélection des vidéos avec VideoListWidget
        video_list_widget = VideoListWidget("video_list", f"{self.prefix}_video_list", self.plugin_manager)
        selected_rows, filtered_df = video_list_widget.select_video_list()

        if not selected_rows:
            st.warning(t("no_videos_selected"))
            return

        selected_videos = filtered_df.iloc[selected_rows]

        # Étape 2 : Afficher les thématiques correspondantes aux mots-clés des vidéos
        st.subheader(t("matched_themes"))
        theme_selector = ThemeSelectorWidget("theme_selector", f"{self.prefix}_theme_selector", self.plugin_manager)

        # Extraire les mots-clés des vidéos
        video_keywords = []
        for _, row in selected_videos.iterrows():
            kws = str(row['keyword']).split(',') if pd.notnull(row['keyword']) else []
            video_keywords.extend([kw.strip() for kw in kws if kw.strip()])
        video_keywords = list(set(video_keywords))

        # Associer les mots-clés aux thématiques
        matched_themes = self.map_keywords_to_themes(video_keywords, theme_selector)

        if not matched_themes:
            st.warning(t("no_themes_matched"))
            return

        # Afficher les thématiques trouvées
        st.write("**Matched Themes**: " + ", ".join(matched_themes))

        # Étape 3 : Sélection des thématiques
        st.subheader(t("theme_mapping"))
        selected_themes = st.multiselect(
            t("themeselector_select_themes"),
            matched_themes,
            default=matched_themes,
            key=f"{self.prefix}_theme_filter"
        )

        if not selected_themes:
            st.warning("Please select at least one theme.")
            return

        # Récupérer les mots-clés pour les thématiques sélectionnées
        keywords = []
        for theme in selected_themes:
            theme_keywords = theme_selector.get_keywords_for_theme(theme)
            for kw in theme_keywords:
                keywords.append(kw["main"])
                keywords.extend(kw["synonyms"])
        keywords = list(set(keywords))

        # Étape 4 : Afficher les produits disponibles
        st.subheader(t("products_list"))
        product_matcher = VideoProductMatchWidget("product_matcher", f"{self.prefix}_product_matcher", self.plugin_manager)
        videos_df = selected_videos.copy()
        products = product_matcher.db.get_all_products()

        if not products:
            st.error(t("no_products_error"))
            return

        # Étape 5 : Configurer le seuil de score et calculer les paires
        st.subheader(t("score_threshold"))
        score_threshold = st.number_input(
            t("score_threshold"),
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            key=f"{self.prefix}_score_threshold"
        )

        if st.button(t("calculate_pairs"), key=f"{self.prefix}_calculate_pairs"):
            # Calculer les scores de pertinence
            scores_df, _, products, video_keywords_list = product_matcher.calculate_relevance_scores(
                videos_df, products, "manual_count", "keywords_only"
            )

            if scores_df is None:
                st.error(t("no_products_matched"))
                return

            # Sélectionner un produit par vidéo (le meilleur score)
            video_product_pairs = []
            for video_id in scores_df.index:
                scores = scores_df.loc[video_id]
                max_score = scores.max()
                if max_score > score_threshold:
                    top_product_id = scores.idxmax()
                    video_row = videos_df[videos_df['video_id'] == video_id].iloc[0]
                    product = next(p for p in products if str(p['id']) == top_product_id)
                    video_product_pairs.append({
                        'comment_id': f"{video_id}_{top_product_id}",
                        'video_id': video_id,
                        'video_url' : video_row['url'],
                        'channel_id': video_row['channel_id'],
                        'video_title': video_row['title'],
                        'channel_title': video_row['channel_title'],
                        'video_description': video_row['description'],
                        'author': '',
                        'product_id': top_product_id,
                        'product_title': product['title'],
                        'product_type': product['type'],
                        'product_description': product['description'],
                        'product_content': product['content'],
                        'product_url': product['url'],
                        'keywords': ', '.join(video_keywords_list[videos_df.index.get_loc(video_row.name)])
                    })

            st.session_state[f"{self.prefix}_video_product_pairs"] = video_product_pairs
            if not video_product_pairs:
                st.error(t("no_products_matched"))

        # Étape 6 : Afficher les paires vidéo-produit (si disponibles)
        if f"{self.prefix}_video_product_pairs" in st.session_state and st.session_state[f"{self.prefix}_video_product_pairs"]:
            st.subheader(t("video_product_pairs"))
            pairs_df = pd.DataFrame(st.session_state[f"{self.prefix}_video_product_pairs"])
            selected_pairs = st.dataframe(
                pairs_df[["video_title", "video_url", "channel_title", "product_title", "product_url", "keywords"]],
                column_config={
                    "video_title": st.column_config.TextColumn("Video Title", width="large"),
                    "video_url": st.column_config.LinkColumn("Video URL", display_text="Watch"),
                    "channel_title": st.column_config.TextColumn("Channel", width="medium"),
                    "product_title": st.column_config.TextColumn("Product Title", width="large"),
                    "product_url": st.column_config.LinkColumn("Product URL", display_text="Visit"),
                    "keywords": st.column_config.TextColumn("Keywords", width="large")
                },
                use_container_width=True,
                height=400,
                selection_mode="multi-row",
                on_select="rerun",
                key=f"{self.prefix}_pairs_dataframe"
            )

            selected_pair_rows = selected_pairs.get('selection', {}).get('rows', [])
            selected_pairs_list = [st.session_state[f"{self.prefix}_video_product_pairs"][i] for i in selected_pair_rows]

        else:
            selected_pairs_list = []

        # Étape 7 : Configurer et afficher le prompt
        default_prompt = t("promo_default_prompt")
        prompt_key = f"{self.prefix}_promotion_prompt"
        if prompt_key not in st.session_state:
            st.session_state[prompt_key] = default_prompt

        st.subheader(t("prompt_label"))
        prompt_template = st.text_area(
            t("prompt_label"),
            value=st.session_state[prompt_key],
            height=150,
            key=prompt_key
        )

        # Étape 8 : Générer les messages promotionnels pour les paires sélectionnées
        if st.button(t("generate_promotions"), key=f"{self.prefix}_generate_promotions") and selected_pairs_list:
            with st.spinner(t("generating_promotions")):
                responses = self.generate_responses_for_list(
                    content_list=selected_pairs_list,
                    prompt_template=prompt_template,
                    sys_prompt= t("promo_sys_prompt")
                )

                st.session_state[f"{self.prefix}_generated_promotions"] = responses

        # Étape 9 : Afficher et valider les messages
        if f"{self.prefix}_generated_promotions" in st.session_state and st.session_state[f"{self.prefix}_generated_promotions"]:
            st.subheader(t("promotional_messages"))
            responses_df = pd.DataFrame([
                {
                    'comment_text': resp['video_description'],
                    'response_text': resp['response'],
                    'video_title': resp['video_title'],
                    'channel_title': resp['channel_title'],
                    'comment_id': resp['comment_id'],
                    'video_id': resp['target_video_id'],
                    'channel_id': resp['channel_id'],
                    'keyword': resp['keyword'],
                    'product_title': resp.get('product_title', ''),
                }
                for resp in st.session_state[f"{self.prefix}_generated_promotions"]
            ])

            # Configurer le DataFrame pour les réponses
            selected_responses = st.dataframe(
                responses_df[["response_text", "video_title", "channel_title", "product_title"]],
                column_config={
                    "response_text": st.column_config.TextColumn("Promotional Message", width="large"),
                    "video_title": st.column_config.TextColumn("Video Title", width="medium"),
                    "channel_title": st.column_config.TextColumn("Channel", width="medium"),
                    "product_title": st.column_config.TextColumn("Product Title", width="medium")
                },
                use_container_width=True,
                height=400,
                selection_mode="multi-row",
                on_select="rerun",
                key=f"{self.prefix}_responses_dataframe"
            )

            # Récupérer les réponses sélectionnées
            selected_response_rows = selected_responses.get('selection', {}).get('rows', [])
            selected_responses_list = [st.session_state[f"{self.prefix}_generated_promotions"][i] for i in selected_response_rows]

            # Étape 10 : Exporter les messages sélectionnés
            overwrite = st.checkbox(t("overwrite_responses_checkbox"), key=f"{self.prefix}_overwrite_promotions")
            if st.button(t("export_promotions"), key=f"{self.prefix}_export_promotions") and selected_response_rows:
                export_responses(selected_responses_list, self.work_dir, "response_list.csv", overwrite)
