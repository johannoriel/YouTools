from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
from lib.youtube_api import YoutubeAPI
from lib.youtube_db import *
from typing import List, Dict, Any, Optional
from datetime import datetime
import pytz
from langdetect import detect

# Ajout des traductions spécifiques au plugin Automarket
translations["en"].update({
    "automarket_tab": "Automarket",
    "automarket_header": "Automated YouTube Marketing Campaign",
    "automarket_select_video": "Select Video to Promote",
    "automarket_comments_per_video": "Comments per Video",
    "automarket_min_subscribers": "Minimum Subscribers",
    "automarket_max_videos_per_keyword": "Max Videos per Keyword",
    "automarket_expiry_days": "Expiry Days",
    "automarket_view_threshold": "View Threshold",
    "automarket_trusted_channel_videos": "Videos from Trusted Channels",
    "automarket_start_campaign": "Start Campaign",
    "automarket_processing": "Processing campaign...",
    "automarket_generating_responses": "Generating reséponses...",
    "automarket_rejected_videos": "Rejected Videos (Log)",
    "automarket_responses": "Generated Responses",
    "automarket_post_responses": "Post Responses",
    "automarket_channel": "Channel",
    "automarket_video": "Video",
    "automarket_comment": "Comment",
    "automarket_response": "Response",
    "automarket_add_to_trusted": "Add to Trusted Channels",
    "automarket_exclude_response": "Exclude Response",
    "automarket_campaign_complete": "Campaign completed successfully!",
    "automarket_no_videos_with_keywords": "No videos with keywords found in the database.",
    "automarket_rejected_reason": "Rejected: {}",
    "automarket_keyword": "Keyword",
    "automarket_criterion": "Criterion",
    "automarket_expand_all": "Expand All",
    "automarket_collapse_all": "Collapse All",
    "automarket_debug_mode": "Debug Mode",
    "automarket_max_comments_debug": "Max Comments in Debug Mode",
    "automarket_excluded_comments": "Excluded Comments (Log)",
    "automarket_exclusion_keyword": "Exclusion Keyword",
    "automarket_quota_consumed": "Quota consumed during campaign: {units} units",
    "automarket_progress": "Progression...",
    "automarket_search_keywords": "Search by Keywords",
    "automarket_search_trusted": "Search in Trusted Channels",
    "automarket_reponse_to_comment": "Response to:",
    "automarket_posting": "Posting responses...",
    "automarket_select_all": "Select All",
    "automarket_deselect_all": "Deselect All",
    "monitor_trends_tab": "Monitor Trends",
    "monitor_trends_header": "Monitor YouTube Trends",
    "monitor_trends_select_keyword": "Select Campaign Keyword",
    "monitor_trends_search_criteria": "Search Criteria",
    "monitor_trends_time": "Recent Videos (Time)",
    "monitor_trends_relevant": "Relevant Videos",
    "monitor_trends_trusted": "Trusted Channels",
    "automarket_selected_videos": "Selected Videos (Log)",
    "automarket_moderated_responses": "Responses in moderation",
})

translations["fr"].update({
    "automarket_tab": "Automarket",
    "automarket_header": "Campagne Marketing Automatisée sur YouTube",
    "automarket_select_video": "Sélectionner la vidéo à promouvoir",
    "automarket_comments_per_video": "Commentaires par vidéo",
    "automarket_min_subscribers": "Abonnés minimum",
    "automarket_max_videos_per_keyword": "Vidéos max par mot-clé",
    "automarket_expiry_days": "Jours d'expiration",
    "automarket_view_threshold": "Seuil de vues",
    "automarket_trusted_channel_videos": "Vidéos des chaînes de confiance",
    "automarket_start_campaign": "Lancer la campagne",
    "automarket_processing": "Traitement de la campagne...",
    "automarket_generating_responses": "Génération des réponses...",
    "automarket_rejected_videos": "Vidéos rejetées (Log)",
    "automarket_responses": "Réponses générées",
    "automarket_post_responses": "Poster les réponses",
    "automarket_channel": "Chaîne",
    "automarket_video": "Vidéo",
    "automarket_comment": "Commentaire",
    "automarket_response": "Réponse",
    "automarket_add_to_trusted": "Ajouter aux chaînes de confiance",
    "automarket_exclude_response": "Exclure la réponse",
    "automarket_campaign_complete": "Campagne terminée avec succès !",
    "automarket_no_videos_with_keywords": "Aucune vidéo avec mots-clés trouvée dans la base de données.",
    "automarket_rejected_reason": "Rejetée : {}",
    "automarket_keyword": "Mot-clé",
    "automarket_criterion": "Critère",
    "automarket_expand_all": "Tout déplier",
    "automarket_collapse_all": "Tout replier",
    "automarket_debug_mode": "Mode Debug",
    "automarket_max_comments_debug": "Nombre max de commentaires en mode Debug",
    "automarket_excluded_comments": "Commentaires Exclus (Log)",
    "automarket_exclusion_keyword": "Mot-clé d'Exclusion",
    "automarket_quota_consumed": "Quota consommé pendant la campagne : {units} unités",
    "automarket_progress": "Progression...",
    "automarket_search_keywords": "Recherche par Mots-clés",
    "automarket_search_trusted": "Recherche dans les Chaînes de Confiance",
    "automarket_reponse_to_comment": "Réponse à:",
    "automarket_posting": "Post des réponses ...",
    "automarket_select_all": "Tout sélectionner",
    "automarket_deselect_all": "Tout désélectionner",
    "monitor_trends_tab": "Surveiller les Tendances",
    "monitor_trends_header": "Surveiller les Tendances YouTube",
    "monitor_trends_select_keyword": "Sélectionner un Mot-clé de Campagne",
    "monitor_trends_search_criteria": "Critères de Recherche",
    "monitor_trends_time": "Vidéos Récentes (Temps)",
    "monitor_trends_relevant": "Vidéos Pertinentes",
    "monitor_trends_trusted": "Chaînes de Confiance",
    "automarket_selected_videos": "Vidéos retenues (Log)",
    "automarket_moderated_responses": "Réponses en modération",
})


class AutomarketPlugin(Plugin):
    def __init__(self, name, plugin_manager):
        super().__init__(name, plugin_manager)
        self.youtube_api = YoutubeAPI(self.plugin_manager.config)
        self._initialize_session_state()

    def _initialize_session_state(self):
        if 'campaign_responses' not in st.session_state:
            st.session_state.campaign_responses = []
        if 'selected_responses' not in st.session_state:
            st.session_state.selected_responses = {}
        if 'rejected_videos' not in st.session_state:
            st.session_state.rejected_videos = []
        if 'expand_all' not in st.session_state:
            st.session_state.expand_all = True
        if 'excluded_comments' not in st.session_state:
            st.session_state.excluded_comments = []
        if 'campaign_target_videos' not in st.session_state:
            st.session_state.campaign_target_videos = []
        if 'campaign_current_comments' not in st.session_state:
            st.session_state.campaign_current_comments = []
        if 'campaign_responses' not in st.session_state:
            st.session_state.campaign_responses = []
        if 'campaign_excluded_comments' not in st.session_state:
            st.session_state.campaign_excluded_comments = []
        # Add this line to initialize target_videos
        if 'target_videos' not in st.session_state:
            st.session_state.target_videos = []

    def get_config_fields(self):
        return {
            "comments_per_video": {
                "type": "number",
                "label": "Default Comments per Video",
                "default": 2
            },
            "min_subscribers": {
                "type": "number",
                "label": "Minimum Subscribers",
                "default": 10000
            },
            "max_videos_per_keyword": {
                "type": "number",
                "label": "Max Videos per Keyword",
                "default": 10
            },
            "expiry_days": {
                "type": "number",
                "label": "Expiry Days",
                "default": 7
            },
            "view_threshold": {
                "type": "number",
                "label": "View Threshold",
                "default": 1000
            },
            "trusted_channel_videos": {
                "type": "number",
                "label": "Videos from Trusted Channels",
                "default": 3
            },
            "response_prompt": {
                "type": "textarea",
                "label": "LLM Prompt for Responses",
                "default": """Suggest a concise response (<500 chars) to this comment, promoting the video at {url} (mention it). Use a direct tone, as if you're the commenter, inspired by this transcript: {transcript}"""
            },
            "exclusion_keyword": {
                "type": "text",
                "label": "Keyword to Exclude Comments",
                "default": "Stop"
            },
            "campaign_keywords": {
                "type": "text",
                "label": "Default Campaign Keywords",
                "default": ""
            },
            "max_campaign_videos": {
                "type": "number",
                "label": "Default Max Videos for Campaign",
                "default": 10
            },
            "max_campaign_comments": {
                "type": "number",
                "label": "Default Max Comments per Video",
                "default": 2
            },
            "response_prompt": {
                "type": "textarea",
                "label": "LLM Prompt for Campaign Responses",
                "default": """Suggest a concise response (<500 chars) to this comment, promoting the video at {url} (mention it). Use a direct tone, as if you're the commenter, inspired by this transcript: {transcript}"""
            }
        }

    def get_tabs(self):
        return [
            {"name": "Lancer une campagne", "plugin": "automarket"},
            {"name": "Réponses existantes", "plugin": "automarket"},
            {"name": "Surveiller les tendances", "plugin": "automarket"}
        ]

    def post_responses(self, config, selected_responses, campaign_timestamp: str):
        """Poste les réponses et les sauvegarde dans la base et stats."""
        from widgets.post_response import PostResponseWidget
        post_response = PostResponseWidget("promoteyoutube", "prw",
                           plugin_manager=self.plugin_manager)
        post_response.post_responses(selected_responses, campaign_timestamp)

        # Calcul des statistiques
        total_videos = len(set(r['target_video_id']
                           for r in selected_responses)) + len(st.session_state.rejected_videos)
        excluded_videos = len(st.session_state.rejected_videos)
        total_comments = len(
            st.session_state.current_comments) if 'current_comments' in st.session_state else len(selected_responses)
        stop_comments = len([c for c in st.session_state.excluded_comments if c['text'].strip(
        ).lower() == config['automarket']['exclusion_keyword'].strip().lower()])
        excluded_comments = len(st.session_state.excluded_comments)
        # Note : semble être une erreur dans ton code original, devrait être len(responses) - len(selected_responses)
        refused_responses = len(selected_responses) - len(selected_responses)
        posted_responses = len(selected_responses)

        save_campaign_stats(campaign_timestamp, {
            'total_videos': total_videos,
            'excluded_videos': excluded_videos,
            'total_comments': total_comments,
            'stop_comments': stop_comments,
            'excluded_comments': excluded_comments,
            'refused_responses': refused_responses,
            'posted_responses': posted_responses,
            'moderated_responses': moderated_count
        })

    def run(self, config):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Campagne auto", t("monitor_trends_tab"), "Campagne manuelle", "Product matcher", "Video-Product Promotion"]
        )

        config_params = {
            'comments_per_video': config['automarket']['comments_per_video'],
            'min_subscribers': config['automarket']['min_subscribers'],
            'max_videos_per_keyword': config['automarket']['max_videos_per_keyword'],
            'expiry_days': config['automarket']['expiry_days'],
            'view_threshold': config['automarket']['view_threshold'],
            'trusted_channel_videos': config['automarket']['trusted_channel_videos'],
            'response_prompt': config['automarket']['response_prompt'],
            'exclusion_keyword': config['automarket']['exclusion_keyword'],
            'llm': config.get('llm', {})
        }

        with tab1:
            from widgets.auto_campaign import AutoCampaignWidget
            AutoCampaignWidget(
                "auto_campaign",
                "auto_campaign",
                self.plugin_manager,
                self.youtube_api,
                config_params,
                self.post_responses
            ).display()

        with tab2:
            from widgets.watch_yt_trends import YoutubeTrendWatcherWidget
            YoutubeTrendWatcherWidget(
                "trend_watcher",
                "trend_watcher",
                self.plugin_manager,
                self.youtube_api,
                config_params,
                self.post_responses
            ).display()

        with tab3:
            from widgets.market_one_video import MarketOneVideoWidget
            campaign_widget = MarketOneVideoWidget(
                name="market_one_video",
                prefix="market_one_video",
                plugin_manager=self.plugin_manager,
                campaign_keywords=config['marketyoutube']['campaign_keywords'],
                max_campaign_videos=int(config['marketyoutube']['max_campaign_videos']),
                max_campaign_comments=int(config['marketyoutube']['max_campaign_comments']),
                response_prompt=config['marketyoutube']['response_prompt']
            )
            campaign_widget.display()

        with tab4:
            from widgets.product_matcher import VideoProductMatchWidget
            product_matcher_widget = VideoProductMatchWidget(
                name="product_matcher",
                prefix="product_matcher",
                plugin_manager=self.plugin_manager,
            )
            product_matcher_widget.display()

        with tab5:
            from widgets.video_product_promotion import VideoProductPromotionWidget
            video_product_promotion_widget = VideoProductPromotionWidget(
                name="video_product_promotion",
                prefix="video_product_promotion",
                plugin_manager=self.plugin_manager,
            )
            video_product_promotion_widget.display()
