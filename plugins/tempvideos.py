from lib.global_vars import translations, t
import streamlit as st
from app import Plugin
from plugins.common import get_credentials
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import RefreshError
import os
import datetime
import re
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
from lib.youtube_db import get_latest_stats

# Ajout des traductions spécifiques
translations["en"].update({
    "temp_videos_tab": "Temporary Videos",
    "temp_videos_header": "Managing Temporary Videos",
    "temp_videos_published_on": "Published on:",
    "temp_videos_expired_days": "Expired for {days} days",
    "temp_videos_expires_in_days": "Expires in {days} days",
    "temp_videos_status": "Status:",
    "temp_videos_no_temp_videos": "No temporary videos found.",
    "temp_videos_unpublish_button": "Unpublish Expired Videos",
    "temp_videos_unpublish_success": "{count} videos have been unpublished.",
    "temp_videos_configure_channel_id": "Please configure the channel ID in the Configuration tab.",
    "manual_unpublish_tab": "Manual Unpublish",
    "manual_unpublish_header": "Manual Video Unpublishing",
    "manual_unpublish_button": "Unpublish Selected Videos",
    "manual_unpublish_success": "{count} selected videos have been unpublished.",
    "manual_unpublish_no_selection": "Please select at least one video to unpublish.",
    "marketyoutube_title": "Title",
    "marketyoutube_url": "URL",
    "marketyoutube_published": "Published",
    "marketyoutube_status": "Status",
    "marketyoutube_views": "Views",
    "marketyoutube_likes": "Likes",
    "marketyoutube_shares": "Shares",
    "marketyoutube_subscribers_gained": "Subscribers Gained",
    "views_at_28_days": "Views at 28 Days",
    "views_at_3_months": "Views at 3 Months",
    "views_at_1_year": "Views at 1 Year",
})

translations["fr"].update({
    "temp_videos_tab": "Vidéos temporaires",
    "temp_videos_header": "Gestion des vidéos temporaires",
    "temp_videos_published_on": "Publié le :",
    "temp_videos_expired_days": "Expiré depuis {days} jours",
    "temp_videos_expires_in_days": "Expire dans {days} jours",
    "temp_videos_status": "Statut :",
    "temp_videos_no_temp_videos": "Aucune vidéo temporaire trouvée.",
    "temp_videos_unpublish_button": "Dépublier les vidéos en dépassement",
    "temp_videos_unpublish_success": "{count} vidéos ont été dépubliées.",
    "temp_videos_configure_channel_id": "Veuillez configurer l'ID de la chaîne dans l'onglet Configuration.",
    "manual_unpublish_tab": "Dépublication manuelle",
    "manual_unpublish_header": "Dépublication manuelle des vidéos",
    "manual_unpublish_button": "Dépublier les vidéos sélectionnées",
    "manual_unpublish_success": "{count} vidéos sélectionnées ont été dépubliées.",
    "manual_unpublish_no_selection": "Veuillez sélectionner au moins une vidéo à dépublier.",
    "marketyoutube_title": "Titre",
    "marketyoutube_url": "URL",
    "marketyoutube_published": "Publié",
    "marketyoutube_status": "Statut",
    "marketyoutube_views": "Vues",
    "marketyoutube_likes": "Likes",
    "marketyoutube_shares": "Partages",
    "marketyoutube_subscribers_gained": "Abonnés gagnés",
    "views_at_28_days": "Vues à 28 jours",
    "views_at_3_months": "Vues à 3 mois",
    "views_at_1_year": "Vues à 1 an"
})

class TempvideosPlugin(Plugin):

    def get_tabs(self):
        return [
            {"name": t("temp_videos_tab"), "plugin": "tempvideos"},
            {"name": t("manual_unpublish_tab"), "plugin": "tempvideos"}
        ]

    def list_videos(self, youtube, channel_id):
        request = youtube.channels().list(part='contentDetails,statistics', id=channel_id)
        response = request.execute()
        playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        videos = []
        next_page_token = None
        while True:
            request = youtube.playlistItems().list(
                part='snippet,status,contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            videos += response['items']
            next_page_token = response.get('nextPageToken')
            if next_page_token is None:
                break
        return videos

    def get_video_stats(self, video_id):
        latest_stats = get_latest_stats(video_id)
        if latest_stats:
            advanced_stats = latest_stats.get('advanced_stats', {})
            return {
                'view_count': latest_stats.get('view_count', 0),
                'likes': advanced_stats.get('likes', 0),
                'share_count': advanced_stats.get('shares', 0),
                'subscribers_gained': advanced_stats.get('subscribersGained', 0),
                'views_at_28_days': latest_stats.get('views_at_28_days', 0),
                'views_at_3_months': latest_stats.get('views_at_3_months', 0),
                'views_at_1_year': latest_stats.get('views_at_1_year', 0)
            }
        return {
            'view_count': 0,
            'likes': 0,
            'share_count': 0,
            'subscribers_gained': 0,
            'views_at_28_days': 0,
            'views_at_3_months': 0,
            'views_at_1_year': 0
        }

    def update_video_privacy(self, youtube, video_id, privacy_status='unlisted'):
        request_body = {
            'id': video_id,
            'status': {
                'privacyStatus': privacy_status
            }
        }
        request = youtube.videos().update(
            part='status',
            body=request_body
        )
        response = request.execute()
        return response

    def check_video_expiration(self, video):
        title = video['snippet']['title']
        published_at = datetime.datetime.strptime(video['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        match = re.match(r'\[(\d+)j\]', title)

        if match:
            days = int(match.group(1))
            delta_days = (datetime.datetime.utcnow() - published_at).days
            days_left = days - delta_days
            return {
                'title': title,
                'video_id': video['snippet']['resourceId']['videoId'],
                'published_at': published_at,
                'expiration_days': days,
                'days_left': days_left,
                'is_expired': days_left <= 0,
                'privacy_status': video['status']['privacyStatus']
            }
        return None

    def display_manual_unpublish(self, youtube, channel_id):
        st.header(t("manual_unpublish_header"))

        videos = self.list_videos(youtube, channel_id)
        video_data = []

        for video in videos:
            video_id = video['snippet']['resourceId']['videoId']
            stats = self.get_video_stats(video_id)
            video_data.append({
                'title': video['snippet']['title'],
                'video_id': video_id,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'published_at': datetime.datetime.strptime(video['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d'),
                'status': video['status']['privacyStatus'],
                'views': stats['view_count'],
                'likes': stats['likes'],
                'shares': stats['share_count'],
                'subscribers_gained': stats['subscribers_gained'],
                'views_at_28_days': stats['views_at_28_days'],
                'views_at_3_months': stats['views_at_3_months'],
                'views_at_1_year': stats['views_at_1_year']
            })

        if not video_data:
            st.info(t("temp_videos_no_temp_videos"))
            return

        df = pd.DataFrame(video_data)

        # JavaScript pour rendre les liens cliquables
        link_renderer = JsCode("""
            class LinkRenderer {
                init(params) {
                    this.eGui = document.createElement('a');
                    this.eGui.href = params.data.url;
                    this.eGui.target = '_blank';
                    this.eGui.innerText = params.value;
                }
                getGui() {
                    return this.eGui;
                }
            }
        """)

        # Configurer AgGrid
        grid_options = {
            "rowSelection": "multiple",
            "rowHeight": 80,
            "columnDefs": [
                {
                    "field": "title",
                    "checkboxSelection": True,
                    "headerCheckboxSelection": True,                    "headerName": t("marketyoutube_title"),
                    "cellRenderer": link_renderer,
                    "flex": 2
                },
                {
                    "field": "url",
                    "headerName": t("marketyoutube_url"),
                    "hide": True
                },
                {
                    "field": "published_at",
                    "headerName": t("marketyoutube_published"),
                    "width": 120
                },
                {
                    "field": "status",
                    "headerName": t("marketyoutube_status"),
                    "width": 100,
                    "filter": "agSetColumnFilter",
                    "filterParams": {
                        "values": ["public", "private", "unlisted"],
                        "suppressSelectAll": True,
                        "closeOnApply": True
                    }
                },
                {
                    "field": "views",
                    "headerName": t("marketyoutube_views"),
                    "width": 100
                },
                {
                    "field": "likes",
                    "headerName": t("marketyoutube_likes"),
                    "width": 100
                },
                {
                    "field": "shares",
                    "headerName": t("marketyoutube_shares"),
                    "width": 100
                },
                {
                    "field": "subscribers_gained",
                    "headerName": t("marketyoutube_subscribers_gained"),
                    "width": 120
                },
                {
                    "field": "views_at_28_days",
                    "headerName": t("views_at_28_days"),
                    "width": 120
                },
                {
                    "field": "views_at_3_months",
                    "headerName": t("views_at_3_months"),
                    "width": 120
                },
                {
                    "field": "views_at_1_year",
                    "headerName": t("views_at_1_year"),
                    "width": 120
                }
            ],
            "defaultColDef": {
                "flex": 1,
                "sortable": True,
                "filter": True,
                "resizable": True
            }
        }

        # Afficher AgGrid
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            height=400,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            key="manual_unpublish_grid"
        )

        selected_rows = grid_response.get('selected_rows', [])

        if st.button(t("manual_unpublish_button")):
            if not selected_rows:
                st.warning(t("manual_unpublish_no_selection"))
            else:
                progress_bar = st.progress(0)
                total_videos = len(selected_rows)
                for i, row in enumerate(selected_rows):
                    self.update_video_privacy(youtube, row['video_id'])
                    progress_bar.progress((i + 1) / total_videos)
                progress_bar.empty()
                st.success(t("manual_unpublish_success").format(count=len(selected_rows)))
                st.rerun()

    def run(self, config):
        tab1, tab2 = st.tabs([
            t("temp_videos_tab"),
            t("manual_unpublish_tab")
        ])

        channel_id = config['common'].get('channel_id')
        if not channel_id:
            st.warning(t("temp_videos_configure_channel_id"))
            return

        credentials = get_credentials()
        youtube = build('youtube', 'v3', credentials=credentials)

        with tab1:
            st.header(t("temp_videos_header"))
            videos = self.list_videos(youtube, channel_id)
            temp_videos = [self.check_video_expiration(video) for video in videos if self.check_video_expiration(video)]

            if temp_videos:
                for video in temp_videos:
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(video['title'])
                    with col2:
                        st.write(f"{t('temp_videos_published_on')} {video['published_at'].strftime('%Y-%m-%d')}")
                    with col3:
                        if video['is_expired']:
                            st.write(t('temp_videos_expired_days').format(days=abs(video['days_left'])))
                        else:
                            st.write(t('temp_videos_expires_in_days').format(days=video['days_left']))
                    with col4:
                        st.write(f"{t('temp_videos_status')} {video['privacy_status']}")

                video_list = "\n".join([f"- [{video['title']}](https://www.youtube.com/watch?v={video['video_id']})" for video in temp_videos])
                st.markdown("### Liste des vidéos avec leur lien (Markdown) :")
                st.code(video_list, language="markdown")

                video_list_html = "<ul>" + "".join([f"<li><a href='https://www.youtube.com/watch?v={video['video_id']}' target='_blank'>{video['title']}</a></li>" for video in temp_videos]) + "</ul>"
                st.markdown("### Liste des vidéos avec leur lien (HTML) :")
                st.markdown(video_list_html, unsafe_allow_html=True)

                if st.button(t("temp_videos_unpublish_button")):
                    expired_videos = [video for video in temp_videos if video['is_expired'] and video['privacy_status'] == 'public']
                    for video in expired_videos:
                        self.update_video_privacy(youtube, video['video_id'])
                    st.success(t("temp_videos_unpublish_success").format(count=len(expired_videos)))
                    st.rerun()
            else:
                st.info(t("temp_videos_no_temp_videos"))

        with tab2:
            self.display_manual_unpublish(youtube, channel_id)
