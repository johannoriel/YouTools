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
from lib.youtube_db import get_latest_stats, get_videos

# Traductions (inchangées, sauf ajout d'une nouvelle clé pour le filtre)
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
    "no_videos_selected": "No videos selected.",
    "no_valid_videos_selected": "No valid videos selected.",
    "select_new_status": "Select the new status for the selected videos:",
    "status_public": "Public",
    "status_private": "Private",
    "status_unlisted": "Unlisted",
    "change_status_button": "Change the status of selected videos",
    "status_update_success": "{count} videos have been updated to the '{status}' status.",
    "unknown_title": "Unknown Title",
    "selected_videos_list": "List of selected videos with their links (Markdown):",
    "exclude_temp_videos": "Exclude temporary videos ([n days] format)",
    "select_playlist": "Select a playlist",
    "create_new_playlist": "Create a new playlist",
    "new_playlist_name": "New playlist name",
    "create_playlist_button": "Create Playlist",
    "add_to_playlist_button": "Add selected videos to playlist",
    "add_to_playlist_success": "{count} videos added to playlist '{playlist_title}'.",
    "no_playlist_selected": "Please select a playlist.",
    "playlist_created_success": "Playlist '{title}' created successfully.",
    "filter_by_playlist": "Filter by playlist",
    "include_exclude_playlist": "Include or exclude videos from selected playlist",
    "include_videos_in_playlist": "Include videos in playlist",
    "exclude_videos_in_playlist": "Exclude videos in playlist",

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
    "views_at_1_year": "Vues à 1 an",
    "no_videos_selected": "Aucune vidéo sélectionnée.",
    "no_valid_videos_selected": "Aucune vidéo sélectionnée valide.",
    "select_new_status": "Sélectionner le nouveau statut pour les vidéos sélectionnées :",
    "status_public": "Public",
    "status_private": "Privé",
    "status_unlisted": "Non répertorié",
    "change_status_button": "Changer le statut des vidéos sélectionnées",
    "status_update_success": "{count} vidéos ont été mises à jour vers le statut '{status}'.",
    "unknown_title": "Titre inconnu",
    "selected_videos_list": "Liste des vidéos sélectionnées avec leur lien (Markdown) :",
    "exclude_temp_videos": "Exclure les vidéos temporaires (format [n jours])",
    "select_playlist": "Sélectionner une playlist",
    "create_new_playlist": "Créer une nouvelle playlist",
    "new_playlist_name": "Nom de la nouvelle playlist",
    "create_playlist_button": "Créer la playlist",
    "add_to_playlist_button": "Ajouter les vidéos sélectionnées à la playlist",
    "add_to_playlist_success": "{count} vidéos ajoutées à la playlist '{playlist_title}'.",
    "no_playlist_selected": "Veuillez sélectionner une playlist.",
    "playlist_created_success": "Playlist '{title}' créée avec succès.",
    "filter_by_playlist": "Filtrer par playlist",
    "include_exclude_playlist": "Inclure ou exclure les vidéos de la playlist sélectionnée",
    "include_videos_in_playlist": "Inclure les vidéos dans la playlist",
    "exclude_videos_in_playlist": "Exclure les vidéos dans la playlist",
})

class TempvideosPlugin(Plugin):

    def get_tabs(self):
        return [
            {"name": t("temp_videos_tab"), "plugin": "tempvideos"},
            {"name": t("manual_unpublish_tab"), "plugin": "tempvideos"}
        ]

    def list_playlists(self, youtube, channel_id):
        playlists = []
        next_page_token = None
        while True:
            request = youtube.playlists().list(
                part='snippet',
                channelId=channel_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            playlists += response['items']
            next_page_token = response.get('nextPageToken')
            if next_page_token is None:
                break
        return playlists

    def create_playlist(self, youtube, title):
        request_body = {
            'snippet': {
                'title': title,
                'description': 'Playlist created via TempvideosPlugin'
            },
            'status': {
                'privacyStatus': 'public'
            }
        }
        request = youtube.playlists().insert(
            part='snippet,status',
            body=request_body
        )
        response = request.execute()
        return response['id']

    def add_videos_to_playlist(self, youtube, playlist_id, video_ids):
        for video_id in video_ids:
            request_body = {
                'snippet': {
                    'playlistId': playlist_id,
                    'resourceId': {
                        'kind': 'youtube#video',
                        'videoId': video_id
                    }
                }
            }
            youtube.playlistItems().insert(
                part='snippet',
                body=request_body
            ).execute()

    def get_playlist_videos(self, youtube, playlist_id):
        video_ids = []
        next_page_token = None
        while True:
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            video_ids += [item['contentDetails']['videoId'] for item in response['items']]
            next_page_token = response.get('nextPageToken')
            if next_page_token is None:
                break
        return video_ids

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
                'views_at_28_days': advanced_stats.get('views_at_28_days', 0),
                'views_at_3_months': advanced_stats.get('views_at_3_months', 0),
                'views_at_1_year': advanced_stats.get('views_at_1_year', 0)
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

    def update_video_privacy(self, youtube, video_id, video_title, privacy_status='unlisted'):
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
        try:
            response = request.execute()
        except Exception as e:
            st.write(f"Error updating video - probably a membership video : {video_id} {video_title} to {privacy_status} : {e}")
            return None

        return response

    def check_video_expiration(self, video):
        title = video['snippet']['title']
        published_at = datetime.datetime.strptime(video['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
        match = re.match(r'\[(\d+)j\]', title)
        privacy_status = video['status']['privacyStatus']

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
                'privacy_status': privacy_status,
                'is_temp_video': True
            }
        return {
            'title': title,
            'video_id': video['snippet']['resourceId']['videoId'],
            'published_at': published_at,
            'privacy_status': privacy_status,
            'is_temp_video': False
        }

    def display_manual_unpublish(self, youtube, channel_id):
        st.header(t("manual_unpublish_header"))

        # Sélection de playlist pour filtrer les vidéos
        st.subheader(t("filter_by_playlist"))
        playlists = self.list_playlists(youtube, channel_id)
        playlist_options = {"": ""} | {playlist['snippet']['title']: playlist['id'] for playlist in playlists}
        filter_playlist = st.selectbox(
            t("filter_by_playlist"),
            options=list(playlist_options.keys()),
            key="filter_playlist_select"
        )
        include_exclude = st.radio(
            t("include_exclude_playlist"),
            options=["include", "exclude"],
            format_func=lambda x: t("include_videos_in_playlist") if x == "include" else t("exclude_videos_in_playlist"),
            key="include_exclude_radio"
        )

        # Case à cocher pour exclure les vidéos temporaires
        exclude_temp_videos = st.checkbox(t("exclude_temp_videos"), value=False)

        videos = get_videos()
        video_data = []

        # Récupérer les vidéos de la playlist sélectionnée si applicable
        playlist_video_ids = []
        if filter_playlist:
            playlist_video_ids = self.get_playlist_videos(youtube, playlist_options[filter_playlist])

        for video in videos:
            video_id = video['video_id']
            temp_check = self.check_video_expiration({
                'snippet': {
                    'title': video['title'],
                    'publishedAt': video['published_at'],
                    'resourceId': {'videoId': video_id}
                },
                'status': {'privacyStatus': video['status']}
            })

            # Filtrer selon l'appartenance à la playlist
            in_playlist = video_id in playlist_video_ids
            if filter_playlist:
                if include_exclude == "include" and not in_playlist:
                    continue
                if include_exclude == "exclude" and in_playlist:
                    continue

            if exclude_temp_videos and temp_check['is_temp_video']:
                continue

            stats = self.get_video_stats(video_id)
            video_data.append({
                'title': video['title'],
                'video_id': video_id,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'published_at': datetime.datetime.strptime(video['published_at'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d'),
                'status': video['status'],
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

        grid_options = {
            "rowSelection": "multiple",
            "rowHeight": 80,
            "columnDefs": [
                {
                    "field": "title",
                    "checkboxSelection": True,
                    "headerCheckboxSelection": True,
                    "headerName": t("marketyoutube_title"),
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

        if not selected_rows is None and not selected_rows.empty:
            video_list = []
            for _, row in selected_rows.iterrows():
                title = row['title'] if pd.notna(row['title']) else t("unknown_title")
                url = row['url'] if pd.notna(row['url']) else ''
                video_list.append(f"- [{title}]({url})")
            st.markdown(t("selected_videos_list"))
            st.code("\n".join(video_list), language="markdown")
        else:
            st.info(t("no_videos_selected"))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(t("select_new_status"))
            new_status = st.selectbox(
                t("select_new_status"),
                options=["public", "private", "unlisted"],
                format_func=lambda x: {"public": t("status_public"), "private": t("status_private"), "unlisted": t("status_unlisted")}[x],
                key="status_select"
            )
            if st.button(t("change_status_button")):
                if selected_rows is None or selected_rows.empty:
                    st.warning(t("manual_unpublish_no_selection"))
                else:
                    progress_bar = st.progress(0)
                    total_videos = len(selected_rows)
                    for i, (_, row) in enumerate(selected_rows.iterrows()):
                        video_id = row['video_id'] if pd.notna(row['video_id']) else None
                        if video_id:
                            self.update_video_privacy(youtube, video_id, row['title'], privacy_status=new_status)
                        progress_bar.progress((i + 1) / total_videos)
                    progress_bar.empty()
                    st.success(t("status_update_success").format(count=len(selected_rows), status=new_status))
                    st.rerun()

        with col2:
            st.subheader(t("select_playlist"))
            playlists = self.list_playlists(youtube, channel_id)
            playlist_options = {playlist['snippet']['title']: playlist['id'] for playlist in playlists}
            selected_playlist = st.selectbox(
                t("select_playlist"),
                options=[""] + list(playlist_options.keys()),
                key="playlist_select"
            )

            st.subheader(t("create_new_playlist"))
            new_playlist_name = st.text_input(t("new_playlist_name"))
            if st.button(t("create_playlist_button")):
                if new_playlist_name:
                    new_playlist_id = self.create_playlist(youtube, new_playlist_name)
                    st.success(t("playlist_created_success").format(title=new_playlist_name))
                    st.rerun()

            if st.button(t("add_to_playlist_button")):
                if selected_rows is None or selected_rows.empty:
                    st.warning(t("manual_unpublish_no_selection"))
                elif not selected_playlist:
                    st.warning(t("no_playlist_selected"))
                else:
                    video_ids = [row['video_id'] for _, row in selected_rows.iterrows() if pd.notna(row['video_id'])]
                    if video_ids:
                        self.add_videos_to_playlist(youtube, playlist_options[selected_playlist], video_ids)
                        st.success(t("add_to_playlist_success").format(count=len(video_ids), playlist_title=selected_playlist))
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

        # Charger les vidéos tagguées pour tab1 une seule fois et les stocker dans session_state
        if 'tagged_videos' not in st.session_state:
            st.session_state.tagged_videos = self.list_videos(youtube, channel_id)

        with tab1:
            st.header(t("temp_videos_header"))
            temp_videos = [self.check_video_expiration(video) for video in st.session_state.tagged_videos if self.check_video_expiration(video).get('is_temp_video')]

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
                        self.update_video_privacy(youtube, video['video_id'], video['title'])
                    st.success(t("temp_videos_unpublish_success").format(count=len(expired_videos)))
                    # Réinitialiser la liste après dépublication
                    del st.session_state.tagged_videos
                    st.rerun()
            else:
                st.info(t("temp_videos_no_temp_videos"))

        with tab2:
            self.display_manual_unpublish(youtube, channel_id)
