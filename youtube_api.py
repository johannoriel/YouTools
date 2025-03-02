from typing import List, Dict, Any, Optional
from plugins.common import get_credentials
from googleapiclient.discovery import build
from datetime import datetime
import pytz
from langdetect import detect
from googleapiclient.errors import HttpError
import json

# Liste des statistiques avancées (peut être modifiée sans restructurer le reste)
ADVANCED_STATS = [
    "averageViewDuration",
    "averageViewPercentage", "comments", "dislikes", "estimatedMinutesWatched",
    "estimatedAdRevenue", "likes", "shares", "subscribersGained", "subscribersLost",
    "views"
]


class YoutubeAPI:
    def __init__(self, config):
        credentials = get_credentials()
        self.youtube = build('youtube', 'v3', credentials=credentials)
        self.analytics = build('youtubeAnalytics', 'v2',
                               credentials=credentials)
        self.channel_id = config['common']['channel_id']

    def get_advanced_stats_list(self):
        """Retourne la liste des statistiques avancées."""
        return ADVANCED_STATS

    def format_count(self, count: int) -> str:
        """
        Formate un nombre en format K/M si > 1000
        """
        if count >= 1_000_000:
            return f"{count/1_000_000:.1f}M"
        elif count >= 1000:
            return f"{count/1000:.1f}K"
        return str(count)

    import math

    def calculate_relevance_score(self, video_data: dict) -> float:
        """
        Calcule un score de pertinence basé sur :
        - l'ancienneté de la vidéo (pénalisation exponentielle après 7 jours)
        - le nombre d'abonnés (plus il y en a, mieux c'est)
        - le nombre de commentaires (moins il y en a, mieux c'est)
        """
        # Convertir la date de publication en datetime
        published_date = datetime.strptime(
            video_data['published_at'], "%Y-%m-%dT%H:%M:%SZ")
        now = datetime.now(pytz.UTC)
        age_in_days = (now - published_date.replace(tzinfo=pytz.UTC)).days

        # Facteurs de pondération
        AGE_WEIGHT = 0.4
        SUBS_WEIGHT = 0.4
        COMMENTS_WEIGHT = 0.2

        # Calcul du score d'âge avec pénalisation exponentielle après 7 jours
        if age_in_days <= 5:
            age_score = 1.0  # Pas de pénalisation pour les vidéos de moins de 7 jours
        else:
            # Pénalisation exponentielle : l'intérêt diminue de moitié tous les 2 jours
            decay_rate = 0.5  # Diminution de 50% tous les 2 jours
            days_over = age_in_days - 5  # Nombre de jours au-delà de 7 jours
            # Décroissance exponentielle
            age_score = decay_rate ** (days_over / 2)

        # Normalisation des autres scores entre 0 et 1
        # Score max à 1M subs
        subs_score = min(1, video_data['subscriber_count'] / 1_000_000)
        # Score max pour < 1000 comments
        comments_score = max(0, 1 - (video_data['comment_count'] / 1000))

        # Calcul du score final
        relevance_score = (
            AGE_WEIGHT * age_score +
            SUBS_WEIGHT * subs_score +
            COMMENTS_WEIGHT * comments_score
        )

        return round(relevance_score * 100)  # Score sur 100

    def get_quota_usage(self, config) -> Dict[str, float]:
        try:
            service_usage = build('serviceusage', 'v1beta1',
                                  credentials=get_credentials())
            # Format correct
            project_id = f"projects/{config['common']['project_number']}"
            request = service_usage.services().consumerQuotaMetrics().list(
                # Spécifie le service YouTube
                parent=f"{project_id}/services/youtube.googleapis.com"
            )
            response = request.execute()

            quota_metrics = response.get('metrics', [])
            if quota_metrics:
                for metric in quota_metrics:
                    # Métrique par défaut pour YouTube Data API
                    if 'youtube.googleapis.com/default' in metric['metric']:
                        print(metric)
                        quota_limit = int(metric.get('consumerQuotaLimits', [{}])[1].get(
                            'quotaBuckets', [{}])[0].get('effectiveLimit', 10000))
                        # metric.get('metricValues', [{}])[0].get('longValue', 0)
                        usage = 0
                        usage_percentage = (
                            usage / quota_limit) * 100 if quota_limit > 0 else 0
                        remaining_percentage = 100 - usage_percentage
                        return {
                            'usage_percentage': round(usage_percentage, 2),
                            'remaining_percentage': round(remaining_percentage, 2),
                            'quota_usage': usage,
                            'quota_limit': quota_limit
                        }
            return {
                'usage_percentage': 0,
                'remaining_percentage': 0,
                'quota_usage': 0,
                'quota_limit': 0  # Valeur par défaut si pas de données
            }
        except Exception as e:
            print(f"Error fetching quota usage: {str(e)}")
            raise e
            return {
                'usage_percentage': 0,
                'remaining_percentage': 0,
                'quota_usage': 0,
                'quota_limit': 0
            }

    def post(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.channel_id:
                print("YouTube Channel ID is missing in configuration.")
                return None

            body = {
                'snippet': {
                    'channelId': self.channel_id,
                    'description': content,
                    'type': 'bulletin'  # Type pour les posts communautaires
                }
            }

            response = self.youtube.activities().insert(
                part='snippet',
                body=body
            ).execute()

            return response
        except Exception as e:
            print(f"YouTube Post: {str(e)}")
            return None

    def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations d'une chaîne YouTube, y compris le nombre d'abonnés.
        :param channel_id: ID de la chaîne
        :return: Dictionnaire contenant les informations de la chaîne
        """
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics",
                id=channel_id
            )
            response = request.execute()

            if response['items']:
                channel_info = response['items'][0]
                return {
                    'title': channel_info['snippet']['title'],
                    'subscriber_count': int(channel_info['statistics']['subscriberCount']),
                    'view_count': int(channel_info['statistics']['viewCount']),
                    'video_count': int(channel_info['statistics']['videoCount'])
                }
            else:
                return None
        except Exception as e:
            print(f"YouTube API Error (get_channel_info): {str(e)}")
            return None

    # social_api.py (modification de la fonction get_channel_videos)

    # social_api.py (modification de la fonction get_channel_videos)

    def get_channel_videos(self, channel_id: str) -> list:
        """
        Récupère toutes les vidéos d'une chaîne YouTube.
        :param channel_id: ID de la chaîne YouTube
        :return: Liste des vidéos
        """
        try:
            # Récupération des informations sur la chaîne
            channel_response = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()

            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            videos = []
            next_page_token = None

            while True:
                # Récupération des vidéos avec gestion de la pagination
                playlist_response = self.youtube.playlistItems().list(
                    part='snippet,status',
                    playlistId=uploads_playlist_id,
                    maxResults=50,  # Nombre maximal de vidéos par requête
                    pageToken=next_page_token  # Gestion des pages
                ).execute()

                # Récupération des IDs des vidéos pour obtenir leur durée
                video_ids = [item['snippet']['resourceId']['videoId']
                             for item in playlist_response['items']]
                video_details = self.youtube.videos().list(
                    part='contentDetails',
                    id=','.join(video_ids)
                ).execute()

                # Création d'un dictionnaire pour mapper les IDs des vidéos à leur durée
                duration_map = {item['id']: item['contentDetails']
                                ['duration'] for item in video_details['items']}

                for item in playlist_response['items']:
                    video_id = item['snippet']['resourceId']['videoId']
                    duration = duration_map.get(video_id, "N/A")

                    # Déterminer si la vidéo est un Short
                    is_short = self._is_short_video(duration)

                    video = {
                        'video_id': video_id,
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'title': item['snippet']['title'],
                        'thumbnail': item['snippet']['thumbnails']['default']['url'],
                        'description': item['snippet']['description'],
                        'published_at': item['snippet']['publishedAt'],
                        'status': item['status']['privacyStatus'],
                        'duration': duration,
                        'is_short': is_short
                    }
                    videos.append(video)

                # Vérification s'il y a une page suivante
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break

            return videos

        except HttpError as e:
            print(f"Une erreur s'est produite : {e}")
            return []

    def _is_short_video(self, duration: str) -> bool:
        """
        Détermine si une vidéo est un Short en fonction de sa durée.
        :param duration: Durée de la vidéo au format ISO 8601 (ex: PT1M30S)
        :return: True si la vidéo est un Short, False sinon
        """
        # Convertir la durée en secondes
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if not match:
            return False

        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0

        total_seconds = hours * 3600 + minutes * 60 + seconds

        # Une vidéo est considérée comme un Short si elle dure moins de 60 secondes
        return total_seconds <= 60

    def get_video_details(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les détails d'une vidéo, y compris le nombre de vues et de commentaires.
        :param video_id: ID de la vidéo
        :return: Dictionnaire contenant les détails de la vidéo
        """
        try:
            request = self.youtube.videos().list(
                part="statistics",
                id=video_id
            )
            response = request.execute()

            if response['items']:
                video_info = response['items'][0]
                return {
                    'view_count': int(video_info['statistics'].get('viewCount', 0)),
                    'comment_count': int(video_info['statistics'].get('commentCount', 0)),
                    'like_count': int(video_info['statistics'].get('likeCount', 0)),
                }
            else:
                return None
        except Exception as e:
            print(f"YouTube API Error (get_video_details): {str(e)}")
            return None

    def get_comments(self, video_id: str, max_results: int = 2, order: str = "relevance") -> List[Dict[str, Any]]:
        """
        Récupère les derniers commentaires d'une vidéo.
        :param video_id: ID de la vidéo
        :param max_results: Nombre maximum de commentaires à récupérer
        :param order: Ordre des commentaires ("relevance" ou "time")
        :return: Liste des commentaires ou liste vide si les commentaires sont désactivés
        """
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                textFormat="plainText",
                order=order  # Utiliser l'ordre spécifié
            )
            response = request.execute()

            comments = []
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'id': item['id'],
                    'text': comment['textDisplay'],
                    'author': comment['authorDisplayName'],
                    'published_at': comment['publishedAt'],
                    'video_id': video_id,
                    'video_title': "N/A"  # Peut être ajouté plus tard si nécessaire
                })

            return comments
        except HttpError as e:
            if e.resp.status == 403 and 'commentsDisabled' in str(e):
                print(
                    f"YouTube API Warning (get_comments): Comments are disabled for video {video_id}")
                return []  # Retourne une liste vide si les commentaires sont désactivés
            else:
                print(f"YouTube API Error (get_comments): {str(e)}")
                return []  # Retourne une liste vide pour les autres erreurs aussi
        except Exception as e:
            print(f"YouTube API Error (get_comments): {str(e)}")
            return []

    def post_comment_reply(self, comment_id: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Poste une réponse à un commentaire.
        """
        try:
            request = self.youtube.comments().insert(
                part="snippet",
                body={
                    "snippet": {
                        "parentId": comment_id,
                        "textOriginal": text
                    }
                }
            )
            response = request.execute()
            return response
        except Exception as e:
            print(f"YouTube API Error (post_comment_reply): {str(e)}")
            return None

    def get_subscriptions(self, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves the user's YouTube channel subscriptions.

        Args:
            max_results: Maximum number of subscriptions to retrieve

        Returns:
            List of subscription channel information
        """
        try:
            subscriptions = []
            next_page_token = None

            while True:
                request = self.youtube.subscriptions().list(
                    part="snippet",
                    mine=True,
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()

                for item in response['items']:
                    channel_id = item['snippet']['resourceId']['channelId']
                    channel_info = self.get_channel_info(channel_id)

                    if channel_info:
                        subscriptions.append({
                            'channel_id': channel_id,
                            'title': item['snippet']['title'],
                            'subscriber_count': channel_info.get('subscriber_count', 0)
                        })

                next_page_token = response.get('nextPageToken')
                if not next_page_token or len(subscriptions) >= max_results:
                    break

            return subscriptions[:max_results]
        except Exception as e:
            print(f"Error fetching subscriptions: {str(e)}")
            return []

    def get_video_infos(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise les informations d'une vidéo pour s'assurer que tous les champs nécessaires sont présents.
        :param video_data: Données brutes de la vidéo
        :return: Dictionnaire normalisé des informations de la vidéo
        """
        # Récupérer les informations de base de la vidéo
        video_id = video_data.get('id') or video_data.get('video_id')
        if not video_id:
            raise ValueError("Video ID is missing in video data")

        # Récupérer les informations supplémentaires si nécessaire
        if 'subscriber_count' not in video_data:
            channel_info = self.get_channel_info(video_data.get('channel_id'))
            video_data['subscriber_count'] = channel_info['subscriber_count'] if channel_info else 0

        if 'view_count' not in video_data:
            video_details = self.get_video_details(video_id)
            video_data['view_count'] = video_details['view_count'] if video_details else 0
            video_data['comment_count'] = video_details['comment_count'] if video_details else 0
            video_data['like_count'] = video_details['like_count'] if video_details else 0

        if 'published_at' in video_data:
            published_date = datetime.strptime(
                video_data['published_at'], "%Y-%m-%dT%H:%M:%SZ")
            now = datetime.now(pytz.UTC)
            days_old = (now - published_date.replace(tzinfo=pytz.UTC)).days
        else:
            days_old = 0  # Valeur par défaut si la date de publication est manquante

        if 'language' not in video_data:
            title = video_data['title']
            description = video_data['description']
            try:
                video_language = detect(title + " " + description)
            except:
                video_language = 'unfound'

        # Assurer que tous les champs nécessaires sont présents
        normalized_video = {
            'video_id': video_id,
            'title': video_data.get('title', 'N/A'),
            'channel_title': video_data.get('channel_title', 'N/A'),
            'channel_id': video_data.get('channel_id', 'N/A'),
            'subscriber_count': video_data.get('subscriber_count', 0),
            'view_count': video_data.get('view_count', 0),
            'like_count': video_data.get('like_count', 0),
            'comment_count': video_data.get('comment_count', 0),
            'published_at': video_data.get('published_at', 'N/A'),
            'days_old': days_old,
            'url': f"https://www.youtube.com/watch?v={video_id}",
            'language': video_data.get('language', video_language),
            'relevance_score': video_data.get('relevance_score', 0)
        }

        return normalized_video

    def search_videos(self, query: str, max_results: int = 5, order: str = "date", language: str = "fr") -> List[Dict[str, Any]]:
        try:
            modified_query = f"{query}"
            api_max_results = min(max_results * 5, 50)

            if language == "fr" and order != "relevance":
                request = self.youtube.search().list(
                    part="snippet",
                    q=modified_query,
                    maxResults=api_max_results,
                    type="video",
                    order=order,
                    location="46.2276,2.2137",
                    locationRadius="1000km"
                )
            else:
                request = self.youtube.search().list(
                    part="snippet",
                    q=modified_query,
                    maxResults=api_max_results,
                    type="video",
                    relevanceLanguage=language,
                    order=order,
                )
            response = request.execute()

            videos = []
            video_ids = [item['id']['videoId'] for item in response['items']]

            # Requête groupée pour les détails des vidéos
            if video_ids:
                video_details_request = self.youtube.videos().list(
                    part="snippet,statistics",
                    # Limité à 50 IDs par appel (max de l'API)
                    id=",".join(video_ids[:50])
                )
                video_details_response = video_details_request.execute()

                # Créer un dictionnaire des détails pour un accès rapide
                video_details_map = {
                    item['id']: item for item in video_details_response['items']}

                for item in response['items']:
                    video_id = item['id']['videoId']
                    details = video_details_map.get(video_id, {})
                    title = item['snippet']['title']
                    description = item['snippet']['description']
                    try:
                        video_language = detect(title + " " + description)
                    except:
                        video_language = 'unknown'
                    channel_title = item['snippet']['channelTitle']
                    channel_id = item['snippet']['channelId']
                    published_at = item['snippet']['publishedAt']

                    # Utiliser les détails groupés si disponibles
                    subscriber_count = 0
                    channel_info = self.get_channel_info(channel_id)
                    if channel_info:
                        subscriber_count = channel_info['subscriber_count']

                    video_data = {
                        'id': video_id,
                        'title': title,
                        'description': description,
                        'channel_title': channel_title,
                        'channel_id': channel_id,
                        'subscriber_count': subscriber_count,
                        'view_count': int(details.get('statistics', {}).get('viewCount', 0)) if details else 0,
                        'like_count': int(details.get('statistics', {}).get('likeCount', 0)) if details else 0,
                        'comment_count': int(details.get('statistics', {}).get('commentCount', 0)) if details else 0,
                        'published_at': published_at,
                        'url': f"https://www.youtube.com/watch?v={video_id}"
                    }

                    normalized_video = self.get_video_infos(video_data)
                    normalized_video['relevance_score'] = self.calculate_relevance_score(
                        normalized_video)
                    videos.append(normalized_video)

            return videos[:max_results]
        except Exception as e:
            print(f"YouTube API Error (search_videos): {str(e)}")
            return []

    def get_trending_videos(self, language: str = "fr", category_id: int = 0, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Récupère les vidéos tendances de YouTube pour une langue et une catégorie spécifique.
        :param language: Code de la langue (ex: 'fr' pour français)
        :param category_id: ID de la catégorie YouTube (0 pour toutes les catégories)
        :param max_results: Nombre de vidéos à récupérer
        :return: Liste des vidéos tendances
        """
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics",
                chart="mostPopular",
                regionCode=language.upper(),
                videoCategoryId=str(category_id),
                maxResults=max_results
            )
            response = request.execute()

            trending_videos = []
            for item in response['items']:
                video = {
                    'id': item['id'],
                    'video_id': item['id'],  # compat
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'channel_title': item['snippet']['channelTitle'],
                    'channel_id': item['snippet']['channelId'],
                    'view_count': int(item['statistics'].get('viewCount', 0)),
                    'comment_count': int(item['statistics'].get('commentCount', 0)),
                    'like_count': int(item['statistics'].get('likeCount', 0)),
                    'published_at': item['snippet']['publishedAt'],
                    'url': f"https://www.youtube.com/watch?v={item['id']}"
                }
                normalized_video = self.get_video_infos(video)
                normalized_video['relevance_score'] = self.calculate_relevance_score(
                    normalized_video)
                trending_videos.append(normalized_video)
            return trending_videos
        except Exception as e:
            print(f"YouTube API Error (get_trending_videos): {str(e)}")
            return []

    def get_channel_recent_videos(self, channel_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Gets the most recent videos from a specific channel with detailed statistics.

        Args:
            channel_id: The YouTube channel ID
            max_results: Maximum number of videos to retrieve

        Returns:
            List of video information including views, likes, comments, etc.
        """
        try:
            # Get channel's uploads playlist ID
            channel_response = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()

            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            # Get videos from uploads playlist
            videos = []
            next_page_token = None

            while len(videos) < max_results:
                playlist_response = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                ).execute()

                video_ids = [item['snippet']['resourceId']['videoId']
                             for item in playlist_response['items']]

                # Get detailed video statistics
                if video_ids:
                    video_response = self.youtube.videos().list(
                        part='statistics,snippet',
                        id=','.join(video_ids)
                    ).execute()

                    for item in video_response['items']:
                        published_at = datetime.strptime(
                            item['snippet']['publishedAt'],
                            "%Y-%m-%dT%H:%M:%SZ"
                        ).replace(tzinfo=pytz.UTC)

                        days_old = (datetime.now(pytz.UTC) - published_at).days

                        # Détecter la langue à partir du titre et de la description
                        title = item['snippet']['title']
                        description = item['snippet']['description']

                        video = {
                            'title': item['snippet']['title'],
                            'video_id': item['id'],
                            'description': description,
                            'channel_title': item['snippet']['channelTitle'],
                            'channel_id': item['snippet']['channelId'],
                            'view_count': int(item['statistics'].get('viewCount', 0)),
                            'like_count': int(item['statistics'].get('likeCount', 0)),
                            'comment_count': int(item['statistics'].get('commentCount', 0)),
                            'days_old': days_old,
                            'published_at': published_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            'url': f"https://www.youtube.com/watch?v={item['id']}",
                        }
                        normalized_video = self.get_video_infos(video)
                        normalized_video['relevance_score'] = self.calculate_relevance_score(
                            normalized_video)
                        videos.append(normalized_video)
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break

            return videos[:max_results]
        except Exception as e:
            print(f"Error fetching channel videos: {str(e)}")
            return []

    def get_advanced_video_stats(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch advanced statistics using Analytics API with dynamic metrics.
        """
        try:
            request = self.youtube.videos().list(
                part="statistics,contentDetails",
                id=video_id
            )
            response = request.execute()

            if not response['items']:
                return None

            video_info = response['items'][0]
            stats = {
                'view_count': int(video_info['statistics'].get('viewCount', 0)),
                'like_count': int(video_info['statistics'].get('likeCount', 0)),
                'comment_count': int(video_info['statistics'].get('commentCount', 0)),
                'duration': video_info['contentDetails']['duration']
            }

            metrics = ",".join(ADVANCED_STATS)
            analytics_response = self.analytics.reports().query(
                ids=f"channel=={self.channel_id}",
                startDate="2014-01-01",
                endDate=datetime.now().strftime("%Y-%m-%d"),
                metrics=metrics,
                dimensions="video",
                filters=f"video=={video_id}"
            ).execute()

            if analytics_response.get('rows'):
                row = analytics_response['rows'][0]
                advanced_stats = {}
                for i, metric in enumerate(ADVANCED_STATS):
                    value = row[i + 1]  # +1 car row[0] est l'ID vidéo
                    advanced_stats[metric] = float(value) if isinstance(
                        value, (int, float)) and '.' in str(value) else int(value) if value else 0
                stats['advanced_stats'] = advanced_stats
            else:
                stats['advanced_stats'] = {
                    metric: 0 for metric in ADVANCED_STATS}

            total_seconds = self._iso_duration_to_seconds(
                video_info['contentDetails']['duration'])
            stats['retention_rate'] = (
                stats['advanced_stats']['averageViewDuration'] / total_seconds * 100) if total_seconds > 0 else 0.0

            return stats
        except HttpError as e:
            print(
                f"YouTube Analytics API Error (get_advanced_video_stats): {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected Error (get_advanced_video_stats): {str(e)}")
            return None

    def debug_advanced_video_stats(self, video_id: str, selected_metrics: List[str]) -> Dict[str, Any]:
        """
        Debug method with selectable metrics from ADVANCED_STATS.
        """
        try:
            metrics_str = ",".join(selected_metrics)
            analytics_response = self.analytics.reports().query(
                ids=f"channel=={self.channel_id}",
                startDate="2014-01-01",
                endDate=datetime.now().strftime("%Y-%m-%d"),
                metrics=metrics_str,
                dimensions="video",
                filters=f"video=={video_id}"
            ).execute()

            request = self.youtube.videos().list(
                part="statistics,contentDetails",
                id=video_id
            )
            response = request.execute()

            result = {
                "analytics_response": analytics_response,
                "basic_stats": response if response['items'] else None,
                "error": None
            }

            if response['items'] and 'averageViewDuration' in selected_metrics and analytics_response.get('rows'):
                duration = response['items'][0]['contentDetails']['duration']
                total_seconds = self._iso_duration_to_seconds(duration)
                avg_view_duration_idx = selected_metrics.index(
                    'averageViewDuration') + 1
                avg_view_duration = float(
                    analytics_response['rows'][0][avg_view_duration_idx])
                result['calculated_retention_rate'] = (
                    avg_view_duration / total_seconds * 100) if total_seconds > 0 else 0.0

            return result
        except HttpError as e:
            print(
                f"YouTube Analytics API Error (debug_advanced_video_stats): {str(e)}")
            return {"analytics_response": None, "basic_stats": None, "error": str(e)}
        except Exception as e:
            print(f"Unexpected Error (debug_advanced_video_stats): {str(e)}")
            return {"analytics_response": None, "basic_stats": None, "error": str(e)}

    def _iso_duration_to_seconds(self, duration: str) -> int:
        """Convert ISO 8601 duration to seconds."""
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if not match:
            return 0
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0
        return hours * 3600 + minutes * 60 + seconds
