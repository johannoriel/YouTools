from typing import List, Dict, Any, Optional
from plugins.common import get_credentials
from googleapiclient.discovery import build
from datetime import datetime
import pytz
from langdetect import detect

class YoutubeAPI:
    def __init__(self, config):
        credentials = get_credentials()
        self.youtube = build('youtube', 'v3', credentials=credentials)
        self.channel_id = config['common']['channel_id']

    def format_count(self, count: int) -> str:
        """
        Formate un nombre en format K/M si > 1000
        """
        if count >= 1_000_000:
            return f"{count/1_000_000:.1f}M"
        elif count >= 1000:
            return f"{count/1000:.1f}K"
        return str(count)

    def calculate_relevance_score(self, video_data: dict) -> float:
        """
        Calcule un score de pertinence basé sur :
        - l'ancienneté de la vidéo (plus c'est récent, mieux c'est)
        - le nombre d'abonnés (plus il y en a, mieux c'est)
        - le nombre de commentaires (moins il y en a, mieux c'est)
        """
        # Convertir la date de publication en datetime
        published_date = datetime.strptime(video_data['published_at'], "%Y-%m-%dT%H:%M:%SZ")
        now = datetime.now(pytz.UTC)
        age_in_days = (now - published_date.replace(tzinfo=pytz.UTC)).days

        # Facteurs de pondération
        AGE_WEIGHT = 0.4
        SUBS_WEIGHT = 0.4
        COMMENTS_WEIGHT = 0.2

        # Normalisation des scores entre 0 et 1
        age_score = max(0, 1 - (age_in_days / 30))  # Score max pour < 30 jours
        subs_score = min(1, video_data['subscriber_count'] / 1_000_000)  # Score max à 1M subs
        comments_score = max(0, 1 - (video_data['comment_count'] / 1000))  # Score max pour < 1000 comments

        # Calcul du score final
        relevance_score = (
            AGE_WEIGHT * age_score +
            SUBS_WEIGHT * subs_score +
            COMMENTS_WEIGHT * comments_score
        )

        return round(relevance_score * 100)  # Score sur 100

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
                video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_response['items']]
                video_details = self.youtube.videos().list(
                    part='contentDetails',
                    id=','.join(video_ids)
                ).execute()

                # Création d'un dictionnaire pour mapper les IDs des vidéos à leur durée
                duration_map = {item['id']: item['contentDetails']['duration'] for item in video_details['items']}

                for item in playlist_response['items']:
                    video_id = item['snippet']['resourceId']['videoId']
                    duration = duration_map.get(video_id, "N/A")

                    # Déterminer si la vidéo est un Short
                    is_short = self._is_short_video(duration)

                    video = {
                        'title': item['snippet']['title'],
                        'video_id': video_id,
                        'thumbnail': item['snippet']['thumbnails']['default']['url'],
                        'status': item['status']['privacyStatus'],  # Statut de la vidéo
                        'duration': duration,  # Durée de la vidéo
                        'is_short': is_short  # Indicateur de Short
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
                    'comment_count': int(video_info['statistics'].get('commentCount', 0))
                }
            else:
                return None
        except Exception as e:
            print(f"YouTube API Error (get_video_details): {str(e)}")
            return None

    def search_videos(self, query: str, max_results: int = 5, order: str = "date", language: str = "fr") -> List[Dict[str, Any]]:
        try:
            # Modifier la query pour inclure la langue
            modified_query = f"{query} in {language}"
            modified_query = f"{query}"

            api_max_results = min(max_results * 5, 50)  # Augmenter le nombre de résultats

            if (language == "fr" and order!="relevance"):
                request = self.youtube.search().list(
                            part="snippet",
                            q=modified_query,
                            maxResults=api_max_results,
                            type="video",
                            order=order,
                            location="46.2276,2.2137",  # Coordonnées approximatives du centre de la France
                            locationRadius="1000km"  # Rayon de recherche de 1000 km
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
            for item in response['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                description = item['snippet']['description']
                try:
                    video_language = detect(title + " " + description)
                except:
                    video_language = 'unknown'
                channel_title = item['snippet']['channelTitle']
                channel_id = item['snippet']['channelId']
                published_at = item['snippet']['publishedAt']

                channel_info = self.get_channel_info(channel_id)
                subscriber_count = channel_info['subscriber_count'] if channel_info else 0

                video_details = self.get_video_details(video_id)
                view_count = video_details['view_count'] if video_details else 0
                comment_count = video_details['comment_count'] if video_details else 0

                video_data = {
                    'id': video_id,
                    'title': title,
                    'channel_title': channel_title,
                    'channel_id': channel_id,
                    'subscriber_count': subscriber_count,
                    'view_count': view_count,
                    'comment_count': comment_count,
                    'published_at': published_at,
                    'language': video_language,
                    'url': f"https://www.youtube.com/watch?v={video_id}"
                }

                video_data['relevance_score'] = self.calculate_relevance_score(video_data)
                videos.append(video_data)

            return videos[:max_results]

        except Exception as e:
            print(f"YouTube API Error (search_videos): {str(e)}")
            return []

    def get_comments(self, video_id: str, max_results: int = 2, order: str = "relevance") -> List[Dict[str, Any]]:
        """
        Récupère les derniers commentaires d'une vidéo.
        :param video_id: ID de la vidéo
        :param max_results: Nombre maximum de commentaires à récupérer
        :param order: Ordre des commentaires ("relevance" ou "time")
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
                    'published_at': comment['publishedAt'],  # Ajouter la date de publication
                    'video_id': video_id,
                    'video_title': "N/A"  # On peut ajouter le titre de la vidéo plus tard si nécessaire
                })

            return comments
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
                trending_videos.append({
                    'id': item['id'],
                    'title': item['snippet']['title'],
                    'channel_title': item['snippet']['channelTitle'],
                    'channel_id': item['snippet']['channelId'],
                    'view_count': int(item['statistics'].get('viewCount', 0)),
                    'comment_count': int(item['statistics'].get('commentCount', 0)),
                    'published_at': item['snippet']['publishedAt'],
                    'url': f"https://www.youtube.com/watch?v={item['id']}"
                })
            return trending_videos
        except Exception as e:
            print(f"YouTube API Error (get_trending_videos): {str(e)}")
            return []
