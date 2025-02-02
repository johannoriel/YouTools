from typing import List, Dict, Any, Optional
import tweepy
from atproto import Client as AtprotoClient, models
from atproto import client_utils
import telegram
import asyncio
import requests
import jwt
import re
from datetime import datetime
import streamlit as st
from typing import List, Dict, Any, Optional
from plugins.common import get_credentials
from googleapiclient.discovery import build

class TwitterAPI:
    def __init__(self, config):
        self.client = tweepy.Client(
            bearer_token=config['common']['twitter_bearer_token'],
            consumer_key=config['common']['twitter_api_key'],
            consumer_secret=config['common']['twitter_api_secret'],
            access_token=config['common']['twitter_access_token'],
            access_token_secret=config['common']['twitter_access_token_secret']
        )
        # Initialisation de l'API v1 si activée
        if config['common'].get('twitter_api_v1_enabled', False):
            self.client_v1 = tweepy.API(
                tweepy.OAuth1UserHandler(
                    config['common']['twitter_api_v1_consumer_key'],
                    config['common']['twitter_api_v1_consumer_secret'],
                    config['common']['twitter_api_v1_access_token'],
                    config['common']['twitter_api_v1_access_token_secret']
                )
            )
        else:
            self.client_v1 = None

    def create_thread(self, posts: List[str]) -> Optional[List[Any]]:
        try:
            previous_tweet_id = None
            responses = []

            for post in posts:
                if previous_tweet_id:
                    response = self.client.create_tweet(
                        text=post,
                        in_reply_to_tweet_id=previous_tweet_id
                    )
                else:
                    response = self.client.create_tweet(text=post)

                previous_tweet_id = response.data['id']
                responses.append(response)

            return responses
        except Exception as e:
            st.error(f"Twitter: {str(e)}")
            return None

    def search_v2(self, query: str, max_results: int = 10, language: str = "fr") -> List[Dict[str, Any]]:
        try:
            # Ajouter le filtre de langue à la requête (ex: "lang:fr")
            if language:
                query = f"{query} lang:{language}"

            # Effectuer la recherche avec les champs et expansions nécessaires
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=["author_id", "text"],
                expansions=["author_id"]
            )

            # Traiter les tweets récupérés
            tweets = []
            for tweet in response.data:
                user = next(u for u in response.includes['users'] if u.id == tweet.author_id)
                tweet_url = f"https://twitter.com/{user.username}/status/{tweet.id}"
                tweets.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'user': user.username,
                    'url': tweet_url
                })

            return tweets
        except Exception as e:
            st.error(f"Twitter API v2 Search Error: {str(e)}")
            return []

    def search_v1(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        try:
            if not self.client_v1:
                st.error("Twitter API v1 is not enabled in configuration.")
                return []

            tweets = self.client_v1.search_tweets(q=query, count=max_results, tweet_mode="extended")
            return [{
                'id': tweet.id_str,
                'text': tweet.full_text,
                'user': tweet.user.screen_name,
                'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}"  # Ajout de l'URL du tweet
            } for tweet in tweets]
        except Exception as e:
            st.error(f"Twitter API v1 Search Error: {str(e)}")
            return []

    def create_tweet(self, text: str, in_reply_to_tweet_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            response = self.client.create_tweet(
                text=text,
                in_reply_to_tweet_id=in_reply_to_tweet_id
            )
            return response.data
        except Exception as e:
            st.error(f"Twitter API v2 Create Tweet Error: {str(e)}")
            return None

class BlueskyAPI:
    def __init__(self, config):
        self.client = AtprotoClient()
        self.client.login(config['common']['bluesky_handle'], config['common']['bluesky_password'])
        self.base_url = "https://public.api.bsky.app"  # URL de l'API publique Bluesky

    def _prepare_post(self, text: str) -> Any:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)

        if not urls:
            return text

        builder = client_utils.TextBuilder()
        segments = re.split(url_pattern, text)
        for i, segment in enumerate(segments):
            if segment:
                builder.text(segment)
            if i < len(urls):
                builder.link(urls[i], urls[i])

        return builder

    def create_thread(self, posts: List[str]) -> Optional[List[Any]]:
        try:
            responses = []
            root_ref = None
            parent_ref = None

            for post in posts:
                prepared_text = self._prepare_post(post)

                if root_ref is None:
                    if isinstance(prepared_text, client_utils.TextBuilder):
                        response = self.client.send_post(text_builder=prepared_text)
                    else:
                        response = self.client.send_post(text=prepared_text)
                    root_ref = models.create_strong_ref(response)
                    parent_ref = root_ref
                else:
                    reply_ref = models.AppBskyFeedPost.ReplyRef(
                        root=root_ref,
                        parent=parent_ref
                    )

                    if isinstance(prepared_text, client_utils.TextBuilder):
                        response = self.client.send_post(text=prepared_text, reply_to=reply_ref)
                    else:
                        response = self.client.send_post(text=prepared_text, reply_to=reply_ref)
                    parent_ref = models.create_strong_ref(response)

                responses.append(response)

            return responses
        except Exception as e:
            st.error(f"Bluesky: {str(e)}")
            return None

    def search_posts(self, query: str, max_results: int = 10, language: str = "fr") -> List[Dict[str, Any]]:
        """
        BUG COTE BLUEKSY => la recherche ne marche pas encore
        Recherche des posts sur Bluesky en fonction des mots-clés.
        :param query: Mots-clés de recherche
        :param max_results: Nombre maximum de posts à récupérer
        :param language: Langue des posts à rechercher (par défaut "fr")
        :return: Liste des posts trouvés
        """
        try:
            # Paramètres de la requête
            params = {
                "q": query,  # Requête de recherche
                "sort": "latest",  # Tri par date (les plus récents en premier)
                "lang": language,  # Filtre par langue
                "limit": min(int(max_results), 100)  # Limite le nombre de résultats (max 100)
            }

            # Appel à l'API Bluesky
            response = requests.get(
                f"{self.base_url}/xrpc/app.bsky.feed.searchPosts",
                params=params
            )

            # Vérification de la réponse
            if response.status_code != 200:
                st.error(f"Bluesky API Search Error: {response.status_code} - {response.text}")
                return []

            # Traitement des résultats
            posts = []
            for post in response.json().get("posts", []):
                posts.append({
                    'id': post['uri'].split('/')[-1],  # Récupère l'ID du post
                    'text': post['record']['text'],  # Texte du post
                    'handle': post['author']['handle'],  # Handle de l'auteur
                    'url': f"https://bsky.app/profile/{post['author']['handle']}/post/{post['uri'].split('/')[-1]}"  # URL du post
                })

            return posts
        except Exception as e:
            st.error(f"Bluesky API Search Error: {str(e)}")
            return []

    def search_posts_api(self, query: str, max_results: int = 10, language: str = "fr") -> List[Dict[str, Any]]:
        """
        BUG COTE BLUEKSY => la recherche ne marche pas encore
        Recherche des posts sur Bluesky en fonction des mots-clés.
        :param query: Mots-clés de recherche
        :param max_results: Nombre maximum de posts à récupérer
        :param language: Langue des posts à rechercher (par défaut "fr")
        :return: Liste des posts trouvés
        """
        try:
            # Création des paramètres de recherche
            params = dict(
                        q=query,  # Requête de recherche
                        limit=min(int(max_results), 100),  # Limite le nombre de résultats (max 100)
                        lang=language,  # Filtre par langue
                        sort="latest"  # Tri par date (les plus récents en premier)
                    )

            # Appel à l'API de recherche
            response = self.client.app.bsky.feed.search_posts(params)

            # Traitement des résultats
            posts = []
            for post in response.posts:
                posts.append({
                    'id': post.uri.split('/')[-1],
                    'text': post.record.text,
                    'handle': post.author.handle,
                    'url': f"https://bsky.app/profile/{post.author.handle}/post/{post.uri.split('/')[-1]}"
                })

            return posts
        except Exception as e:
            import traceback
            st.error(f"Bluesky API Search Error: {str(e)}")
            st.error("Full traceback:")
            st.code(traceback.format_exc())
            return []

    def create_post(self, text: str, in_reply_to_post_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Crée un post sur Bluesky, éventuellement en réponse à un autre post.
        :param text: Texte du post
        :param in_reply_to_post_id: ID du post auquel répondre (optionnel)
        :return: Réponse de l'API
        """
        try:
            if in_reply_to_post_id:
                # Si c'est une réponse, on récupère le post parent
                parent_post = self.client.get_post(in_reply_to_post_id)
                reply_ref = models.AppBskyFeedPost.ReplyRef(
                    root=parent_post,
                    parent=parent_post
                )
                response = self.client.send_post(text=text, reply_to=reply_ref)
            else:
                # Sinon, on crée un post simple
                response = self.client.send_post(text=text)

            return {
                'id': response.uri.split('/')[-1],  # Récupère l'ID du post créé
                'text': text,
                'url': f"https://bsky.app/profile/{self.client.me.handle}/post/{response.uri.split('/')[-1]}"  # URL du post
            }
        except Exception as e:
            st.error(f"Bluesky API Create Post Error: {str(e)}")
            return None

class TelegramAPI:
    def __init__(self, config):
        self.bot_token = config['common']['telegram_bot_token']
        self.channel_id = config['common']['telegram_channel_id']

    async def _post_async(self, text: str) -> Optional[Any]:
        try:
            bot = telegram.Bot(token=self.bot_token)
            response = await bot.send_message(
                chat_id=self.channel_id,
                text=text,
                parse_mode='HTML'
            )
            return response
        except Exception as e:
            st.error(f"Telegram: {str(e)}")
            return None

    def post(self, text: str) -> Optional[Any]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._post_async(text))
        finally:
            loop.close()

class GhostAPI:
    def __init__(self, config):
        self.ghost_url = config['common']['ghost_url']
        self.ghost_api_key = config['common']['ghost_api_key']

    def post(self, title: str, content: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.ghost_url or not self.ghost_api_key:
                st.error("Ghost URL or API Key is missing in configuration.")
                return None

            id, secret = self.ghost_api_key.split(':')
            iat = int(datetime.now().timestamp())

            header = {'alg': 'HS256', 'typ': 'JWT', 'kid': id}
            payload = {
                'iat': iat,
                'exp': iat + 5 * 60,
                'aud': '/admin/'
            }

            token = jwt.encode(payload, bytes.fromhex(secret), algorithm='HS256', headers=header)
            headers = {'Authorization': f'Ghost {token}'}
            body = {
                'posts': [{
                    'title': title,
                    'html': content,
                    'status': 'published'
                }]
            }

            response = requests.post(
                f"{self.ghost_url}/ghost/api/admin/posts/?source=html",
                headers=headers,
                json=body
            )

            if response.status_code == 201:
                return response.json()
            else:
                st.error(f"Ghost API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Ghost: {str(e)}")
            return None

class YoutubeAPI:
    def __init__(self, config):
        credentials = get_credentials()
        self.youtube = build('youtube', 'v3', credentials=credentials)
        self.channel_id = config['common']['channel_id']

    def post(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.channel_id:
                st.error("YouTube Channel ID is missing in configuration.")
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
            st.error(f"YouTube Post: {str(e)}")
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
            st.error(f"YouTube API Error (get_channel_info): {str(e)}")
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
            st.error(f"Une erreur s'est produite : {e}")
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
            st.error(f"YouTube API Error (get_video_details): {str(e)}")
            return None

    def search_videos(self, query: str, max_results: int = 5, order: str = "date", language: str = "fr") -> List[Dict[str, Any]]:
        """
        Recherche des vidéos sur YouTube en fonction des mots-clés.
        :param query: Mots-clés de recherche
        :param max_results: Nombre maximum de vidéos à récupérer
        :param order: Ordre des résultats ("date" pour les plus récentes, "relevance" pour la pertinence)
        :param language: Langue des vidéos à rechercher (par défaut "fr" pour le français)
        """
        try:
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                maxResults=max_results,
                type="video",
                order=order,
                relevanceLanguage=language  # Ajouter le filtre de langue
            )
            response = request.execute()

            videos = []
            for item in response['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                channel_title = item['snippet']['channelTitle']  # Nom de la chaîne
                channel_id = item['snippet']['channelId']  # ID de la chaîne

                # Récupérer les informations de la chaîne
                channel_info = self.get_channel_info(channel_id)
                subscriber_count = channel_info['subscriber_count'] if channel_info else 0

                # Récupérer les détails de la vidéo
                video_details = self.get_video_details(video_id)
                view_count = video_details['view_count'] if video_details else 0
                comment_count = video_details['comment_count'] if video_details else 0

                videos.append({
                    'id': video_id,
                    'title': title,
                    'channel_title': channel_title,  # Ajouter le nom de la chaîne
                    'channel_id': channel_id,  # Ajouter l'ID de la chaîne
                    'subscriber_count': subscriber_count,  # Ajouter le nombre d'abonnés
                    'view_count': view_count,  # Ajouter le nombre de vues
                    'comment_count': comment_count,  # Ajouter le nombre de commentaires
                    'url': f"https://www.youtube.com/watch?v={video_id}"
                })

            return videos
        except Exception as e:
            st.error(f"YouTube API Error (search_videos): {str(e)}")
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
            st.error(f"YouTube API Error (get_comments): {str(e)}")
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
            st.error(f"YouTube API Error (post_comment_reply): {str(e)}")
            return None
