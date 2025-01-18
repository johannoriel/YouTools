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

    def search_v2(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        try:
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=["author_id", "text"],
                expansions=["author_id"]
            )

            tweets = []
            for tweet in response.data:
                user = next(u for u in response.includes['users'] if u.id == tweet.author_id)
                tweet_url = f"https://twitter.com/{user.username}/status/{tweet.id}"
                tweets.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'user': user.username,
                    'url': tweet_url  # Ajout de l'URL du tweet
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

    def search_videos(self, query: str, max_results: int = 5, order: str = "date") -> List[Dict[str, Any]]:
        """
        Recherche des vidéos sur YouTube en fonction des mots-clés.
        :param query: Mots-clés de recherche
        :param max_results: Nombre maximum de vidéos à récupérer
        :param order: Ordre des résultats ("date" pour les plus récentes, "relevance" pour la pertinence)
        """
        try:
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                maxResults=max_results,
                type="video",
                order=order  # Utiliser l'ordre spécifié
            )
            response = request.execute()

            videos = []
            for item in response['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                channel_title = item['snippet']['channelTitle']  # Nom de la chaîne
                channel_id = item['snippet']['channelId']  # ID de la chaîne
                videos.append({
                    'id': video_id,
                    'title': title,
                    'channel_title': channel_title,  # Ajouter le nom de la chaîne
                    'channel_id': channel_id,  # Ajouter l'ID de la chaîne
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
