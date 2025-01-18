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

class YoutubePostAPI:
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
