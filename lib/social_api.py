from typing import List, Dict, Any, Optional
import tweepy
from atproto import Client as AtprotoClient, models
from atproto import client_utils
import telegram
import asyncio
import requests
import jwt
import re
import os
from datetime import datetime
import streamlit as st
import json
import pytz
from langdetect import detect
from googlesearch import search
import urllib
import time


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

    def create_thread(self, posts: List[Any]) -> Optional[List[Any]]:
        try:
            previous_tweet_id = None
            responses = []

            for post in posts:
                if isinstance(post, tuple):  # First post with meme
                    text, media_path = post
                    if not self.client_v1:
                        st.error("Twitter v1 API needed for media upload")
                        return None

                    # Upload media using v1 API
                    media = self.client_v1.media_upload(filename=media_path)
                    media_ids = [media.media_id]

                    # Create tweet with media using v2 API
                    response = self.client.create_tweet(
                        text=text,
                        media_ids=media_ids
                    )
                else:
                    # Regular text tweet
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
                user = next(
                    u for u in response.includes['users'] if u.id == tweet.author_id)
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

            tweets = self.client_v1.search_tweets(
                q=query, count=max_results, tweet_mode="extended")
            return [{
                'id': tweet.id_str,
                'text': tweet.full_text,
                'user': tweet.user.screen_name,
                # Ajout de l'URL du tweet
                'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}"
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

    def get_following_timeline(self, user_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
        try:
            response = self.client.get_home_timeline(
                max_results=max_results,
                tweet_fields=["created_at", "text", "author_id",
                              "public_metrics", "lang", "source", "referenced_tweets"],
                expansions=["author_id"],
                user_fields=["name", "username", "profile_image_url"],
                # Exclure les réponses et retweets pour une timeline plus propre
                exclude=["replies", "retweets"],
                user_auth=True
            )
            tweets = []
            for tweet in response.data:
                user = next(
                    u for u in response.includes['users'] if u.id == tweet.author_id)
                # Chercher le parent_id dans referenced_tweets (type "replied_to")
                parent_id = None
                if tweet.referenced_tweets:
                    for ref_tweet in tweet.referenced_tweets:
                        if ref_tweet.type == "replied_to":
                            parent_id = str(ref_tweet.id)
                            break
                tweets.append({
                    'id': str(tweet.id),
                    'text': tweet.text,
                    'user': user.username,
                    'name': user.name,
                    'profile_image_url': user.profile_image_url,
                    'created_at': tweet.created_at.isoformat(),
                    'lang': tweet.lang,
                    'source': tweet.source,
                    'public_metrics': {
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'reply_count': tweet.public_metrics['reply_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'quote_count': tweet.public_metrics['quote_count']
                    },
                    'url': f"https://twitter.com/{user.username}/status/{tweet.id}",
                    'parent_id': parent_id
                })
            return tweets
        except Exception as e:
            st.error(f"Twitter API Timeline Error: {str(e)}")
            return []

    def get_following_timeline_v1(self, max_results: int = 20) -> List[Dict[str, Any]]:
        try:
            api_v1 = self.client_v1
            response = api_v1.home_timeline(
                count=max_results,
                exclude_replies=True,
                include_entities=True
            )
            tweets = []
            for tweet in response:
                tweets.append({
                    'id': str(tweet.id),
                    'text': tweet.text,
                    'user': tweet.user.screen_name,
                    'name': tweet.user.name,
                    'profile_image_url': tweet.user.profile_image_url_https,
                    'created_at': tweet.created_at.isoformat(),
                    'lang': tweet.lang,
                    'source': tweet.source,
                    'public_metrics': {
                        'retweet_count': tweet.retweet_count,
                        'reply_count': 0,  # Non disponible en v1, défini à 0
                        'like_count': tweet.favorite_count,
                        'quote_count': 0  # Non disponible en v1, défini à 0
                    },
                    'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}",
                    'parent_id': str(tweet.in_reply_to_status_id) if tweet.in_reply_to_status_id else None
                })
            return tweets
        except Exception as e:
            st.error(f"Twitter API V1 Timeline Error: {str(e)}")
            return []

    def organize_tweets_into_threads(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Regrouper les tweets par thread
        threads = {}
        for tweet in tweets:
            if tweet['parent_id'] is None:
                # Tweet racine (pas de parent)
                threads[tweet['id']] = {'root_tweet': tweet, 'replies': []}
            else:
                # Tweet réponse : chercher le thread parent
                root_id = tweet['parent_id']
                while root_id:
                    # Remonter jusqu'à la racine du thread
                    parent_tweet = next(
                        (t for t in tweets if t['id'] == root_id), None)
                    if parent_tweet and parent_tweet['parent_id']:
                        root_id = parent_tweet['parent_id']
                    else:
                        break
                if root_id and root_id in threads:
                    threads[root_id]['replies'].append(tweet)
                else:
                    # Si le parent n'est pas dans les tweets récupérés, créer un thread orphelin
                    threads[tweet['id']] = {'root_tweet': tweet, 'replies': []}

        # Convertir les threads en liste pour l'affichage
        thread_list = []
        for thread_id, thread in threads.items():
            thread_list.append({
                'root_tweet': thread['root_tweet'],
                'replies': thread['replies']
            })

        return thread_list

    def get_rate_limit_status(self) -> Dict[str, Any]:
        try:
            api_v1 = self.client_v1
            status = api_v1.rate_limit_status()
            # import streamlit as st
            # st.write(status)
            # Extraire les informations pour les endpoints pertinents
            timeline_limit = status['resources']['statuses']['/statuses/home_timeline']
            update_limit = status['resources']['tweets&POST']['/tweets&POST']
            return {
                'timeline': {
                    'remaining': timeline_limit['remaining'],
                    'limit': timeline_limit['limit'],
                    'reset': datetime.fromtimestamp(timeline_limit['reset']).isoformat() if timeline_limit['reset'] else None
                },
                'update': {
                    'remaining': update_limit['remaining'],
                    'limit': update_limit['limit'],
                    'reset': datetime.fromtimestamp(update_limit['reset']).isoformat() if update_limit['reset'] else None
                }
            }
        except Exception as e:
            st.error(f"Twitter API Rate Limit Error: {str(e)}")
            return {
                'timeline': {'remaining': 0, 'limit': 0, 'reset': None},
                'update': {'remaining': 0, 'limit': 0, 'reset': None}
            }

    def create_tweet_v1(self, text: str, in_reply_to_tweet_id: str = None) -> bool:
        try:
            api_v1 = self.client_v1
            status = api_v1.update_status(
                status=text,
                in_reply_to_status_id=in_reply_to_tweet_id
            )
            return bool(status)
        except Exception as e:
            st.error(f"Twitter API V1 Create Tweet Error: {str(e)}")
            return False


class BlueskyAPI:
    def __init__(self, config):
        self.client = AtprotoClient()
        self.client.login(config['common']['bluesky_handle'],
                          config['common']['bluesky_password'])
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

    def create_thread(self, posts: List[Any]) -> Optional[List[Any]]:
        try:
            responses = []
            root_ref = None
            parent_ref = None

            for post in posts:
                if isinstance(post, tuple):  # First post with meme
                    text, media_path = post
                    prepared_text = self._prepare_post(text)

                    # Read the image file and determine mime type
                    with open(media_path, 'rb') as f:
                        img_data = f.read()

                    # Simple mime type detection based on extension
                    ext = os.path.splitext(media_path)[1].lower()
                    mime_type = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif'
                    }.get(ext, 'image/jpeg')  # Default to jpeg if unknown

                    # Upload blob with correct parameters
                    blob = self.client.upload_blob(
                        img_data)

                    if root_ref is None:
                        response = self.client.send_post(
                            text=prepared_text if isinstance(
                                prepared_text, str) else prepared_text.build(),
                            embed=models.AppBskyEmbedImages.Main(
                                images=[models.AppBskyEmbedImages.Image(
                                    image=blob.blob,  # Changed from blob to blob.blob
                                    alt="Generated meme"
                                )]
                            )
                        )
                        root_ref = models.create_strong_ref(response)
                        parent_ref = root_ref
                else:
                    prepared_text = self._prepare_post(post)
                    if root_ref is None:
                        response = self.client.send_post(text=prepared_text)
                        root_ref = models.create_strong_ref(response)
                        parent_ref = root_ref
                    else:
                        reply_ref = models.AppBskyFeedPost.ReplyRef(
                            root=root_ref,
                            parent=parent_ref
                        )
                        response = self.client.send_post(
                            text=prepared_text, reply_to=reply_ref)
                        parent_ref = models.create_strong_ref(response)

                responses.append(response)

            return responses
        except Exception as e:
            st.error(f"Bluesky: {str(e)}")
            raise e

    def search_posts_manual(self, query: str, max_results: int = 10, language: str = "fr") -> List[Dict[str, Any]]:
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
                # Limite le nombre de résultats (max 100)
                "limit": min(int(max_results), 100)
            }

            # Appel à l'API Bluesky
            response = requests.get(
                f"{self.base_url}/xrpc/app.bsky.feed.searchPosts",
                params=params
            )

            # Vérification de la réponse
            if response.status_code != 200:
                st.error(
                    f"Bluesky API Search Error: {response.status_code} - {response.text}")
                return []

            # Traitement des résultats
            posts = []
            for post in response.json().get("posts", []):
                posts.append({
                    'id': post['uri'].split('/')[-1],  # Récupère l'ID du post
                    'text': post['record']['text'],  # Texte du post
                    'handle': post['author']['handle'],  # Handle de l'auteur
                    # URL du post
                    'url': f"https://bsky.app/profile/{post['author']['handle']}/post/{post['uri'].split('/')[-1]}"
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
        bquery = " OR ".join(query.split())
        try:
            # Création des paramètres de recherche
            params = dict(
                q=bquery,  # Requête de recherche
                # Limite le nombre de résultats (max 100)
                limit=min(int(max_results), 100),
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

    def search_posts_google(self, query: str, max_results: int = 10, language: str = "fr") -> List[Dict[str, Any]]:
        """
        Recherche de posts Bluesky via Google et récupération du contenu via atproto.
        """
        try:
            # Forcer max_results à être un entier
            max_results = int(max_results)

            # Construire la requête Google avec restriction au domaine Bluesky
            google_query = f'site:bsky.app "{query}" lang:{language}'
            print(f"Query google >>>>>>>>>> {google_query}")
            posts = []
            for url in search(google_query, num_results=max_results, lang=language):
                # Vérifier que l'URL est un post Bluesky valide
                if "/profile/" in url and "/post/" in url:
                    handle = url.split("/profile/")[1].split("/post/")[0]
                    post_id = url.split("/post/")[1]

                    # Construire l'URI ATProtocol pour récupérer le post
                    # Note : Nous avons besoin du DID de l'auteur, mais pour simplifier, on utilise get_post_thread avec l'URI partiel
                    try:
                        # Récupérer le post via l'API atproto
                        post_uri = f"at://{handle}/app.bsky.feed.post/{post_id}"
                        post_response = self.client.get_post_thread(
                            uri=post_uri)

                        # Extraire les détails du post principal
                        post = post_response.thread.post
                        posts.append({
                            'id': post_id,
                            'text': post.record.text,
                            'handle': handle,
                            'url': url
                        })
                    except Exception as post_error:
                        st.warning(
                            f"Impossible de récupérer le post {post_id} : {str(post_error)}")
                        # Ajouter un placeholder si la récupération échoue
                        posts.append({
                            'id': post_id,
                            'text': f"[Erreur lors de la récupération du post] (URL: {url})",
                            'handle': handle,
                            'url': url
                        })

            if not posts:
                st.warning("Aucun post Bluesky trouvé via Google.")
            return posts
        except ValueError as ve:
            st.error(f"Erreur de conversion dans les paramètres : {str(ve)}")
            return []
        except Exception as e:
            st.error(f"Google Search Error: {str(e)}")
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
                # Récupère l'ID du post créé
                'id': response.uri.split('/')[-1],
                'text': text,
                # URL du post
                'url': f"https://bsky.app/profile/{self.client.me.handle}/post/{response.uri.split('/')[-1]}"
            }
        except Exception as e:
            st.error(f"Bluesky API Create Post Error: {str(e)}")
            return None


class TelegramAPI:
    def __init__(self, config):
        self.bot_token = config['common']['telegram_bot_token']
        self.channel_id = config['common']['telegram_channel_id']

    async def _post_async(self, content: Any) -> Optional[Any]:
        try:
            bot = telegram.Bot(token=self.bot_token)
            if isinstance(content, tuple):  # First post with meme
                text, media_path = content
                with open(media_path, 'rb') as photo:
                    response = await bot.send_photo(
                        chat_id=self.channel_id,
                        photo=photo,
                        caption=text,
                        parse_mode='HTML'
                    )
            else:
                response = await bot.send_message(
                    chat_id=self.channel_id,
                    text=content,
                    parse_mode='HTML'
                )
            return response
        except Exception as e:
            st.error(f"Telegram: {str(e)}")
            return None

    def post(self, content: Any) -> Optional[Any]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._post_async(content))
        finally:
            loop.close()

    def search_channels(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Recherche des canaux Telegram via Google (placeholder, car l'API ne permet pas une recherche native).
        """
        try:
            google_query = f'site:t.me "{query}" -inurl:(/s/)'
            channels = []
            for url in search(google_query, num_results=max_results):
                if "t.me/" in url and "/s/" not in url:
                    channel_id = url.split("t.me/")[1].split("/")[0]
                    channels.append({
                        'id': channel_id,
                        'text': f"Channel: @{channel_id}",
                        'url': url
                    })
            return channels
        except Exception as e:
            st.error(f"Telegram Channel Search Error: {str(e)}")
            return []


class GhostAPI:
    def __init__(self, config):
        self.ghost_url = config['common']['ghost_url']
        self.ghost_api_key = config['common']['ghost_api_key']

    def post(self, title: str, content: str, publish_immediately: bool = True, feature_image: Optional[str] = None) -> Optional[Dict[str, Any]]:
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

            # Upload image if provided
            feature_image_url = None
            if feature_image and os.path.exists(feature_image):
                with open(feature_image, 'rb') as f:
                    files = {
                        'file': (os.path.basename(feature_image), f, 'image/png'),
                        'ref': (None, feature_image)
                    }
                    image_response = requests.post(
                        f"{self.ghost_url}/ghost/api/admin/images/upload/",
                        headers=headers,
                        files=files
                    )
                    image_response.raise_for_status()
                    feature_image_url = image_response.json().get('images', [{}])[0].get('url')

            # Prepare post data
            body = {
                'posts': [{
                    'title': title,
                    'html': content,
                    'status': 'published' if publish_immediately else 'draft'
                }]
            }
            if feature_image_url:
                body['posts'][0]['feature_image'] = feature_image_url

            response = requests.post(
                f"{self.ghost_url}/ghost/api/admin/posts/?source=html",
                headers=headers,
                json=body
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Ghost: {str(e)}")
            return None

class LinkedinAPI:
    def __init__(self, config):
        self.client_id = config['common']['linkedin_client_id']
        self.client_secret = config['common']['linkedin_client_secret']
        self.base_url = "https://api.linkedin.com"
        self.access_token = config['common']['linkedin_access_token']
        self.redirect_uri = config['common'].get('linkedin_redirect_uri', 'https://your-app.com/callback')
        self.api_version = config['common'].get('linkedin_api_version', '202504')  # Default to 202504
        self.person_urn = None  # Will be set by _get_person_urn

    def _get_access_token(self, code: Optional[str] = None, refresh_token: Optional[str] = None) -> Optional[str]:
        """
        Retrieves an access token using authorization code or refresh token.
        :param code: Authorization code from OAuth2 redirect (optional).
        :param refresh_token: Refresh token to obtain a new access token (optional).
        :return: Access token or None if the request fails.
        """
        try:
            auth_url = "https://www.linkedin.com/oauth/v2/accessToken"
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            if code:
                payload = {
                    'grant_type': 'authorization_code',
                    'code': code,
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'redirect_uri': self.redirect_uri
                }
            elif refresh_token:
                payload = {
                    'grant_type': 'refresh_token',
                    'refresh_token': refresh_token,
                    'client_id': self.client_id,
                    'client_secret': self.client_secret
                }
            else:
                st.error("No authorization code or refresh token provided.")
                return None

            response = requests.post(auth_url, data=payload, headers=headers)
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            return self.access_token
        except Exception as e:
            st.error(f"LinkedIn API Error (get_access_token): {str(e)}")
            return None

    def _get_person_urn(self) -> Optional[str]:
        """
        Retrieves the authenticated user's person_urn using the /v2/userinfo endpoint.
        :return: Person URN (e.g., urn:li:person:{id}) or None if the request fails.
        """
        try:
            api_url_me = f"{self.base_url}/v2/userinfo"
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'X-Restli-Protocol-Version': '2.0.0',
                'LinkedIn-Version': self.api_version
            }
            response = requests.get(api_url_me, headers=headers)
            response.raise_for_status()
            user_data = response.json()
            self.person_urn = f"urn:li:person:{user_data['sub']}"
            return self.person_urn
        except Exception as e:
            st.error(f"LinkedIn API Error (get_person_urn): {str(e)}")
            return None

    def post_article(self, title: str, content: str, source_url: Optional[str] = None, feature_image: Optional[str] = None) -> Optional[Dict[str, Any]]:
            """
            Publishes an article using the Posts API (version 2025-07), with optional image upload.
            :param title: Article title.
            :param content: Article content (plain text or markdown, max 3000 characters).
            :param source_url: URL of the article source (optional).
            :param feature_image: Path to the image file (optional).
            :return: API response or None if the request fails.
            """
            try:
                # Ensure person_urn is set
                if not self.person_urn:
                    self.person_urn = self._get_person_urn()
                    if not self.person_urn:
                        return None

                # Use the latest API version
                latest_version = "202507"  # Updated to latest version
                headers = {
                    'Authorization': f'Bearer {self.access_token}',
                    'X-Restli-Protocol-Version': '2.0.0',
                    'LinkedIn-Version': latest_version,  # Use latest version
                    'Content-Type': 'application/json'
                }

                # Upload image if provided
                image_urn = None
                if feature_image and os.path.exists(feature_image):
                    # Validate image file type
                    ext = os.path.splitext(feature_image)[1].lower()
                    if ext not in ['.png', '.jpg', '.jpeg', '.gif']:
                        st.error("Unsupported image format. Use PNG, JPEG, or GIF.")
                        return None

                    # Initialize image upload with proper headers
                    init_url = f"{self.base_url}/rest/images?action=initializeUpload"
                    init_headers = {
                        'Authorization': f'Bearer {self.access_token}',
                        'X-Restli-Protocol-Version': '2.0.0',
                        'LinkedIn-Version': latest_version,
                        'Content-Type': 'application/json'
                    }
                    init_body = {
                        'initializeUploadRequest': {
                            'owner': self.person_urn
                        }
                    }

                    init_response = requests.post(init_url, headers=init_headers, json=init_body)
                    init_response.raise_for_status()
                    init_data = init_response.json()['value']
                    upload_url = init_data['uploadUrl']
                    image_urn = init_data['image']

                    # Upload image file
                    with open(feature_image, 'rb') as f:
                        upload_headers = {
                            'Authorization': f'Bearer {self.access_token}',
                            'Content-Type': f"image/{ext.lstrip('.')}"
                        }
                        upload_response = requests.put(upload_url, headers=upload_headers, data=f)
                        upload_response.raise_for_status()

                # Truncate content to 3000 characters (LinkedIn limit)
                content = content[:3000]

                # Simplified post body structure for article sharing
                if source_url:
                    # Post with article/link sharing
                    body = {
                        'author': self.person_urn,
                        'commentary': content,
                        'visibility': 'PUBLIC',
                        'distribution': {
                            'feedDistribution': 'MAIN_FEED'
                        },
                        'content': {
                            'article': {
                                'source': source_url,
                                'title': title
                            }
                        },
                        'lifecycleState': 'PUBLISHED',
                        'isReshareDisabledByAuthor': False
                    }

                    # Add thumbnail if image was uploaded
                    if image_urn:
                        body['content']['article']['thumbnail'] = image_urn
                else:
                    # Simple text post with optional image
                    body = {
                        'author': self.person_urn,
                        'commentary': f"{title}\n\n{content}",
                        'visibility': 'PUBLIC',
                        'distribution': {
                            'feedDistribution': 'MAIN_FEED'
                        },
                        'lifecycleState': 'PUBLISHED',
                        'isReshareDisabledByAuthor': False
                    }

                    # Add image if uploaded
                    if image_urn:
                        body['content'] = {
                            'media': {
                                'title': title,
                                'id': image_urn
                            }
                        }

                # Debug: Print the request body
                print("LinkedIn API Request Body:", json.dumps(body, indent=2))

                # Create post
                response = requests.post(
                    f"{self.base_url}/rest/posts",
                    headers=headers,
                    json=body
                )

                # Debug: Print response details
                print(f"LinkedIn API Response Status: {response.status_code}")
                if response.status_code != 201:
                    print(f"LinkedIn API Response Text: {response.text}")

                response.raise_for_status()
                post_id = response.headers.get('x-restli-id', 'N/A')
                return {'id': post_id, 'status': 'success'}

            except requests.exceptions.HTTPError as e:
                error_details = ""
                try:
                    error_details = e.response.json()
                except:
                    error_details = e.response.text

                st.error(f"LinkedIn API HTTP Error: {e.response.status_code}")
                st.error(f"Error details: {error_details}")

                # Common error solutions
                if e.response.status_code == 422:
                    st.error("Possible causes of 422 error:")
                    st.error("1. Invalid post structure or missing required fields")
                    st.error("2. Content exceeds LinkedIn limits")
                    st.error("3. Invalid person URN or permissions")
                    st.error("4. Outdated API version")
                elif e.response.status_code == 401:
                    st.error("Authentication error - check your access token")
                elif e.response.status_code == 403:
                    st.error("Permission error - ensure you have 'w_member_social' permission")

                return None
            except Exception as e:
                st.error(f"LinkedIn API Error (post_article): {str(e)}")
                return None


    def search_posts(self, query: str, max_results: int = 10, language: str = "fr") -> List[Dict[str, Any]]:
        """
        Searches for posts on LinkedIn based on keywords.
        :param query: Search keywords.
        :param max_results: Maximum number of posts to retrieve.
        :param language: Language of posts to search (default "fr").
        :return: List of found posts.
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'X-Restli-Protocol-Version': '2.0.0',
                'LinkedIn-Version': self.api_version
            }
            params = {
                'q': query,
                'count': max_results,
                'sort': 'relevance',
                'locale.language': language
            }
            response = requests.get(
                f"{self.base_url}/v2/search", headers=headers, params=params)
            response.raise_for_status()
            posts = response.json().get('elements', [])

            formatted_posts = []
            for post in posts:
                formatted_posts.append({
                    'id': post.get('id'),
                    'text': post.get('commentary', {}).get('text', ''),
                    'author': post.get('author', {}).get('name', 'N/A'),
                    'published_at': post.get('lastModifiedTime', {}).get('time', 'N/A'),
                    'url': post.get('url', 'N/A')
                })

            return formatted_posts
        except Exception as e:
            st.error(f"LinkedIn API Error (search_posts): {str(e)}")
            return []

    def get_post_comments(self, post_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves comments for a LinkedIn post.
        :param post_id: ID of the post.
        :param max_results: Maximum number of comments to retrieve.
        :return: List of comments.
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'X-Restli-Protocol-Version': '2.0.0',
                'LinkedIn-Version': self.api_version
            }
            response = requests.get(
                f"{self.base_url}/v2/socialActions/{post_id}/comments", headers=headers)
            response.raise_for_status()
            comments = response.json().get('elements', [])

            formatted_comments = []
            for comment in comments:
                formatted_comments.append({
                    'id': comment.get('id'),
                    'text': comment.get('message', {}).get('text', ''),
                    'author': comment.get('actor', {}).get('name', 'N/A'),
                    'published_at': comment.get('lastModifiedTime', {}).get('time', 'N/A')
                })

            return formatted_comments[:max_results]
        except Exception as e:
            st.error(f"LinkedIn API Error (get_post_comments): {str(e)}")
            return []

    def post_comment(self, post_id: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Posts a comment on a LinkedIn post.
        :param post_id: ID of the post.
        :param text: Comment text.
        :return: API response or None if the request fails.
        """
        try:
            if not self.person_urn:
                self.person_urn = self._get_person_urn()
                if not self.person_urn:
                    return None

            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'X-Restli-Protocol-Version': '2.0.0',
                'LinkedIn-Version': self.api_version,
                'Content-Type': 'application/json',
                'X-Li-Pem-Metadata': 'w_member_social'
            }
            body = {
                'actor': self.person_urn,
                'message': {
                    'text': text
                },
                'object': f"urn:li:share:{post_id}"
            }
            response = requests.post(
                f"{self.base_url}/v2/socialActions/{post_id}/comments", headers=headers, json=body)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"LinkedIn API Error (post_comment): {str(e)}")
            return None


class WordPressAPI:
    def __init__(self, config):
        self.client_id = config['common']['wordpress_client_id']
        self.client_secret = config['common']['wordpress_client_secret']
        self.base_url = "https://public-api.wordpress.com"
        self.access_token = config['common'].get('wordpress_access_token')
        self.redirect_uri = config['common'].get('wordpress_redirect_uri', 'http://localhost:8501/')
        self.site_id = config['common'].get('wordpress_site_id')  # Optional, needed for site-specific actions

    def _get_access_token(self, code: Optional[str] = None, state: Optional[str] = None) -> Optional[str]:
        """
        Retrieves an access token using authorization code or returns authorization URL.
        :param code: Authorization code from OAuth2 redirect (optional).
        :return: Access token if code is provided, authorization URL if not, or None if the request fails.
        """
        try:
            # If no code is provided, generate and return the authorization URL
            if not code:
                state = os.urandom(16).hex()  # Generate random state for CSRF protection
                auth_url = (
                    f"https://public-api.wordpress.com/oauth2/authorize?"
                    f"client_id={self.client_id}&"
                    f"redirect_uri={urllib.parse.quote(self.redirect_uri)}&"
                    f"response_type=code&"
                    f"scope=posts&"
                    f"state={state}"
                )
                return auth_url

            # Verify state parameter to prevent CSRF
            if not state:
                st.error("State parameter missing. Possible CSRF attack.")
                return None

            # If code is provided, exchange it for an access token
            token_url = "https://public-api.wordpress.com/oauth2/token"
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            payload = {
                'grant_type': 'authorization_code',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'state': state,
                'redirect_uri': self.redirect_uri
            }
            st.write(payload)

            response = requests.post(token_url, data=payload, headers=headers)
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            return self.access_token
        except requests.exceptions.HTTPError as e:
            error_details = ""
            try:
                error_details = e.response.json()
            except:
                error_details = e.response.text
            st.error(f"WordPress API Authentication Error: {e.response.status_code}")
            st.error(f"Error details: {error_details}")
            return None
        except Exception as e:
            st.error(f"WordPress API Authentication Error: {str(e)}")
            return None

    def post(self, title: str, content: str, publish_immediately: bool = True, feature_image: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Publishes a post on WordPress.com.
        :param title: Post title.
        :param content: Post content (HTML or plain text).
        :param publish_immediately: Whether to publish immediately or save as draft.
        :param feature_image: Path to the image file (optional).
        :return: API response or None if the request fails.
        """
        try:
            if not self.access_token:
                st.error("No access token available. Please authenticate first.")
                return None

            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }

            # Determine the endpoint based on whether site_id is provided
            if self.site_id:
                post_url = f"{self.base_url}/rest/v1.1/sites/{self.site_id}/posts/new"
            else:
                st.error("WordPress site ID is missing in configuration.")
                return None

            # Upload image if provided
            feature_image_id = None
            if feature_image and os.path.exists(feature_image):
                ext = os.path.splitext(feature_image)[1].lower()
                if ext not in ['.png', '.jpg', '.jpeg', '.gif']:
                    st.error("Unsupported image format. Use PNG, JPEG, or GIF.")
                    return None

                with open(feature_image, 'rb') as f:
                    files = {
                        'media[]': (os.path.basename(feature_image), f, f'image/{ext.lstrip(".")}')
                    }
                    image_response = requests.post(
                        f"{self.base_url}/rest/v1.1/sites/{self.site_id}/media/new",
                        headers={'Authorization': f'Bearer {self.access_token}'},
                        files=files
                    )
                    image_response.raise_for_status()
                    feature_image_id = image_response.json().get('media', [{}])[0].get('ID')

            # Prepare post data
            body = {
                'title': title,
                'content': content,
                'status': 'publish' if publish_immediately else 'draft'
            }
            if feature_image_id:
                body['featured_image'] = feature_image_id

            # Create post
            response = requests.post(post_url, headers=headers, json=body)
            response.raise_for_status()
            post_data = response.json()
            return {
                'id': post_data.get('ID'),
                'title': post_data.get('title'),
                'url': post_data.get('URL'),
                'status': post_data.get('status')
            }
        except requests.exceptions.HTTPError as e:
            error_details = ""
            try:
                error_details = e.response.json()
            except:
                error_details = e.response.text
            st.error(f"WordPress API HTTP Error: {e.response.status_code}")
            st.error(f"Error details: {error_details}")
            return None
        except Exception as e:
            st.error(f"WordPress API Error (post): {str(e)}")
            return None

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from substack import Api
from substack.post import Post
from substack.exceptions import SubstackAPIException
import streamlit as st

class SubstackAPI:
    def __init__(self, config):
        self.email = config['common']['substack_email']
        self.password = config['common']['substack_password']
        publication_urls = config['common']['substack_publication_url']
        # Convertir la chaîne en liste si nécessaire
        if isinstance(publication_urls, str):
            self.publication_urls = [url.strip() for url in publication_urls.split(';') if url.strip()]
        else:
            self.publication_urls = publication_urls
        self.cookies_base_path = "substack_cookies"
        self.selenium_cookies_base_path = "selenium_cookies"
        self.api = None
        # Créer un dictionnaire pour stocker les instances d'API par URL
        self.api_instances = {}

    def get_publication_urls(self) -> List[str]:
        """
        Retourne la liste des URLs de publication configurées.
        :return: Liste des URLs.
        """
        return self.publication_urls

    def _get_cookies_paths(self, publication_url: str) -> tuple[str, str]:
        """
        Génère les chemins des fichiers de cookies en fonction de l'URL de publication.
        :param publication_url: URL de la publication.
        :return: Tuple contenant les chemins des cookies (standard et Selenium).
        """
        # Nettoyer l'URL pour créer un nom de fichier valide
        safe_url = re.sub(r'[^\w\-]', '_', publication_url)
        cookies_path = f"{self.cookies_base_path}_{safe_url}.json"
        selenium_cookies_path = f"{self.selenium_cookies_base_path}_{safe_url}.json"
        return cookies_path, selenium_cookies_path

    def _renew_cookie(self, email: str, password: str, publication_url: str) -> None:
        """
        Log in to Substack using Selenium and save cookies in JSON format for a specific publication.
        :param email: Substack account email.
        :param password: Substack account password.
        :param publication_url: URL of the publication.
        """
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        #chrome_options.add_argument("--headless")  # Run in headless mode for automation
        service = Service("/usr/bin/chromedriver")  # Adjust path if needed
        driver = webdriver.Chrome(service=service, options=chrome_options)

        try:
            print("Starting Substack login process with selenium...")
            driver.get("https://substack.com/sign-in")
            wait = WebDriverWait(driver, 20)
            email_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
            email_field.send_keys(email)

            sign_in_link = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Sign in with password")))
            sign_in_link.click()

            password_field = wait.until(EC.presence_of_element_located((By.NAME, "password")))
            password_field.send_keys(password)
            password_field.send_keys(Keys.RETURN)
            print("Login submitted")

            time.sleep(5)  # Wait for login to complete
            cookies = driver.get_cookies()
            cookies_path, selenium_cookies_path = self._get_cookies_paths(publication_url)
            with open(cookies_path, "w") as file:
                json.dump(cookies, file)
            print(f"Selenium cookies saved to '{cookies_path}'.")

            # Convert Selenium cookies (list of dicts) to {name: value}
            cookie_dict = {c["name"]: c["value"] for c in cookies}
            with open(selenium_cookies_path, "w") as f:
                json.dump(cookie_dict, f)

        finally:
            driver.quit()

    def _initialize_api(self, publication_url: Optional[str] = None, force = False) -> None:
        """
        Initialize the Substack API with cookies for the specified publication, renewing if necessary.
        :param publication_url: URL of the publication to initialize (optional, defaults to first URL).
        """
        try:
            # Utiliser la première URL par défaut si aucune n'est spécifiée
            publication_url = publication_url or self.publication_urls[0]
            cookies_path, selenium_cookies_path = self._get_cookies_paths(publication_url)

            # Vérifier si l'API est déjà initialisée pour cette URL
            if publication_url in self.api_instances:
                self.api = self.api_instances[publication_url]
                return

            # Vérifier si les cookies existent et sont valides
            if not os.path.exists(selenium_cookies_path) or not self._is_cookie_valid(publication_url) or force:
                self._renew_cookie(self.email, self.password, publication_url)

            # Initialiser l'API avec les cookies
            self.api = Api(
                cookies_path=selenium_cookies_path,
                publication_url=publication_url
            )
            self.api_instances[publication_url] = self.api
            print(f"Successfully authenticated with Substack API for {publication_url}")

        except Exception as e:
            st.error(f"Substack API Authentication Error for {publication_url}: {str(e)}")
            raise

    def _is_cookie_valid(self, publication_url: str) -> bool:
        """
        Test if the stored cookies are still valid for the specified publication by attempting a simple API call.
        :param publication_url: URL of the publication.
        :return: True if cookies are valid, False otherwise.
        """
        try:
            print(f"Checking cookie validity for {publication_url}")
            cookies_path, selenium_cookies_path = self._get_cookies_paths(publication_url)
            if not os.path.exists(selenium_cookies_path):
                return False
            temp_api = Api(cookies_path=selenium_cookies_path, publication_url=publication_url)
            temp_api.get_user_profile()  # Simple API call to test authentication
            return True
        except SubstackAPIException as e:
            print(f"Substack API Authentication Error for {publication_url}: {str(e)}")
            return False

    def retry_on_error(self, func, max_retries=3, delay=1):
        """
        Retry function on Substack API errors with exponential backoff.
        """
        for attempt in range(max_retries):
            try:
                return func()
            except SubstackAPIException as e:
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
                    # Renew cookies on failure
                    publication_url = func.__self__.publication_url if hasattr(func.__self__, 'publication_url') else self.publication_urls[0]
                    self._renew_cookie(self.email, self.password, publication_url)
                    self._initialize_api(publication_url)
                    continue
                raise

    def post(self, title: str, content: str, publish_immediately: bool = False, feature_image: Optional[str] = None, publication_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Creates a Substack post from markdown content, with optional image, for the specified publication.
        :param title: Post title.
        :param content: Markdown content.
        :param publish_immediately: Whether to publish immediately or save as draft.
        :param feature_image: Path to the image file (optional).
        :param publication_url: URL of the publication (optional, defaults to first URL).
        :return: Draft or published post details.
        """
        try:
            self._initialize_api(publication_url)
            # Get user ID
            profile = self.retry_on_error(lambda: self.api.get_user_profile())
            user_id = profile.get("id")
            if not user_id:
                raise ValueError("Could not get user ID from profile")

            # Create post object
            post = Post(title=title, subtitle="", user_id=user_id)

            # Convert markdown to Substack-compatible blocks
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line:
                    if line.startswith("## "):
                        post.add({"type": "heading", "level": 2, "content": line[3:]})
                    elif line.startswith("# "):
                        post.add({"type": "heading", "content": line[2:]})
                    elif line.startswith("!["):
                        # Handle markdown image: ![alt](url)
                        match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
                        if match:
                            alt, src = match.groups()
                            post.add({"type": "captionedImage", "src": src, "caption": alt})
                    else:
                        # Handle bold (**text**), italic (*text*), and links ([text](url)) within paragraphs
                        paragraph_content = []
                        current_text = line
                        current_pos = 0

                        # Process all Markdown patterns iteratively
                        patterns = [
                              (r"\*\*(.*?)\*\*", lambda m: {"content": m.group(1), "marks": [{"type": "strong"}]}),  # Bold
                              (r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", lambda m: {"content": m.group(1), "marks": [{"type": "em"}]}),  # Italic (non-greedy, avoid bold)
                              (r"\[(.*?)\]\((.*?)\)", lambda m: {"content": m.group(1), "marks": [{"type": "link", "href": m.group(2)}]})  # Link
                        ]

                        while current_text:
                            earliest_match = None
                            earliest_start = len(current_text)
                            earliest_content = None
                            earliest_end = 0

                            # Find the earliest match among all patterns
                            for pattern, content_func in patterns:
                                match = re.search(pattern, current_text)
                                if match and match.start() < earliest_start:
                                    earliest_match = match
                                    earliest_start = match.start()
                                    earliest_end = match.end()
                                    earliest_content = content_func(match)

                            if earliest_match:
                                # Add text before the match
                                if earliest_start > 0:
                                    paragraph_content.append({"content": current_text[:earliest_start]})
                                # Add the matched content
                                paragraph_content.append(earliest_content)
                                # Update current_text to continue after the match
                                current_text = current_text[earliest_end:]
                            else:
                                # No more matches, add remaining text
                                paragraph_content.append({"content": current_text})
                                current_text = ""

                        # Add paragraph if content exists
                        if paragraph_content:
                            post.add({"type": "paragraph", "content": paragraph_content})

            # Add local image if provided
            if feature_image and os.path.exists(feature_image):
                image = self.retry_on_error(lambda: self.api.get_image(feature_image))
                post.add({"type": "captionedImage", "src": image.get("url")})

            # Save as draft
            draft = self.retry_on_error(lambda: self.api.post_draft(post.get_draft()))
            draft_id = draft.get("id")
            if not draft_id:
                raise ValueError("Failed to create draft - no ID returned")

            result = {"id": draft_id, "title": title, "status": "draft"}

            # Publish immediately if requested
            if publish_immediately:
                self.retry_on_error(lambda: self.api.prepublish_draft(draft_id))
                published = self.retry_on_error(lambda: self.api.publish_draft(draft_id))
                result["status"] = "published"
                result["url"] = published.get("url", "")

            return result

        except Exception as e:
            st.error(f"Substack API Error (post): {str(e)}")
            return None

    def list_drafts(self, publication_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists all draft posts for the specified publication.
        :param publication_url: URL of the publication (optional, defaults to first URL).
        :return: List of draft post details.
        """
        try:
            self._initialize_api(publication_url)
            drafts = self.retry_on_error(lambda: self.api.get_drafts())
            formatted_drafts = []
            for draft in drafts:
                formatted_drafts.append({
                    "id": draft.get("id"),
                    "title": draft.get("title"),
                    "created_at": draft.get("created_at"),
                    "url": draft.get("url", ""),
                })
            return formatted_drafts
        except Exception as e:
            st.error(f"Substack API Error (list_drafts): {str(e)}")
            return []

    def publish_draft(self, draft_id: str, publication_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Publishes a draft post by ID for the specified publication.
        :param draft_id: ID of the draft to publish.
        :param publication_url: URL of the publication (optional, defaults to first URL).
        :return: Published post details or None if the request fails.
        """
        try:
            self._initialize_api(publication_url)
            self.retry_on_error(lambda: self.api.prepublish_draft(draft_id))
            published = self.retry_on_error(lambda: self.api.publish_draft(draft_id))
            return {
                "id": draft_id,
                "title": published.get("title", ""),
                "status": "published",
                "url": published.get("url", "")
            }
        except Exception as e:
            st.error(f"Substack API Error (publish_draft): {str(e)}")
            return None
