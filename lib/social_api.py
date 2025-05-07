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

    def post(self, title: str, content: str, publish_immediately: bool = True) -> Optional[Dict[str, Any]]:
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
                    'status': 'published' if publish_immediately else 'draft'
                }]
            }
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
        self.base_url = "https://api.linkedin.com/v2"
        # self._get_access_token()
        self.access_token = config['common']['linkedin_access_token']

    def _get_access_token(self) -> Optional[str]:
        """
        Récupère le token d'accès LinkedIn via OAuth2.
        """
        try:
            auth_url = "https://www.linkedin.com/oauth/v2/accessToken"
            payload = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            response = requests.post(auth_url, data=payload, headers=headers)
            response.raise_for_status()  # Lève une exception si le statut n'est pas 200
            return response.json().get('access_token')
        except Exception as e:
            print(f"LinkedIn API Error (get_access_token): {str(e)}")
            return None

    def search_posts(self, query: str, max_results: int = 10, language: str = "fr") -> List[Dict[str, Any]]:
        """
        Recherche des posts sur LinkedIn en fonction des mots-clés.
        :param query: Mots-clés de recherche
        :param max_results: Nombre maximum de posts à récupérer
        :param language: Langue des posts à rechercher (par défaut "fr")
        :return: Liste des posts trouvés
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'X-Restli-Protocol-Version': '2.0.0'
            }
            params = {
                'q': query,
                'count': max_results,
                'sort': 'relevance',
                'locale.language': language
            }
            response = requests.get(
                f"{self.base_url}/search", headers=headers, params=params)
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
            print(f"LinkedIn API Error (search_posts): {str(e)}")
            return []

    def get_post_comments(self, post_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les commentaires d'un post LinkedIn.
        :param post_id: ID du post
        :param max_results: Nombre maximum de commentaires à récupérer
        :return: Liste des commentaires
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'X-Restli-Protocol-Version': '2.0.0'
            }
            response = requests.get(
                f"{self.base_url}/socialActions/{post_id}/comments", headers=headers)
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
            print(f"LinkedIn API Error (get_post_comments): {str(e)}")
            return []

    def post_comment(self, post_id: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Poste un commentaire sur un post LinkedIn.
        :param post_id: ID du post
        :param text: Texte du commentaire
        :return: Réponse de l'API
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'X-Restli-Protocol-Version': '2.0.0'
            }
            body = {
                'actor': f"urn:li:person:{self.client_id}",
                'message': {
                    'text': text
                },
                'object': f"urn:li:share:{post_id}"
            }
            response = requests.post(
                f"{self.base_url}/socialActions/{post_id}/comments", headers=headers, json=body)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"LinkedIn API Error (post_comment): {str(e)}")
            return None
