from typing import List, Dict, Any, Optional
import streamlit as st
import re, os

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
        :param content: Article content (Markdown, converted to plain text, max 3000 characters).
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
            latest_version = "202507"
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'X-Restli-Protocol-Version': '2.0.0',
                'LinkedIn-Version': latest_version,
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

                # Initialize image upload
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

            # Convert Markdown content to plain text
            def markdown_to_plain_text(markdown_text: str) -> str:
                """
                Convert Markdown content to plain text by removing Markdown formatting.
                """
                if not markdown_text:
                    return markdown_text

                # Convert Markdown to HTML using markdown2
                html = markdown2.markdown(markdown_text)

                # Remove HTML tags and clean up
                text = re.sub(r'<[^>]+>', '', html)  # Remove HTML tags
                text = re.sub(r'\n\s*\n', '\n', text)  # Remove extra newlines
                text = text.strip()  # Remove leading/trailing whitespace

                return text

            # Prepare the full text content (title + converted content)
            plain_content = markdown_to_plain_text(content)
            full_text = f"{title}\n\n{plain_content}" if title else plain_content

            # Escape special characters that cause issues with LinkedIn API
            def escape_linkedin_text(text):
                """
                Escape special characters that cause LinkedIn API to truncate posts
                Simple backslash escaping for problematic characters
                """
                if not text:
                    return text

                # Characters that need to be escaped with backslash
                chars_to_escape = ['(', ')', '[', ']', '{', '}', '@', '_', '~']

                for char in chars_to_escape:
                    text = text.replace(char, r'\{}'.format(char))

                return text

            # Apply escaping to the full text
            print("Full_text:\n", repr(full_text))
            full_text = escape_linkedin_text(full_text)
            print("Escaped full_text:\n", repr(full_text))

            # Truncate to LinkedIn's limit (3000 characters for commentary)
            if len(full_text) > 3000:
                full_text = full_text[:2997] + "..."
                st.warning(f"Content truncated to 3000 characters (LinkedIn limit)")

            # Build the post body based on content type
            body = {
                'author': self.person_urn,
                'commentary': full_text,
                'visibility': 'PUBLIC',
                'distribution': {
                    'feedDistribution': 'MAIN_FEED'
                },
                'lifecycleState': 'PUBLISHED',
                'isReshareDisabledByAuthor': False
            }

            # Add content based on what we have
            if source_url:
                # Post with article/link sharing
                body['content'] = {
                    'article': {
                        'source': source_url,
                        'title': title or "Shared Article"
                    }
                }
                # Add thumbnail if image was uploaded
                if image_urn:
                    body['content']['article']['thumbnail'] = image_urn

            elif image_urn:
                # Post with image only (no external link)
                body['content'] = {
                    'media': {
                        'title': title or "Image Post",
                        'id': image_urn
                    }
                }
            # If neither source_url nor image, it's a pure text post (no 'content' field needed)

            # Debug: Print the request body (remove in production)
            print("LinkedIn API Request Body:\n", json.dumps(body, indent=2))

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

            # Get post ID from response
            response_data = response.json() if response.content else {}
            post_id = response.headers.get('x-restli-id') or response_data.get('id', 'N/A')

            return {
                'id': post_id,
                'status': 'success',
                'character_count': len(full_text)
            }

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
