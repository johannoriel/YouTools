from typing import List, Dict, Any, Optional
import streamlit as st
import re, os

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
