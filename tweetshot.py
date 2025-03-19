import argparse
import logging
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
import subprocess
import random
import time
import urllib.parse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TweetScreenshot:
    def __init__(self):
        self.driver = None
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        ]

    def find_chromedriver(self):
        """Trouve le chemin du chromedriver"""
        try:
            chromedriver_path = subprocess.check_output(
                ['which', 'chromedriver']).decode().strip()
            logger.debug(
                f"Chromedriver trouvé dans le PATH: {chromedriver_path}")
            return chromedriver_path
        except subprocess.CalledProcessError:
            logger.debug("Chromedriver non trouvé dans le PATH")

        possible_paths = [
            '/usr/bin/chromedriver',
            '/usr/local/bin/chromedriver',
            '/snap/bin/chromedriver',
            '/usr/lib/chromium/chromedriver'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logger.debug(f"Chromedriver trouvé à: {path}")
                return path
        raise FileNotFoundError(
            "Chromedriver non trouvé dans les chemins standards")

    def setup_driver(self):
        """Configure et initialise le driver Chrome"""
        try:
            logger.debug("Configuration des options Chrome...")
            chrome_options = Options()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            user_agent = random.choice(self.user_agents)
            chrome_options.add_argument(f'--user-agent={user_agent}')
            logger.debug(f"Utilisation de l'agent utilisateur: {user_agent}")

            chrome_paths = ['/usr/bin/chromium',
                            '/usr/bin/chromium-browser', '/snap/bin/chromium']
            chrome_path = next(
                (path for path in chrome_paths if os.path.exists(path)), None)
            if not chrome_path:
                raise FileNotFoundError(
                    "Chromium n'est pas trouvé dans les chemins standards")
            chrome_options.binary_location = chrome_path

            chromedriver_path = self.find_chromedriver()
            logger.debug(f"Utilisation du chromedriver: {chromedriver_path}")

            service = Service(executable_path=chromedriver_path,
                              service_log_path=os.devnull)
            self.driver = webdriver.Chrome(
                service=service, options=chrome_options)
            logger.info("Driver Chrome initialisé avec succès")
        except WebDriverException as e:
            logger.error(
                f"Erreur lors de l'initialisation du driver: {str(e)}")
            raise

    def remove_overlays(self):
        """Supprime ou masque les overlays RGPD et de connexion"""
        try:
            logger.debug("Tentative de suppression des overlays...")
            overlay_selectors = [
                'div[role="dialog"]',
                'div[data-testid="sheetDialog"]',
                'div.css-1dbjc4n.r-1jgb5lz',
                'div#layers > div:not([data-testid="primaryColumn"])',
            ]
            hide_script = """
                var selectors = arguments[0];
                selectors.forEach(function(selector) {
                    var elements = document.querySelectorAll(selector);
                    elements.forEach(function(el) {
                        el.style.display = 'none';
                    });
                });
            """
            self.driver.execute_script(hide_script, overlay_selectors)
            time.sleep(1)
            logger.info("Overlays masqués avec succès")
        except Exception as e:
            logger.warning(
                f"Erreur lors de la suppression des overlays: {str(e)}")

    def adjust_window_size(self, element, margin=50):
        """Ajuste la taille de la fenêtre pour inclure tout l'élément avec une marge"""
        try:
            element_height = element.size['height']
            element_width = element.size['width']
            element_y = element.location['y']

            current_window_height = self.driver.execute_script(
                "return window.innerHeight;")
            required_height = element_y + element_height + margin

            if required_height > current_window_height:
                self.driver.set_window_size(
                    element_width + margin * 2, required_height)
                logger.debug(
                    f"Fenêtre redimensionnée à {element_width + margin * 2}x{required_height}")
            else:
                self.driver.set_window_size(
                    element_width + margin * 2, current_window_height)
                logger.debug(
                    f"Fenêtre ajustée à {element_width + margin * 2}x{current_window_height}")

            self.driver.execute_script(
                "arguments[0].scrollIntoView(true);", element)
            time.sleep(0.5)
        except Exception as e:
            logger.warning(
                f"Erreur lors de l'ajustement de la fenêtre: {str(e)}")

    def get_tweet_embed(self, url):
        """Récupère l'embed HTML (blockquote) via publish.twitter.com"""
        try:
            # Construire l'URL pour publish.twitter.com
            encoded_url = urllib.parse.quote(url)
            publish_url = f"https://publish.twitter.com/?query={encoded_url}&widget=Video"

            logger.info(f"Chargement de l'embed via: {publish_url}")
            self.driver.get(publish_url)

            # Attendre que le code embed soit chargé (dans un élément avec classe "EmbedCode-code")
            embed_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(
                    (By.CLASS_NAME, "EmbedCode-code"))
            )

            # Récupérer le texte HTML (blockquote et script)
            embed_html = embed_element.text.strip()
            logger.info("Embed HTML récupéré avec succès")
            return embed_html
        except TimeoutException:
            logger.error(
                "Timeout : l'embed n'a pas été trouvé dans le délai imparti")
            return None
        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération de l'embed: {str(e)}")
            return None

    def has_video(self, tweet_element):
        """Vérifie si le tweet contient une vidéo"""
        try:
            tweet_element.find_element(By.TAG_NAME, "video")
            return "video_present"
        except NoSuchElementException:
            return ""

    def capture_tweet(self, url, margin=50):
        """Capture un screenshot du tweet entier et retourne l'embed HTML et une indication vidéo"""
        try:
            if not self.driver:
                self.setup_driver()

            # Récupérer l'embed HTML en premier (avant de charger le tweet)
            embed_html = self.get_tweet_embed(url)

            logger.info(f"Ouverture de l'URL: {url}")
            self.driver.get(url)

            logger.debug("Attente de l'élément <article>...")
            tweet_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )

            self.remove_overlays()
            self.adjust_window_size(tweet_element, margin)

            # Capturer le screenshot du tweet entier
            screenshot_data = tweet_element.screenshot_as_png
            logger.info("Screenshot capturé en mémoire avec succès")

            # Vérifier si une vidéo est présente
            video_indicator = self.has_video(tweet_element)

            return screenshot_data, embed_html, video_indicator

        except TimeoutException:
            logger.error(
                "Timeout : l'élément <article> n'a pas été trouvé dans le délai imparti")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la capture du tweet: {str(e)}")
            raise
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None

    def capture_tweet_to_file(self, url, output_path="tweetshot.png", margin=50):
        """Capture le tweet, sauvegarde le screenshot, et affiche l'embed HTML si demandé"""
        try:
            screenshot_data, embed_html, video_indicator = self.capture_tweet(
                url, margin)
            with open(output_path, "wb") as f:
                f.write(screenshot_data)
            logger.info(f"Screenshot sauvegardé à: {output_path}")

            # En ligne de commande, afficher l'embed HTML et l'indicateur vidéo
            if embed_html:
                print("Embed HTML:")
                print(embed_html)
            if video_indicator:
                print(f"Indicateur vidéo: {video_indicator}")
            return embed_html, video_indicator  # Retourne pour usage externe
        except Exception as e:
            logger.error(
                f"Erreur lors de la sauvegarde du screenshot: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Capture un screenshot et un embed HTML d’un tweet')
    parser.add_argument('url', type=str, help='URL du tweet à capturer')
    parser.add_argument('--margin', type=int, default=50,
                        help='Marge autour du tweet en pixels')
    args = parser.parse_args()

    tweet_screenshot = TweetScreenshot()
    embed_html, video_indicator = tweet_screenshot.capture_tweet_to_file(
        args.url, margin=args.margin)
    if video_indicator:
        print(f"Indicateur vidéo: {video_indicator}")


if __name__ == "__main__":
    main()
