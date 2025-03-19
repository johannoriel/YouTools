import streamlit as st
from linkify_it import LinkifyIt
import requests
import logging
from urllib.parse import urlparse
import streamlit.components.v1 as components

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tweet(object):
    def __init__(self, url, embed_str=False):
        if not embed_str:
            # Utiliser l'API oEmbed de Twitter
            # https://publish.twitter.com/oembed
            api = f"https://publish.twitter.com/oembed?url={url}"
            try:
                response = requests.get(api, timeout=10)
                response.raise_for_status()  # Vérifier les erreurs HTTP
                #st.write(response.json())
                self.text = response.json()["html"]
            except (requests.RequestException, ValueError) as e:
                logger.error(
                    f"Erreur lors de la récupération de l'embed pour {url}: {str(e)}")
                self.text = f"<p>Erreur lors de l'affichage du tweet : {str(e)}</p>"
        else:
            self.text = url

    def component(self):
        return components.html(self.text, height=1000)

class Tweet2(object):
    def __init__(self, url, embed_str=False):
        if not embed_str:
            # Utiliser l'API oEmbed de Twitter
            api = f"https://publish.twitter.com/oembed?url={url}"
            try:
                response = requests.get(api, timeout=10)
                response.raise_for_status()
                html = response.json()["html"]
                self.text = html
                # Ajouter un script local pour forcer l'exécution et contourner CSP
                self.text += """
                <script>
                    if (!window.twttr) {
                        (function(d, s, id) {
                            var js, fjs = d.getElementsByTagName(s)[0];
                            if (d.getElementById(id)) return;
                            js = d.createElement(s); js.id = id;
                            js.src = "https://platform.twitter.com/widgets.js";
                            js.async = true;
                            js.charset = "utf-8";
                            fjs.parentNode.insertBefore(js, fjs);
                            window.twttr = (function(t) {
                                t._e = [];
                                t.ready = function(f) { t._e.push(f); };
                                return t;
                            }(window.twttr || {}));
                        }(document, "script", "twitter-wjs"));
                    }
                    // Initialiser les widgets après chargement
                    window.twttr.ready(function(twttr) {
                        twttr.widgets.load();
                    });
                </script>
                """
            except (requests.RequestException, ValueError) as e:
                logger.error(f"Erreur lors de la récupération de l'embed pour {url}: {str(e)}")
                self.text = f"<p>Erreur lors de l'affichage du tweet : {str(e)}</p>"
        else:
            self.text = url

    def component(self):
        return st.components.v1.html(self.text)

def is_twitter_url(url):
    """Vérifie si l'URL est une URL Twitter/X."""
    parsed_url = urlparse(url)
    return parsed_url.netloc in ['twitter.com', 'x.com']


def is_youtube_url(url):
    """Vérifie si l'URL est une URL YouTube."""
    parsed_url = urlparse(url)
    return parsed_url.netloc in ['youtube.com', 'www.youtube.com', 'youtu.be']


def process_lines(lines):
    """Traite les lignes pour identifier URL et blocs Markdown."""
    result = []
    current_markdown = []
    linkify = LinkifyIt()  # Initialisation de linkify-it

    for line in lines:
        # Supprimer les espaces et sauts de ligne
        line = line.strip()
        if not line:
            continue

        # Extraire les URLs avec linkify-it
        matches = linkify.match(line)
        if matches:  # Si une URL est trouvée
            # Vérifier si matches est une liste ou un seul objet Match
            if isinstance(matches, list):
                # Prendre la première URL si plusieurs sont trouvées
                if matches:
                    url = matches[0].url
                else:
                    continue  # Pas d'URL trouvée dans la liste, passer à la ligne suivante
            else:
                # Si c'est un seul objet Match
                url = matches.url

            # Si c'est une URL Twitter, utiliser l'API oEmbed
            if is_twitter_url(url):
                try:
                    tweet = Tweet(url)
                    result.append({
                        "type": "tweet",
                        "component": tweet,
                        "url": url
                    })
                except Exception as e:
                    logger.error(f"Erreur avec le tweet {url}: {str(e)}")
                    result.append(
                        {"type": "error", "message": f"Erreur avec le tweet {url}"})
            elif is_youtube_url(url):
                # Pour une vidéo YouTube, utiliser st.video
                result.append({
                    "type": "youtube",
                    "url": url
                })
            else:
                # Pour les autres URL, utiliser une iframe
                result.append({
                    "type": "web",
                    "url": url
                })

            # Ajouter le Markdown accumulé s'il y en a
            if current_markdown:
                result.append(
                    {"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
        else:
            # Accumuler les lignes Markdown
            current_markdown.append(line)

    # Ajouter le dernier bloc Markdown s'il y en a
    if current_markdown:
        result.append(
            {"type": "markdown", "content": "\n".join(current_markdown)})

    return result


def main():
    st.title("Générateur de Présentation Rapide")

    # Zone de préparation : saisie de texte
    st.header("1. Zone de Préparation")
    input_text = st.text_area(
        "Collez vos lignes (URL ou texte Markdown) ici, une par ligne :", height=200)

    if st.button("Generate"):
        st.header("2. Présentation Générée")
        if input_text:
            lines = input_text.split("\n")
            processed_items = process_lines(lines)

            for item in processed_items:
                if item["type"] == "markdown":
                    st.markdown(item["content"])
                elif item["type"] == "tweet":
                    item["component"].component()
                elif item["type"] == "youtube":
                    st.video(item["url"])
                elif item["type"] == "web":
                    st.components.v1.iframe(
                        item["url"], height=400, scrolling=True)
                elif item["type"] == "error":
                    st.error(item["message"])
                st.markdown("---")
        else:
            st.warning(
                "Veuillez entrer du texte avant de générer la présentation.")


if __name__ == "__main__":
    main()
