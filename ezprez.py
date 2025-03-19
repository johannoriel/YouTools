import streamlit as st
from linkify_it import LinkifyIt
import requests
import logging
from urllib.parse import urlparse
import streamlit.components.v1 as components
import configparser
from pathlib import Path
import re
import subprocess
import tempfile
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les répertoires depuis le fichier .ini
def load_directories():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.get('Paths', 'directories', fallback='').split('\n')

# Trouver un fichier dans les répertoires prédéfinis
def find_file(filename, directories):
    for directory in directories:
        full_path = Path(directory) / filename
        if full_path.exists():
            return str(full_path)
    return None

# Convertir une URL en image avec wkhtmltoimage
def url_to_image(url):
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            output_file = tmp_file.name
            subprocess.run(['wkhtmltoimage', url, output_file], check=True)
            return output_file
    except Exception as e:
        logger.error(f"Erreur lors de la conversion de l'URL {url} en image : {str(e)}")
        return None

class Tweet(object):
    def __init__(self, url, embed_str=False):
        if not embed_str:
            api = f"https://publish.twitter.com/oembed?url={url}"
            try:
                response = requests.get(api, timeout=10)
                response.raise_for_status()
                data = response.json()
                self.text = data["html"]
                self.title = data.get("title", url)
            except (requests.RequestException, ValueError) as e:
                logger.error(f"Erreur lors de la récupération du tweet {url}: {str(e)}")
                self.text = f"<p>Erreur tweet: {str(e)}</p>"
                self.title = "Erreur"
        else:
            self.text = url
            self.title = url

    def component(self):
        return components.html(self.text, height=600)

def is_twitter_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc in ['twitter.com', 'x.com']

def is_youtube_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc in ['youtube.com', 'www.youtube.com', 'youtu.be']

def process_lines(lines, directories):
    result = []
    current_markdown = []
    linkify = LinkifyIt()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Détection des fichiers locaux simples
        extensions = {'.jpg': 'image', '.png': 'image', '.mp4': 'video', '.flv': 'video'}
        for ext, content_type in extensions.items():
            if line.endswith(ext):
                result.append({"type": content_type, "content": line})
                if current_markdown:
                    result.append({"type": "markdown", "content": "\n".join(current_markdown)})
                    current_markdown = []
                continue

        # Détection des liens Markdown
        md_file_match = re.match(r'!?\[(.*?)\]\((file://.*?)\)', line)
        md_link_match = re.match(r'!?\[(.*?)\]\((https?://.*?)\)', line)
        md_image_match = re.match(r'!\[(.*?)\]\(([^h].*?)\)', line)  # Image relative sans http/file

        if md_file_match:  # Fichier local absolu
            file_path = md_file_match.group(2).replace('file://', '')
            ext = Path(file_path).suffix.lower()
            content_type = extensions.get(ext, 'unknown')
            if content_type in ['image', 'video']:
                result.append({"type": content_type, "content": file_path, "title": md_file_match.group(1) or Path(file_path).name})
            if current_markdown:
                result.append({"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
            continue

        if md_image_match:  # Image relative
            filepath = md_image_match.group(2)
            ext = Path(filepath).suffix.lower()
            if ext in extensions:
                result.append({"type": extensions[ext], "content": filepath, "title": md_image_match.group(1) or Path(filepath).name})
            if current_markdown:
                result.append({"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
            continue

        if md_link_match:  # URL en Markdown
            url = md_link_match.group(2)
            title = md_link_match.group(1) or url
            if is_twitter_url(url):
                tweet = Tweet(url)
                result.append({"type": "tweet", "component": tweet, "url": url, "title": title})
            elif is_youtube_url(url):
                result.append({"type": "youtube", "url": url, "title": title})
            else:
                result.append({"type": "web", "url": url, "title": title})
            if current_markdown:
                result.append({"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
            continue

        # URLs simples
        matches = linkify.match(line)
        if matches:
            url = matches[0].url
            if is_twitter_url(url):
                tweet = Tweet(url)
                result.append({"type": "tweet", "component": tweet, "url": url, "title": url})
            elif is_youtube_url(url):
                result.append({"type": "youtube", "url": url, "title": url})
            else:
                result.append({"type": "web", "url": url, "title": url})
            if current_markdown:
                result.append({"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
            continue

        current_markdown.append(line)

    if current_markdown:
        result.append({"type": "markdown", "content": "\n".join(current_markdown)})

    return result

def display_item(item, directories):
    if item["type"] == "markdown":
        st.markdown(item["content"])
    elif item["type"] == "tweet":
        st.subheader(item["title"])
        item["component"].component()
    elif item["type"] == "youtube":
        st.subheader(item["title"])
        st.video(item["url"])
    elif item["type"] == "image":
        filepath = find_file(item["content"], directories) if not item["content"].startswith('/') else item["content"]
        st.subheader(item["title"])
        st.image(filepath)
    elif item["type"] == "video":
        filepath = find_file(item["content"], directories) if not item["content"].startswith('/') else item["content"]
        st.subheader(item["title"])
        st.video(filepath)
    elif item["type"] == "web":
        st.subheader(item["title"])
        image_path = url_to_image(item["url"])
        if image_path:
            st.image(image_path)
            os.remove(image_path)  # Nettoyage
        else:
            st.error("Impossible de convertir l'URL en image")

def main():
    st.title("Présentation Rapide")
    directories = load_directories()

    # Zone de préparation
    st.header("1. Zone de Préparation")
    input_text = st.text_area("Collez vos lignes ici :", height=200)

    col1, col2 = st.columns(2)
    with col1:
        preview = st.button("Preview")
    with col2:
        launch = st.button("Launch")

    if preview:
        st.header("2. Aperçu")
        if input_text:
            lines = input_text.split("\n")
            processed_items = process_lines(lines, directories)
            for item in processed_items:
                display_item(item, directories)
                st.markdown("---")

    if launch or ('presentation_mode' in st.session_state and st.session_state['presentation_mode']):
        if not input_text:
            st.warning("Veuillez entrer du contenu avant de lancer la présentation.")
            return
        st.session_state['presentation_mode'] = True
        lines = input_text.split("\n")
        slides = process_lines(lines, directories)

        if 'current_slide' not in st.session_state:
            st.session_state['current_slide'] = 0

        current = st.session_state['current_slide']

        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if st.button("Précédent") and current > 0:
                st.session_state['current_slide'] -= 1
        with col3:
            if st.button("Suivant") and current < len(slides) - 1:
                st.session_state['current_slide'] += 1

        if slides:
            st.subheader(f"Slide {current + 1}/{len(slides)}")
            display_item(slides[current], directories)

        st.markdown("""
            <script>
            document.addEventListener('keydown', function(e) {
                if (e.key === 'ArrowLeft') {
                    document.getElementById('prev').click();
                }
                if (e.key === 'ArrowRight') {
                    document.getElementById('next').click();
                }
            });
            </script>
            <button id="prev" style="display:none" onclick="streamlitCallback('Précédent')">Prev</button>
            <button id="next" style="display:none" onclick="streamlitCallback('Suivant')">Next</button>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
