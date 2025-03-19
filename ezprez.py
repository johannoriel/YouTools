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
import time
from streamlit_shortcuts import button, add_keyboard_shortcuts

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

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "---":
            if current_markdown:
                result.append({"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue

        if line == "--" and i > 0 and i + 1 < len(lines):
            prev_item = None
            if current_markdown:
                prev_item = {"type": "markdown", "content": "\n".join(current_markdown)}
                current_markdown = []
            elif result:
                prev_item = result.pop()

            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            next_line = lines[i].strip()
            next_item = parse_single_line(next_line, directories, linkify)
            if prev_item and next_item:
                result.append({"type": "group", "items": [prev_item, next_item]})
            i += 1
            continue

        if not line and i + 1 < len(lines) and lines[i + 1].strip() in ["---", "--"]:
            i += 1
            continue

        item = parse_single_line(line, directories, linkify)
        if item["type"] == "markdown":
            current_markdown.append(item["content"])
        else:
            if current_markdown:
                result.append({"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
            result.append(item)
        i += 1

    if current_markdown:
        result.append({"type": "markdown", "content": "\n".join(current_markdown)})

    return result

def parse_single_line(line, directories, linkify):
    extensions = {'.jpg': 'image', '.png': 'image', '.mp4': 'video', '.flv': 'video'}
    for ext, content_type in extensions.items():
        if line.endswith(ext):
            return {"type": content_type, "content": line}

    md_file_match = re.match(r'!?\[(.*?)\]\((file://.*?)\)', line)
    md_link_match = re.match(r'!?\[(.*?)\]\((https?://.*?)\)', line)
    md_image_match = re.match(r'!\[(?:\|(\d+))?(.*?)\]\(([^h].*?)\)', line)

    if md_file_match:
        file_path = md_file_match.group(2).replace('file://', '')
        ext = Path(file_path).suffix.lower()
        content_type = extensions.get(ext, 'unknown')
        if content_type in ['image', 'video']:
            title = md_file_match.group(1).strip()
            return {"type": content_type, "content": file_path, "title": title if title else None}

    if md_image_match:
        size = md_image_match.group(1)
        title = md_image_match.group(2).strip()
        filepath = md_image_match.group(3)
        ext = Path(filepath).suffix.lower()
        if ext in extensions:
            item = {"type": extensions[ext], "content": filepath, "title": title if title else None}
            if size:
                item["size"] = int(size)
            return item

    if md_link_match:
        url = md_link_match.group(2)
        title = md_link_match.group(1).strip()
        if is_twitter_url(url):
            tweet = Tweet(url)
            return {"type": "tweet", "component": tweet, "url": url, "title": title if title else None}
        elif is_youtube_url(url):
            return {"type": "youtube", "url": url, "title": title if title else None}
        else:
            return {"type": "web", "url": url, "title": title if title else None}

    matches = linkify.match(line)
    if matches:
        url = matches[0].url
        if is_twitter_url(url):
            tweet = Tweet(url)
            return {"type": "tweet", "component": tweet, "url": url, "title": None}
        elif is_youtube_url(url):
            return {"type": "youtube", "url": url, "title": None}
        else:
            return {"type": "web", "url": url, "title": None}

    return {"type": "markdown", "content": line}

def display_item(item, directories, is_presentation=False):
    if item["type"] == "markdown":
        st.markdown(item["content"])
    elif item["type"] == "tweet":
        if item["title"]:
            st.subheader(item["title"])
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        item["component"].component()
        st.markdown("</div>", unsafe_allow_html=True)
    elif item["type"] == "youtube":
        if item["title"]:
            st.subheader(item["title"])
        st.video(item["url"])
    elif item["type"] == "image":
        filepath = find_file(item["content"], directories) if not item["content"].startswith('/') else item["content"]
        if item["title"]:
            st.subheader(item["title"])
        max_height = item.get("size", 500 if is_presentation else 200)
        st.image(filepath, use_container_width=True)
        st.markdown(f"""
            <style>
            img {{
                max-height: {max_height}px;
                object-fit: contain;
                overflow-y: auto;
            }}
            </style>
        """, unsafe_allow_html=True)
    elif item["type"] == "video":
        filepath = find_file(item["content"], directories) if not item["content"].startswith('/') else item["content"]
        if item["title"]:
            st.subheader(item["title"])
        st.video(filepath)
    elif item["type"] == "web":
        if item["title"]:
            st.subheader(item["title"])
        image_path = url_to_image(item["url"])
        if image_path:
            st.image(image_path, use_container_width=True)
            os.remove(image_path)
        else:
            st.error("Impossible de convertir l'URL en image")
    elif item["type"] == "group":
        col1, col2 = st.columns(2)
        with col1:
            display_item(item["items"][0], directories, is_presentation)
        with col2:
            display_item(item["items"][1], directories, is_presentation)

def main():
    if 'presentation_mode' not in st.session_state:
        st.session_state['presentation_mode'] = False

    if st.session_state['presentation_mode']:
        st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title=None)
    else:
        st.set_page_config(layout="centered", initial_sidebar_state="expanded", page_title="Présentation Rapide")

    if not st.session_state['presentation_mode']:
        st.title("Présentation Rapide")

    directories = load_directories()

    with st.sidebar:
        st.header("Préparation")
        input_text = st.text_area("Collez vos lignes ici :", height=200, key="input_text")

        col1, col2 = st.columns(2)
        with col1:
            preview = button("Preview", "Ctrl+P", lambda: st.session_state.update({'slides': process_lines(st.session_state.get('input_text', '').split("\n"), directories)}), hint=True)
        with col2:
            launch = button("Launch", "Ctrl+Enter", lambda: st.session_state.update({'presentation_mode': True, 'current_slide': 0, 'input_text': st.session_state.get('input_text', ''), 'slides': process_lines(st.session_state.get('input_text', '').split("\n"), directories)}), hint=True)

        if st.session_state['presentation_mode']:
            st.header("Navigation")
            col1, col2 = st.columns(2)
            with col1:
                button("Précédent", "ArrowLeft", lambda: [st.session_state.update({'current_slide': max(0, st.session_state['current_slide'] - 1)})], hint=True)
            with col2:
                button("Suivant", "ArrowRight", lambda: [st.session_state.update({'current_slide': min(len(st.session_state['slides']) - 1, st.session_state['current_slide'] + 1)})], hint=True)

            col3, col4 = st.columns(2)
            with col3:
                button("Première", "Home", lambda: st.session_state.update({'current_slide': 0}), hint=True)
            with col4:
                button("Dernière", "End", lambda: st.session_state.update({'current_slide': len(st.session_state['slides']) - 1}), hint=True)

            button("Exit", "Escape", lambda: st.session_state.update({'presentation_mode': False}), hint=True)

    if not st.session_state['presentation_mode'] and 'slides' in st.session_state:
        st.header("Aperçu")
        for item in st.session_state['slides']:
            display_item(item, directories, is_presentation=False)
            st.markdown("---")

    if st.session_state['presentation_mode']:
        if 'slides' not in st.session_state or not st.session_state['slides']:
            st.warning("Veuillez entrer du contenu et générer les slides avant de lancer la présentation.")
            st.session_state['presentation_mode'] = False
            return

        slides = st.session_state['slides']
        if 'current_slide' not in st.session_state:
            st.session_state['current_slide'] = 0

        current = st.session_state['current_slide']

        if slides:
            #st.subheader(f"Slide {current + 1}/{len(slides)}")
            display_item(slides[current], directories, is_presentation=True)

        add_keyboard_shortcuts({
            'ArrowLeft': 'Précédent',
            'ArrowRight': 'Suivant',
            'Home': 'Première',
            'End': 'Dernière',
            'Ctrl+Enter': 'Launch',
            'Ctrl+P': 'Preview',
            'Escape': 'Exit'
        })

if __name__ == "__main__":
    main()
