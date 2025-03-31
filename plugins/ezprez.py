# Quick Presentation App
# This application is designed to create and display simple presentations from text input.
# Intent: Provide a lightweight tool for quickly assembling slides from Markdown-like syntax,
# supporting various media types (images, videos, tweets, YouTube links, web screenshots).
#
# Key Features:
# - Two modes: Preview (centered layout, all slides visible) and Presentation (wide layout, one slide at a time).
# - Input via a sidebar text area, always accessible.
# - Slide separation with '---'; column grouping with '--' within a slide.
# - Supports custom image heights (e.g., ![|725](path)), centered content display, and empty title handling.
# - Media handling: Local files (jpg, png, mp4, flv), Twitter embeds, YouTube videos, and web screenshots.
# - Navigation: Keyboard shortcuts (Ctrl+P: Preview, Ctrl+Enter: Launch, ArrowLeft/Right: Prev/Next, Home/End: First/Last, Escape: Exit).
# - Filters out empty slides and ignores orphan empty lines around separators.
#
# Specs:
# - Built with Streamlit for a web-based interface.
# - Uses wkhtmltoimage for web screenshots and config.ini for directory paths.
# - Designed for simplicity and speed, with minimal UI clutter in presentation mode.

from global_vars import translations, t
from app import Plugin
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
from streamlit_shortcuts import button
from urllib.parse import unquote
import random
import time


# Configure logging for debugging purposes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "ezprez_tab": "Ezprez Presentation",
    "ezprez_header": "Quick Presentation",
    "ezprez_preparation_header": "Preparation",
    "ezprez_input_label": "Paste your lines here:",
    "ezprez_preview_button": "Preview",
    "ezprez_launch_button": "Launch",
    "ezprez_navigation_header": "Navigation",
    "ezprez_previous_button": "Previous",
    "ezprez_next_button": "Next",
    "ezprez_first_button": "First",
    "ezprez_last_button": "Last",
    "ezprez_skip_button": "Skip",
    "ezprez_exit_button": "Exit",
    "ezprez_green_bg_label": "Green Background",
    "ezprez_vertical_center_label": "Center Vertically",
    "ezprez_no_content_warning": "Please enter content and generate slides before launching the presentation.",
    "ezprez_config_directories_label": "Directories for file search",
    "ezprez_config_directories_default": "Enter directories separated by newlines",
})

translations["fr"].update({
    "ezprez_tab": "Présentation Ezprez",
    "ezprez_header": "Présentation Rapide",
    "ezprez_preparation_header": "Préparation",
    "ezprez_input_label": "Collez vos lignes ici :",
    "ezprez_preview_button": "Aperçu",
    "ezprez_launch_button": "Lancer",
    "ezprez_navigation_header": "Navigation",
    "ezprez_previous_button": "Précédent",
    "ezprez_next_button": "Suivant",
    "ezprez_first_button": "Premier",
    "ezprez_last_button": "Dernier",
    "ezprez_skip_button": "Passer",
    "ezprez_exit_button": "Quitter",
    "ezprez_green_bg_label": "Fond Vert",
    "ezprez_vertical_center_label": "Centrer Verticalement",
    "ezprez_no_content_warning": "Veuillez entrer du contenu et générer des diapositives avant de lancer la présentation.",
    "ezprez_config_directories_label": "Répertoires pour la recherche de fichiers",
    "ezprez_config_directories_default": "Entrez les répertoires séparés par des sauts de ligne",
})


def find_file(filename, directories):
    """Searches for a file in the specified directories and returns its full path if found.
    Handles both encoded (e.g., %20) and decoded (e.g., space) filenames."""
    from urllib.parse import unquote
    # Décoder le nom de fichier (ex. %20 -> espace)
    decoded_filename = unquote(filename)

    for directory in directories:
        # Tester avec le nom décodé
        full_path_decoded = Path(directory) / decoded_filename
        if full_path_decoded.exists():
            return str(full_path_decoded)

        # Tester avec le nom original (encodé)
        full_path_encoded = Path(directory) / filename
        if full_path_encoded.exists():
            return str(full_path_encoded)

    return None


# Convert a web URL to an image using wkhtmltoimage


def url_to_image(url):
    """Converts a webpage URL to a PNG image using wkhtmltoimage and returns the file path."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            output_file = tmp_file.name
            subprocess.run(['wkhtmltoimage', url, output_file], check=True)
            return output_file
    except Exception as e:
        logger.error(f"Error converting URL {url} to image: {str(e)}")
        return None

# Class to handle Twitter embeds (modified)


class Tweet(object):
    """Handles fetching and embedding a tweet from a Twitter URL with optional custom height."""

    def __init__(self, url, embed_str=False, height=800):
        if not embed_str:
            api = f"https://publish.twitter.com/oembed?hide_thread=true&url={url}&widget=Video"
            try:
                response = requests.get(api, timeout=10)
                response.raise_for_status()
                data = response.json()
                self.text = f'<div class="myTweet">{data["html"]}</div>'
                self.title = data.get("title", url)
            except (requests.RequestException, ValueError) as e:
                logger.error(f"Error fetching tweet {url}: {str(e)}")
                self.text = f'<div class="myTweet"><p>Tweet error: {str(e)}</p></div>'
                self.title = "Error"
        else:
            self.text = f'<div class="myTweet">{url}</div>'
            self.title = url
        self.height = height

    def component(self):
        """Returns the tweet as an HTML component for Streamlit with specified height."""
        return components.html(self.text, height=self.height)

# Check if a URL is a Twitter link


def is_twitter_url(url):
    """Returns True if the URL is from Twitter or X."""
    parsed_url = urlparse(url)
    return parsed_url.netloc in ['twitter.com', 'x.com']

# Check if a URL is a YouTube link


def is_youtube_url(url):
    """Returns True if the URL is from YouTube."""
    parsed_url = urlparse(url)
    return parsed_url.netloc in ['youtube.com', 'www.youtube.com', 'youtu.be']

# Process input lines into slides


def process_lines(lines, directories):
    """
    Processes a list of input lines into slides based on specific rules:
    - '---' separates slides.
    - '--' creates an empty group item to be filled later with previous and next items.
    - Empty lines around separators are ignored.
    - Empty slides are filtered out.
    - Handles multiple items returned by parse_single_line (e.g., from included files).
    - Ignores content between %% and %% as comments.
    """
    result = []
    current_markdown = []
    linkify = LinkifyIt()

    # Pré-traitement pour supprimer les commentaires
    filtered_lines = []
    in_comment = False

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("%%") and stripped_line.endswith("%%"):
            continue
        if stripped_line == "%%":
            in_comment = not in_comment
            continue
        if not in_comment:
            filtered_lines.append(line)

    if in_comment:
        logger.warning("Unclosed comment block detected (missing closing %%)")

    # Première passe : construire la liste initiale avec des groupes vides pour '--'
    i = 0
    while i < len(filtered_lines):
        line = filtered_lines[i].strip()

        # Handle slide separator
        if line == "---":
            if current_markdown and "\n".join(current_markdown).strip():
                result.append(
                    {"type": "markdown", "content": "\n".join(current_markdown)})
            current_markdown = []
            i += 1
            while i < len(filtered_lines) and not filtered_lines[i].strip():
                i += 1
            continue

        # Handle column separator
        if line == "--":
            if current_markdown and "\n".join(current_markdown).strip():
                result.append(
                    {"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
            result.append({"type": "group", "items": []})  # Groupe vide
            i += 1
            while i < len(filtered_lines) and not filtered_lines[i].strip():
                i += 1
            continue

        # Ignore empty lines before separators
        if not line and i + 1 < len(filtered_lines) and filtered_lines[i + 1].strip() in ["---", "--"]:
            i += 1
            continue

        # Process individual line
        items = parse_single_line(line, directories, linkify)
        if isinstance(items, list):
            if current_markdown and "\n".join(current_markdown).strip():
                result.append(
                    {"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
            result.extend(items)
        else:
            if items["type"] == "markdown":
                current_markdown.append(items["content"])
            else:
                if current_markdown and "\n".join(current_markdown).strip():
                    result.append(
                        {"type": "markdown", "content": "\n".join(current_markdown)})
                    current_markdown = []
                result.append(items)
        i += 1

    if current_markdown and "\n".join(current_markdown).strip():
        result.append(
            {"type": "markdown", "content": "\n".join(current_markdown)})

    # Deuxième passe : remplir les groupes vides
    final_result = []
    i = 0
    while i < len(result):
        if result[i]["type"] == "group" and not result[i]["items"]:
            # Vérifier qu'il y a un élément avant et après
            if i > 0 and i + 1 < len(result):
                prev_item = final_result.pop()  # Retirer l'élément précédent
                next_item = result[i + 1]       # Prendre l'élément suivant
                result[i]["items"] = [prev_item, next_item]
                final_result.append(result[i])
                i += 2  # Sauter l'élément suivant déjà utilisé
            else:
                # Si pas d'éléments avant ou après, ignorer le groupe vide
                i += 1
        else:
            final_result.append(result[i])
            i += 1

    return final_result


def parse_single_line(line, directories, linkify):
    """
    Parses a single line into an item or list of items based on its content.
    Updated to detect online images from URLs.
    """
    extensions = {'.jpg': 'image', '.png': 'image', '.jpeg': 'image', '.gif': 'image',
                  '.mp4': 'video', '.flv': 'video'}

    # Check for local or online files by extension
    for ext, content_type in extensions.items():
        if line.strip().endswith(ext):
            # If it’s a URL (starts with http/https), treat as online content
            if line.strip().startswith(('http://', 'https://')):
                return {"type": content_type, "content": line.strip(), "is_online": True}
            # Otherwise, assume local file
            return {"type": content_type, "content": line.strip()}

    # Existing patterns (Obsidian, Markdown files, etc.) remain unchanged until image match
    obsidian_pattern = re.match(r'obsidian://open\?vault=.*?&file=(.*)', line)
    md_file_pattern = re.match(r'(.+\.md)$', line.strip())
    md_link_pattern = re.match(r'\[(.*?)\]\((.+\.md)\)', line)
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

    if obsidian_pattern:
        file_path = unquote(obsidian_pattern.group(1))
        # Add .md to Obsidian link
        full_path = find_file(file_path + '.md', directories)
        if full_path:
            return include_file_content(full_path, directories, linkify)
        return {"type": "markdown", "content": f"File not found: {file_path}.md"}

    if md_file_pattern:
        file_path = md_file_pattern.group(1)
        full_path = find_file(file_path, directories)
        if full_path:
            return include_file_content(full_path, directories, linkify)
        return {"type": "markdown", "content": f"File not found: {file_path}"}

    if md_link_pattern:
        title = md_link_pattern.group(1).strip(
        ) if md_link_pattern.group(1).strip() else None
        file_path = md_link_pattern.group(2)
        full_path = find_file(file_path, directories)
        if full_path:
            items = include_file_content(full_path, directories, linkify)
            if title and items:
                # Apply title to the first item if provided
                items[0]["title"] = title
            return items
        return {"type": "markdown", "content": f"File not found: {file_path}"}

    if md_image_match:
        size = md_image_match.group(1)
        title = md_image_match.group(2).strip()
        filepath = md_image_match.group(3)
        ext = Path(filepath).suffix.lower()
        is_online = filepath.startswith(('http://', 'https://'))
        if ext in extensions or (is_online and ext in extensions):
            item = {
                "type": extensions.get(ext, 'image'),
                "content": filepath,
                "title": title if title else None,
                "is_online": is_online
            }
            if size:
                item["size"] = int(size)
            return item

    if md_link_match and is_twitter_url(md_link_match.group(2)):
        raw_title = md_link_match.group(1)
        url = md_link_match.group(2)
        size = None
        title = raw_title.strip() if raw_title else None
        if raw_title and '|' in raw_title:
            title_str, size_str = raw_title.split('|', 1)
            if size_str.strip().isdigit():
                size = int(size_str.strip())
                title = title_str.strip() if title_str.strip() else None
        tweet = Tweet(url, height=size if size else 800)
        return {"type": "tweet", "component": tweet, "url": url, "title": title}

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
        ext = Path(url).suffix.lower()
        if ext in extensions:
            return {"type": extensions[ext], "content": url, "is_online": True}
        elif is_twitter_url(url):
            tweet = Tweet(url)
            return {"type": "tweet", "component": tweet, "url": url, "title": None}
        elif is_youtube_url(url):
            return {"type": "youtube", "url": url, "title": None}
        else:
            return {"type": "web", "url": url, "title": None}

    return {"type": "markdown", "content": line}


def include_file_content(filepath, directories, linkify):
    """Reads a file and processes its content into a list of items."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Split into lines and process into items
        lines = content.splitlines()
        items = process_lines(lines, directories)
        return items if items else [{"type": "markdown", "content": "Empty file"}]
    except Exception as e:
        return [{"type": "markdown", "content": f"Error reading file {filepath}: {str(e)}"}]

# Center content using columns


def center_content(in_group, display_func, *args, **kwargs):
    """
    Centers content by wrapping it in a 1-6-1 column layout.
    Args:
        display_func: The Streamlit function to display the content (e.g., st.image, st.video).
        *args, **kwargs: Arguments to pass to the display function.
    """
    if in_group:
        display_func(*args, **kwargs)
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.write("")
        with col2:
            display_func(*args, **kwargs)
        with col3:
            st.write("")


def preprocess_markdown(content):
    """Remplace ==texte== par :orange-background[texte] dans une ligne markdown."""
    import re
    return re.sub(r'==([^=]+)==', r':orange-background[\1]', content)


@st.dialog("Video")
def display_video(filepath):
    st.video(filepath)

# New function to generate animation CSS


def generate_animation_css(animation_type, target_column=None, is_exit=False):
    css = "<style>\n"
    if is_exit:
        # Animations sortantes
        if animation_type == 'left':
            css += """
            @keyframes exitLeft {
                0% { transform: translateX(0); opacity: 1; }
                100% { transform: translateX(-100%); opacity: 1; }
            }
            """
        elif animation_type == 'right':
            css += """
            @keyframes exitRight {
                0% { transform: translateX(0); opacity: 1; }
                100% { transform: translateX(100%); opacity: 1; }
            }
            """
        elif animation_type == 'top':
            css += """
            @keyframes exitTop {
                0% { transform: translateY(0); opacity: 1; }
                100% { transform: translateY(-100%); opacity: 1; }
            }
            """
        elif animation_type == 'bottom':
            css += """
            @keyframes exitBottom {
                0% { transform: translateY(0); opacity: 1; }
                100% { transform: translateY(100%); opacity: 1; }
            }
            """
        # Par défaut, cible tout le contenu principal pour les animations sortantes
        target_selector = ".stMain"
        css += f"""
        {target_selector} .stImage img, {target_selector} .stVideo, {target_selector} .stMarkdown > div,
        {target_selector} div.stVerticalBlock:has(iframe[title="st.iframe"]) {{
            animation: exit{animation_type.capitalize()} 3s ease-in forwards;
        }}
        </style>
        """
    else:
        # Animations entrantes
        if animation_type == 'left':
            css += """
            @keyframes left {
                0% { transform: translateX(-100%); opacity: 1; }
                100% { transform: translateX(0); opacity: 1; }
            }
            """
        elif animation_type == 'right':
            css += """
            @keyframes right {
                0% { transform: translateX(100%); opacity: 1; }
                100% { transform: translateX(0); opacity: 1; }
            }
            """
        elif animation_type == 'top':
            css += """
            @keyframes top {
                0% { transform: translateY(-100%); opacity: 1; }
                100% { transform: translateY(0); opacity: 1; }
            }
            """
        elif animation_type == 'bottom':
            css += """
            @keyframes bottom {
                0% { transform: translateY(100%); opacity: 1; }
                100% { transform: translateY(0); opacity: 1; }
            }
            """
        elif animation_type == 'zoomIn':
            css += """
            @keyframes zoomIn {
                0% { transform: scale(0); opacity: 1; }
                100% { transform: scale(1); opacity: 1; }
            }
            """
        elif animation_type == 'zoomOut':
            css += """
            @keyframes zoomOut {
                0% { transform: scale(1.5); opacity: 1; }
                100% { transform: scale(1); opacity: 1; }
            }
            """
        elif animation_type == 'rock':
            css += """
            @keyframes rock {
                0% { transform: rotate(0deg); }
                25% { transform: rotate(5deg); }
                75% { transform: rotate(-5deg); }
                100% { transform: rotate(0deg); }
            }
            """

        # Déterminer le sélecteur en fonction de la colonne cible
        if target_column == 'right':
            target_selector = "div.stHorizontalBlock div.stColumn:nth-child(2)"
        elif target_column == 'left':
            target_selector = "div.stHorizontalBlock div.stColumn:nth-child(1)"
        else:
            target_selector = ".stMain"  # Par défaut, cible tout le contenu principal

        # Appliquer l'animation aux éléments cibles
        css += f"""
        {target_selector} .stImage img, {target_selector} .stVideo, {target_selector} .stMarkdown > div {{
            animation: {animation_type} 3s ease-out;
        }}
        {target_selector} div.stVerticalBlock:has(iframe[title="st.iframe"]) {{
            animation: {animation_type} 3s ease-out;
        }}
        </style>
        """

    return css


# Display an item in the app (modified for vertical centering)


def display_item(item, directories, is_presentation=False, in_group=False, animation_type=None, target_column=None, is_exit=False):
    """
    Displays an item based on its type:
    - All items are wrapped in a single column with optional vertical centering in presentation mode.
    - Markdown: Renders as text.
    - Tweet: Centered embed with optional title and custom height.
    - YouTube: Centered video with optional title.
    - Image: Centered with custom or default height (200px preview, 500px presentation).
    - Video: Centered local video with optional title.
    - Web: Centered webpage screenshot with optional title.
    - Group: Two items in side-by-side columns, vertically centered if enabled.
    """
    def include_animation():
        if is_presentation and animation_type:
            st.markdown(generate_animation_css(animation_type,
                        target_column, is_exit), unsafe_allow_html=True)
        else:
            st.markdown("<style></style>", unsafe_allow_html=True)

    include_animation()
    vertical_center = st.session_state.get(
        'vertical_center', False) and is_presentation
    alignment = "center" if vertical_center else "top"

    if item["type"] == "group":
        col1, col2 = st.columns(2, vertical_alignment=alignment)
        with col1:
            if target_column == "left":
                include_animation()  # BUG : trigger also right animation
            display_item(item["items"][0], directories,
                         is_presentation, True, animation_type, target_column, is_exit)
        with col2:
            if target_column == "right":
                include_animation()
            display_item(item["items"][1], directories,
                         is_presentation, True, animation_type, target_column, is_exit)

    else:
        # Wrap all non-group items in a single column for consistent vertical alignment
        (col,) = st.columns(1, vertical_alignment=alignment)
        with col:
            if item["type"] == "markdown":
                st.markdown(preprocess_markdown(item["content"]))
            elif item["type"] == "tweet":
                if "title" in item and item["title"]:
                    st.subheader(item["title"])
                center_content(in_group, lambda: item["component"].component())
            elif item["type"] == "youtube":
                if "title" in item and item["title"]:
                    st.subheader(item["title"])
                center_content(in_group, st.video, item["url"])
            elif item["type"] == "image":
                filepath = item["content"]
                if not item.get("is_online", False):
                    filepath = find_file(item["content"], directories) if not item["content"].startswith(
                        '/') else item["content"]
                if "title" in item and item["title"]:
                    st.subheader(item["title"])
                max_height = item.get("size", 800 if is_presentation else 200)
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
                filepath = find_file(item["content"], directories) if not item["content"].startswith(
                    '/') else item["content"]
                if "title" in item and item["title"]:
                    if item["title"] == "popup" and is_presentation:
                        display_video(filepath)
                    else:
                        st.subheader(item["title"])
                        center_content(in_group, st.video, filepath)
                else:
                    center_content(in_group, st.video, filepath)
            elif item["type"] == "web":
                if "title" in item and item["title"]:
                    st.subheader(item["title"])
                image_path = url_to_image(item["url"])
                if image_path:
                    st.image(image_path, use_container_width=True)
                    os.remove(image_path)
                else:
                    st.error("Failed to convert URL to image")


def estimate_markdown_size(content):
    """Estime la taille relative du contenu Markdown."""
    lines = content.split("\n")
    total_weight = 0
    for line in lines:
        line = line.strip()
        if line.startswith("# "):
            total_weight += 40  # Poids pour h1
        elif line.startswith("## "):
            total_weight += 30  # Poids pour h2
        elif line.startswith("### "):
            total_weight += 20  # Poids pour h3
        elif line:
            # Poids pour texte (10 par "bloc" de 80 caractères)
            total_weight += 10 * (len(line) // 80 + 1)
    return total_weight, len(lines)


class EzprezPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)

    def get_config_fields(self):
        """Définit les champs de configuration du plugin."""
        return {
            "ezprez_directories": {
                "type": "textarea",
                "label": t("ezprez_config_directories_label"),
                "default": t("ezprez_config_directories_default")
            }
        }

    def get_tabs(self):
        """Définit les onglets du plugin dans l'interface."""
        return [{"name": t("ezprez_tab"), "plugin": "ezprez"}]

    def run(self, config):
        """
        Main function for the presentation plugin:
        - Two modes: Preview (centered layout) and Presentation (wide layout).
        - Sidebar contains input area and controls, always accessible.
        - In Preview mode: Displays all slides with separators.
        - In Presentation mode: Shows one slide at a time, sidebar collapsed, navigation controls visible.
        - Keyboard shortcuts: Ctrl+P (Preview), Ctrl+Enter (Launch), ArrowLeft/Right (Prev/Next), Home/End (First/Last), Escape (Exit).
        """
        if 'presentation_mode' not in st.session_state:
            st.session_state['presentation_mode'] = False
        if 'exit_animation' not in st.session_state:
            # Stocke l'animation sortante en cours
            st.session_state['exit_animation'] = None
        if 'next_slide_ready' not in st.session_state:
            # Indique si la prochaine slide est prête
            st.session_state['next_slide_ready'] = False

        if not st.session_state['presentation_mode']:
            st.header(t("ezprez_header"))

        directories = [os.path.expanduser(dir.strip()) for dir in config.get(
            "ezprez", {}).get("ezprez_directories", "").split("\n") if dir.strip()]

        with st.sidebar:
            st.header(t("ezprez_preparation_header"))
            input_text = st.text_area(
                t("ezprez_input_label"), height=200, key="input_text")

            col1, col2 = st.columns(2)
            with col1:
                button(t("ezprez_preview_button"), "Ctrl+P", lambda: st.session_state.update(
                    {'presentation_mode': False, 'slides': process_lines(st.session_state.get('input_text', '').split("\n"), directories)}))
            with col2:
                button(t("ezprez_launch_button"), "Ctrl+Enter", lambda: st.session_state.update(
                    {'presentation_mode': True, 'input_text': st.session_state.get('input_text', ''),
                        'slides': process_lines(st.session_state.get('input_text', '').split("\n"), directories)}))

            if st.session_state['presentation_mode']:
                st.header(t("ezprez_navigation_header"))
                col1, col2 = st.columns(2)
                with col1:
                    button(t("ezprez_previous_button"), "ArrowLeft", lambda: st.session_state.update(
                        {'current_slide': max(0, st.session_state['current_slide'] - 1)}))
                with col2:
                    button(t("ezprez_next_button"), "ArrowRight", lambda: st.session_state.update(
                        {'current_slide': min(len(st.session_state['slides']) - 1, st.session_state['current_slide'] + 1)}))

                col3, col4 = st.columns(2)
                with col3:
                    button(t("ezprez_first_button"), "Home",
                           lambda: st.session_state.update({'current_slide': 0}))
                with col4:
                    button(t("ezprez_last_button"), "End", lambda: st.session_state.update(
                        {'current_slide': len(st.session_state['slides']) - 1}))

                col5, col6 = st.columns(2)
                with col5:
                    button(t("ezprez_skip_button"), "ArrowDown", lambda: st.session_state.update(
                        {'current_slide': min(len(st.session_state['slides']) - 1, st.session_state['current_slide'] + 2)}))
                with col6:
                    button(t("ezprez_exit_button"), "Escape", lambda: st.session_state.update(
                        {'presentation_mode': False}))

                # New animation controls
                st.subheader("Animation Controls")
                button("Random Animation Next", "PageUp", lambda: st.session_state.update({
                    'current_animation': random.choice(['left', 'right', 'top', 'bottom', 'zoomIn', 'zoomOut']),
                }))
                col7, col8 = st.columns(2)
                with col7:
                    button("Rock Left", "Ctrl+A", lambda: st.session_state.update({
                        'current_animation': 'rock',
                        'target_column': 'left',
                    }))
                    button("Slide Left", "Ctrl+ArrowLeft", lambda: st.session_state.update({
                        'exit_animation': 'left',
                        'next_slide': min(len(st.session_state['slides']) - 1, st.session_state['current_slide'] + 1),
                        'next_slide_ready': False
                    }))
                    button("Slide Up", "Ctrl+ArrowUp", lambda: st.session_state.update({
                        'exit_animation': 'top',
                        'next_slide': min(len(st.session_state['slides']) - 1, st.session_state['current_slide'] + 1),
                        'next_slide_ready': False
                    }))
                with col8:
                    button("Rock Right", "Ctrl+Z", lambda: st.session_state.update({
                        'current_animation': 'rock',
                        'target_column': 'right'
                    }))
                    button("Slide Right", "Ctrl+ArrowRight", lambda: st.session_state.update({
                        'exit_animation': 'right',
                        'next_slide': min(len(st.session_state['slides']) - 1, st.session_state['current_slide'] + 1),
                        'next_slide_ready': False
                    }))
                    button("Slide Down", "Ctrl+ArrowDown", lambda: st.session_state.update({
                        'exit_animation': 'bottom',
                        'next_slide': min(len(st.session_state['slides']) - 1, st.session_state['current_slide'] + 1),
                        'next_slide_ready': False
                    }))

                # Existing presentation controls (green background, vertical center, font size)
                green_bg = st.checkbox(
                    t("ezprez_green_bg_label"), value=False, key="green_bg")
                if green_bg:
                    st.markdown("""
                        <style>
                        .stMain { background-color: #00FF00; }
                        .stMain h1, .stMain h2, .stMain h3, .stMain h4, .stMain h5, .stMain h6,
                        .stMain p, .stMain ul, .stMain ol, .stMain blockquote, .stMain table {
                            background-color: #000000; color: #FFFFFF; padding: 10px; margin: 5px 0; display: inline-block;
                        }
                        .stMain ul, .stMain ol { display: block; padding: 10px 10px 10px 30px; }
                        </style>
                    """, unsafe_allow_html=True)
                st.checkbox(t("ezprez_vertical_center_label"),
                            value=False, key="vertical_center")
                auto_scale = st.checkbox(
                    "AutoScale", value=False, key="auto_scale")
                font_size_scale = st.slider(
                    "Font Size Scale", 1.0, 6.0, 2.0, 0.1, key="font_size_scale", disabled=auto_scale)

        # Apply font size scaling
        if st.session_state['presentation_mode']:
            font_size_scale = st.session_state.get('font_size_scale', 1.0)
            slides = st.session_state.get('slides', [])
            current = st.session_state.get('current_slide', 0)
            if auto_scale and slides and current < len(slides) and slides[current]["type"] == "markdown":
                weight, line_count = estimate_markdown_size(
                    slides[current]["content"])
                font_size_scale = max(1.0, min(4.0, 300 / max(weight, 1)))
            if auto_scale and slides and current < len(slides) and slides[current]["type"] == "group":
                if slides[current]["items"][0]["type"] == "markdown":
                    weight, line_count = estimate_markdown_size(
                        slides[current]["items"][0]["content"])
                    font_size_scale = max(1.0, min(4.0, 250 / max(weight, 1)))
            st.markdown(f"""
                <style>
                .stMain {{ font-size: calc(1rem * {font_size_scale}); }}
                .stMain h1 {{ font-size: calc(2.5rem * {font_size_scale}); }}
                .stMain h2 {{ font-size: calc(2rem * {font_size_scale}); }}
                .stMain h3 {{ font-size: calc(1.5rem * {font_size_scale}); }}
                .stMain p, .stMain li, .stMain td {{ font-size: calc(1rem * {font_size_scale}); }}
                </style>
            """, unsafe_allow_html=True)

        if not st.session_state['presentation_mode'] and 'slides' in st.session_state:
            for item in st.session_state['slides']:
                display_item(item, directories, is_presentation=False)
                st.markdown("---")

        if st.session_state['presentation_mode']:
            if 'slides' not in st.session_state or not st.session_state['slides']:
                st.warning(t("ezprez_no_content_warning"))
                st.session_state['presentation_mode'] = False
                return

            slides = st.session_state['slides']
            if 'current_slide' not in st.session_state:
                st.session_state['current_slide'] = 0

            current = st.session_state['current_slide']

            if st.session_state['exit_animation'] and not st.session_state['next_slide_ready']:
                # Afficher la slide actuelle avec l'animation sortante
                display_item(slides[current], directories, is_presentation=True,
                             animation_type=st.session_state['exit_animation'], is_exit=True)
                # Simuler la fin de l'animation (immédiat dans Streamlit, pas de vrai délai)
                st.session_state['next_slide_ready'] = True
                time.sleep(3)
                st.rerun()  # Forcer un rerendu pour passer à l'étape suivante
            elif st.session_state['next_slide_ready']:
                # Passer à la slide suivante sans animation
                st.session_state['current_slide'] = st.session_state['next_slide']
                st.session_state['exit_animation'] = None
                st.session_state['next_slide_ready'] = False
                display_item(slides[st.session_state['current_slide']],
                             directories, is_presentation=True, animation_type=None)
            else:
                # Affichage normal avec animation entrante si spécifiée
                animation_type = st.session_state.get('current_animation')
                target_column = st.session_state.get('target_column')
                display_item(slides[current], directories,
                             is_presentation=True, animation_type=animation_type, target_column=target_column)
                st.session_state['current_animation'] = None
                st.session_state['target_column'] = None


if __name__ == "__main__":
    # Pour tester le plugin indépendamment (optionnel)
    st.write("Ezprez Plugin standalone test")
    plugin_manager = None  # Simuler un plugin manager pour les tests
    plugin = EzprezPlugin("ezprez", plugin_manager)
    plugin.run({})
