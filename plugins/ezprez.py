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

    def __init__(self, url, embed_str=False, height=600):
        if not embed_str:
            api = f"https://publish.twitter.com/oembed?hide_thread=true&url={url}&widget=Video"
            try:
                response = requests.get(api, timeout=10)
                response.raise_for_status()
                data = response.json()
                self.text = data["html"]
                self.title = data.get("title", url)
            except (requests.RequestException, ValueError) as e:
                logger.error(f"Error fetching tweet {url}: {str(e)}")
                self.text = f"<p>Tweet error: {str(e)}</p>"
                self.title = "Error"
        else:
            self.text = url
            self.title = url
        self.height = height  # Store custom height

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
    - '--' groups two items into columns within the same slide.
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

        # Vérifier si la ligne est un commentaire sur une seule ligne
        if stripped_line.startswith("%%") and stripped_line.endswith("%%"):
            continue

        # Début ou fin d'un commentaire
        if stripped_line == "%%":
            in_comment = not in_comment
            continue

        # Ignorer la ligne si elle est dans un commentaire
        if not in_comment:
            filtered_lines.append(line)

    # Vérifier qu'on n'est pas resté dans un commentaire non fermé
    if in_comment:
        logger.warning("Unclosed comment block detected (missing closing %%)")

    # Traitement des lignes filtrées
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
        if line == "--" and i > 0 and i + 1 < len(filtered_lines):
            prev_item = None
            if current_markdown and "\n".join(current_markdown).strip():
                prev_item = {"type": "markdown", "content": "\n".join(current_markdown)}
                current_markdown = []
            elif result:
                prev_item = result.pop()

            i += 1
            while i < len(filtered_lines) and not filtered_lines[i].strip():
                i += 1
            if i >= len(filtered_lines):
                if prev_item:
                    result.append(prev_item)
                break

            # Collect and parse lines for the next column individually
            next_items = []
            while i < len(filtered_lines) and filtered_lines[i].strip() not in ["---", "--"]:
                if filtered_lines[i].strip():
                    parsed = parse_single_line(filtered_lines[i], directories, linkify)
                    if isinstance(parsed, list):
                        next_items.extend(parsed)
                    else:
                        next_items.append(parsed)
                i += 1

            # If we have items for both columns, group them
            if next_items:
                next_item = next_items[0] if len(next_items) == 1 else {
                    "type": "markdown", "content": "\n".join(item["content"] for item in next_items if item["type"] == "markdown")
                } or next_items[0]  # Fallback to first item if no markdown
                if prev_item and next_item:
                    result.append({"type": "group", "items": [prev_item, next_item]})
                elif prev_item:
                    result.append(prev_item)
                if len(next_items) > 1:
                    result.extend(next_items[1:])  # Add any additional items
            continue

        # Ignore empty lines before separators
        if not line and i + 1 < len(filtered_lines) and filtered_lines[i + 1].strip() in ["---", "--"]:
            i += 1
            continue

        # Process individual line, which may return a single item or a list
        items = parse_single_line(line, directories, linkify)
        if isinstance(items, list):
            # If parse_single_line returns a list (e.g., from included file), extend result
            if current_markdown and "\n".join(current_markdown).strip():
                result.append(
                    {"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
            result.extend(items)
        else:
            # Handle single item as before
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

    return result


def parse_single_line(line, directories, linkify):
    """
    Parses a single line into an item or list of items based on its content.
    """
    extensions = {'.jpg': 'image', '.png': 'image', '.jpeg': 'image', '.gif': 'image',
                  '.mp4': 'video', '.flv': 'video'}

    for ext, content_type in extensions.items():
        if line.endswith(ext):
            return {"type": content_type, "content": line}

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
        if ext in extensions:
            item = {
                "type": extensions[ext], "content": filepath, "title": title if title else None}
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
        tweet = Tweet(url, height=size if size else 600)
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
        if is_twitter_url(url):
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

# Display an item in the app (modified for vertical centering)


def display_item(item, directories, is_presentation=False, in_group=False):
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
    vertical_center = st.session_state.get(
        'vertical_center', False) and is_presentation
    alignment = "center" if vertical_center else "top"

    if item["type"] == "group":
        col1, col2 = st.columns(2, vertical_alignment=alignment)
        with col1:
            display_item(item["items"][0], directories, is_presentation, True)
        with col2:
            display_item(item["items"][1], directories, is_presentation, True)
    else:
        # Wrap all non-group items in a single column for consistent vertical alignment
        (col,) = st.columns(1, vertical_alignment=alignment)
        with col:
            #st.info(item['type'])
            if item["type"] == "markdown":
                st.markdown(preprocess_markdown(item["content"]))
            elif item["type"] == "tweet":
                if item["title"]:
                    st.subheader(item["title"])
                center_content(in_group, lambda: item["component"].component())
            elif item["type"] == "youtube":
                if item["title"]:
                    st.subheader(item["title"])
                center_content(in_group, st.video, item["url"])
            elif item["type"] == "image":
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
                if item["title"]:
                    st.subheader(item["title"])
                center_content(in_group, st.video, filepath)
            elif item["type"] == "web":
                if item["title"]:
                    st.subheader(item["title"])
                image_path = url_to_image(item["url"])
                if image_path:
                    st.image(image_path, use_container_width=True)
                    os.remove(image_path)
                else:
                    st.error("Failed to convert URL to image")


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

        if not st.session_state['presentation_mode']:
            st.header(t("ezprez_header"))

        directories = [os.path.expanduser(dir.strip()) for dir in config.get(
            "ezprez", {}).get("ezprez_directories", "").split("\n") if dir.strip()]

        # Sidebar for input and navigation
        with st.sidebar:
            st.header(t("ezprez_preparation_header"))
            input_text = st.text_area(
                t("ezprez_input_label"), height=200, key="input_text")

            col1, col2 = st.columns(2)
            with col1:
                button(t("ezprez_preview_button"), "Ctrl+P", lambda: st.session_state.update(
                    {'slides': process_lines(st.session_state.get('input_text', '').split("\n"), directories)}), hint=True)
            with col2:
                button(t("ezprez_launch_button"), "Ctrl+Enter", lambda: st.session_state.update({'presentation_mode': True, 'input_text': st.session_state.get(
                    'input_text', ''), 'slides': process_lines(st.session_state.get('input_text', '').split("\n"), directories)}), hint=True)

            # Navigation controls in presentation mode
            if st.session_state['presentation_mode']:
                st.header(t("ezprez_navigation_header"))
                col1, col2 = st.columns(2)
                with col1:
                    button(t("ezprez_previous_button"), "ArrowLeft", lambda: st.session_state.update(
                        {'current_slide': max(0, st.session_state['current_slide'] - 1)}), hint=True)
                with col2:
                    button(t("ezprez_next_button"), "ArrowRight", lambda: st.session_state.update({'current_slide': min(
                        len(st.session_state['slides']) - 1, st.session_state['current_slide'] + 1)}), hint=True)

                col3, col4 = st.columns(2)
                with col3:
                    button(t("ezprez_first_button"), "Home", lambda: st.session_state.update(
                        {'current_slide': 0}), hint=True)
                with col4:
                    button(t("ezprez_last_button"), "End", lambda: st.session_state.update(
                        {'current_slide': len(st.session_state['slides']) - 1}), hint=True)

                button(t("ezprez_exit_button"), "Escape", lambda: st.session_state.update(
                    {'presentation_mode': False}), hint=True)

                # Add checkbox for green background in presentation mode
                green_bg = st.checkbox(
                    t("ezprez_green_bg_label"), value=False, key="green_bg")
                if green_bg:
                    st.markdown("""
                        <style>
                        /* Apply green background only to stMain */
                        .stMain {
                            background-color: #00FF00;
                        }
                        /* Style for Markdown elements within stMain */
                        .stMain h1, .stMain h2, .stMain h3, .stMain h4, .stMain h5, .stMain h6,
                        .stMain p, .stMain ul, .stMain ol, .stMain li, .stMain blockquote {
                            background-color: #000000;
                            color: #FFFFFF;
                            padding: 10px;
                            margin: 5px 0;
                            display: inline-block;
                        }
                        .stMain ul, .stMain ol {
                            display: block;
                            padding: 10px 10px 10px 30px;
                        }
                        .stMain li {
                            margin: 0;
                            display: block;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                # Add checkbox for vertical centering in presentation mode
                st.checkbox(t("ezprez_vertical_center_label"),
                            value=False, key="vertical_center")
                # Add font size slider for presentation mode
                font_size_scale = st.slider(
                    "Font Size Scale", min_value=1.0, max_value=6.0, value=2.0, step=0.1, key="font_size_scale")

        # Apply font size scaling in presentation mode only to main content
        if st.session_state['presentation_mode']:
            font_size_scale = st.session_state.get('font_size_scale', 1.0)
            st.markdown(f"""
                <style>
                /* Target only the main content area (stMain) */
                .stMain {{
                    font-size: calc(1rem * {font_size_scale});
                }}
                /* Increase heading sizes relative to the base font size within stMain */
                .stMain h1 {{
                    font-size: calc(2.5rem * {font_size_scale});
                }}
                .stMain h2 {{
                    font-size: calc(2rem * {font_size_scale});
                }}
                .stMain h3 {{
                    font-size: calc(1.5rem * {font_size_scale});
                }}
                .stMain p, .stMain li {{
                    font-size: calc(1rem * {font_size_scale});
                }}
                </style>
            """, unsafe_allow_html=True)

        # Preview mode: Show all slides
        if not st.session_state['presentation_mode'] and 'slides' in st.session_state:
            for item in st.session_state['slides']:
                display_item(item, directories, is_presentation=False)
                st.markdown("---")

        # Presentation mode: Show one slide at a time
        if st.session_state['presentation_mode']:
            if 'slides' not in st.session_state or not st.session_state['slides']:
                st.warning(t("ezprez_no_content_warning"))
                st.session_state['presentation_mode'] = False
                return

            slides = st.session_state['slides']
            if 'current_slide' not in st.session_state:
                st.session_state['current_slide'] = 0

            current = st.session_state['current_slide']
            if slides:
                display_item(slides[current], directories,
                             is_presentation=True)


if __name__ == "__main__":
    # Pour tester le plugin indépendamment (optionnel)
    st.write("Ezprez Plugin standalone test")
    plugin_manager = None  # Simuler un plugin manager pour les tests
    plugin = EzprezPlugin("ezprez", plugin_manager)
    plugin.run({})
