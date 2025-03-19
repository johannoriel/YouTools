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

# Configure logging for debugging purposes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load directory paths from a config.ini file
def load_directories():
    """Loads predefined directories from config.ini to search for local files."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.get('Paths', 'directories', fallback='').split('\n')

# Find a file in the predefined directories
def find_file(filename, directories):
    """Searches for a file in the specified directories and returns its full path if found."""
    for directory in directories:
        full_path = Path(directory) / filename
        if full_path.exists():
            return str(full_path)
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
            api = f"https://publish.twitter.com/oembed?url={url}"
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
    """
    result = []
    current_markdown = []
    linkify = LinkifyIt()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Handle slide separator
        if line == "---":
            if current_markdown and "\n".join(current_markdown).strip():
                result.append({"type": "markdown", "content": "\n".join(current_markdown)})
            current_markdown = []
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue

        # Handle column separator
        if line == "--" and i > 0 and i + 1 < len(lines):
            prev_item = None
            if current_markdown and "\n".join(current_markdown).strip():
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

        # Ignore empty lines before separators
        if not line and i + 1 < len(lines) and lines[i + 1].strip() in ["---", "--"]:
            i += 1
            continue

        # Process individual line
        item = parse_single_line(line, directories, linkify)
        if item["type"] == "markdown":
            current_markdown.append(item["content"])
        else:
            if current_markdown and "\n".join(current_markdown).strip():
                result.append({"type": "markdown", "content": "\n".join(current_markdown)})
                current_markdown = []
            result.append(item)
        i += 1

    if current_markdown and "\n".join(current_markdown).strip():
        result.append({"type": "markdown", "content": "\n".join(current_markdown)})

    return result

# Parse a single line into an item (corrected tweet regex)
def parse_single_line(line, directories, linkify):
    """
    Parses a single line into an item based on its content:
    - Supports file extensions (.jpg, .png, .mp4, .flv).
    - Handles Markdown links for files, images, and URLs.
    - Supports custom image height with |xxx syntax (e.g., ![|725](path)).
    - Supports custom tweet height with |xxx syntax (e.g., [|725](tweet_url) or [title|725](tweet_url)).
    - Titles are None if empty in Markdown links.
    """
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

    if md_link_match and is_twitter_url(md_link_match.group(2)):
        raw_title = md_link_match.group(1)  # Capture everything before the URL
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
        print(title)
        if is_twitter_url(url):
            tweet = Tweet(url)
            return {"type": "tweet", "component": tweet, "url": url, "title": title if title else None}
        elif is_youtube_url(url):
            return {"type": "youtube", "url": url, "title": title if title else None}
        else:
            return {"type": "web", "url": url, "title": title if title else None}

    matches = linkify.match(line)
    if matches:
        print("Link Match")
        url = matches[0].url
        if is_twitter_url(url):
            tweet = Tweet(url)
            return {"type": "tweet", "component": tweet, "url": url, "title": None}
        elif is_youtube_url(url):
            return {"type": "youtube", "url": url, "title": None}
        else:
            return {"type": "web", "url": url, "title": None}

    return {"type": "markdown", "content": line}


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
    vertical_center = st.session_state.get('vertical_center', False) and is_presentation
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
            if item["type"] == "markdown":
                st.markdown(item["content"])
            elif item["type"] == "tweet":
                if item["title"]:
                    st.subheader(item["title"])
                center_content(in_group,lambda: item["component"].component())
            elif item["type"] == "youtube":
                if item["title"]:
                    st.subheader(item["title"])
                center_content(in_group,st.video, item["url"])
            elif item["type"] == "image":
                filepath = find_file(item["content"], directories) if not item["content"].startswith('/') else item["content"]
                if item["title"]:
                    st.subheader(item["title"])
                max_height = item.get("size", 500 if is_presentation else 200)
                center_content(in_group,st.image, filepath, use_container_width=True)
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
                center_content(in_group,st.video, filepath)
            elif item["type"] == "web":
                if item["title"]:
                    st.subheader(item["title"])
                image_path = url_to_image(item["url"])
                if image_path:
                    center_content(in_group,st.image, image_path, use_container_width=True)
                    os.remove(image_path)
                else:
                    st.error("Failed to convert URL to image")

# Main application logic (modified)
def main():
    """
    Main function for the presentation app:
    - Two modes: Preview (centered layout) and Presentation (wide layout).
    - Sidebar contains input area and controls, always accessible.
    - In Preview mode: Displays all slides with separators.
    - In Presentation mode: Shows one slide at a time, sidebar collapsed, navigation controls visible.
    - Keyboard shortcuts: Ctrl+P (Preview), Ctrl+Enter (Launch), ArrowLeft/Right (Prev/Next), Home/End (First/Last), Escape (Exit).
    """
    if 'presentation_mode' not in st.session_state:
        st.session_state['presentation_mode'] = False

    # Configure page layout based on mode
    if st.session_state['presentation_mode']:
        st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title=None)
    else:
        st.set_page_config(layout="centered", initial_sidebar_state="expanded", page_title="Quick Presentation")

    # Display title only in preview mode
    if not st.session_state['presentation_mode']:
        st.title("Quick Presentation")

    directories = load_directories()

    # Sidebar for input and navigation
    with st.sidebar:
        st.header("Preparation")
        input_text = st.text_area("Paste your lines here:", height=200, key="input_text")

        col1, col2 = st.columns(2)
        with col1:
            preview = button("Preview", "Ctrl+P", lambda: st.session_state.update({'slides': process_lines(st.session_state.get('input_text', '').split("\n"), directories)}), hint=True)
        with col2:
            launch = button("Launch", "Ctrl+Enter", lambda: st.session_state.update({'presentation_mode': True, 'current_slide': 0, 'input_text': st.session_state.get('input_text', ''), 'slides': process_lines(st.session_state.get('input_text', '').split("\n"), directories)}), hint=True)

        # Navigation controls in presentation mode
        if st.session_state['presentation_mode']:
            st.header("Navigation")
            col1, col2 = st.columns(2)
            with col1:
                button("Previous", "ArrowLeft", lambda: st.session_state.update({'current_slide': max(0, st.session_state['current_slide'] - 1)}), hint=True)
            with col2:
                button("Next", "ArrowRight", lambda: st.session_state.update({'current_slide': min(len(st.session_state['slides']) - 1, st.session_state['current_slide'] + 1)}), hint=True)

            col3, col4 = st.columns(2)
            with col3:
                button("First", "Home", lambda: st.session_state.update({'current_slide': 0}), hint=True)
            with col4:
                button("Last", "End", lambda: st.session_state.update({'current_slide': len(st.session_state['slides']) - 1}), hint=True)

            button("Exit", "Escape", lambda: st.session_state.update({'presentation_mode': False}), hint=True)

            # Add checkbox for green background in presentation mode
            green_bg = st.checkbox("Green Background", value=False, key="green_bg")
            if green_bg:
                st.markdown("""
                    <style>
                    .stApp {
                        background-color: #00FF00;
                    }
                    /* Style for Markdown elements */
                    h1, h2, h3, h4, h5, h6, p, ul, ol, li, blockquote {
                        background-color: #000000;
                        color: #FFFFFF;
                        padding: 10px;
                        margin: 5px 0;
                        display: inline-block;
                    }
                    ul, ol {
                        display: block;
                        padding: 10px 10px 10px 30px;
                    }
                    li {
                        margin: 0;
                        display: block;
                    }
                    </style>
                """, unsafe_allow_html=True)

            # Add checkbox for vertical centering in presentation mode
            st.checkbox("Center Vertically", value=False, key="vertical_center")

    # Preview mode: Show all slides
    if not st.session_state['presentation_mode'] and 'slides' in st.session_state:
        st.header("Preview")
        for item in st.session_state['slides']:
            display_item(item, directories, is_presentation=False)
            st.markdown("---")

    # Presentation mode: Show one slide at a time
    if st.session_state['presentation_mode']:
        if 'slides' not in st.session_state or not st.session_state['slides']:
            st.warning("Please enter content and generate slides before launching the presentation.")
            st.session_state['presentation_mode'] = False
            return

        slides = st.session_state['slides']
        if 'current_slide' not in st.session_state:
            st.session_state['current_slide'] = 0

        current = st.session_state['current_slide']

        if slides:
            display_item(slides[current], directories, is_presentation=True)

if __name__ == "__main__":
    main()
