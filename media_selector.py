import streamlit as st
import os
from PIL import Image
import cv2
from streamlit_image_select import image_select

def get_thumbnail(media_path):
    if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(media_path)
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((200, 200))
        else:
            img = Image.new('RGB', (200, 200), color='gray')
        cap.release()
    else:
        try:
            img = Image.open(media_path)
            if img.mode == 'RGBA':  # Convert RGBA to RGB to avoid JPEG issues
                img = img.convert('RGB')
            img.thumbnail((200, 200))
        except:
            img = Image.new('RGB', (200, 200), color='gray')
    return img

def media_selector(media_dirs, extensions, streamlit_component=st):
    # Normalize input to a list of directories
    if isinstance(media_dirs, str):
        media_dirs = [media_dirs]
    elif not isinstance(media_dirs, list):
        raise ValueError("media_dirs must be a string or a list of strings")

    # List files from all selected directories (no recursion)
    media_files = []
    media_paths = []
    for dir_path in media_dirs:
        if not os.path.exists(dir_path):
            continue
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and
                 any(f.lower().endswith(ext.lower()) for ext in extensions)]
        media_files.extend(files)
        media_paths.extend(os.path.join(dir_path, f) for f in files)

    media_names = [os.path.splitext(f)[0] for f in media_files]
    media_dates = [os.path.getmtime(path) for path in media_paths]

    # Filter and sort UI
    search_col, sort_col = streamlit_component.columns(2)
    with search_col:
        search_query = st.text_input("Rechercher un média (ex. 'ru')", "", key="search")
    with sort_col:
        sort_options = [
            "Alphabétique (A-Z)",
            "Alphabétique (Z-A)",
            "Date (plus ancien au plus récent)",
            "Date (plus récent au plus ancien)"
        ]
        sort_choice = st.selectbox("Trier par :", sort_options, key="sort")

    # Filter by search query
    if search_query:
        filtered_indices = [i for i, name in enumerate(media_names) if search_query.lower() in name.lower()]
    else:
        filtered_indices = list(range(len(media_files)))

    # Sort based on user choice
    if sort_choice == "Alphabétique (A-Z)":
        sorted_indices = sorted(filtered_indices, key=lambda i: media_names[i].lower())
    elif sort_choice == "Alphabétique (Z-A)":
        sorted_indices = sorted(filtered_indices, key=lambda i: media_names[i].lower(), reverse=True)
    elif sort_choice == "Date (plus ancien au plus récent)":
        sorted_indices = sorted(filtered_indices, key=lambda i: media_dates[i])
    else:  # "Date (plus récent au plus ancien)"
        sorted_indices = sorted(filtered_indices, key=lambda i: media_dates[i], reverse=True)

    # Apply sorting
    filtered_media_paths = [media_paths[i] for i in sorted_indices]
    filtered_media_names = [media_names[i] for i in sorted_indices]

    # Generate thumbnails with caching
    @st.cache_data
    def load_thumbnails(paths):
        return [get_thumbnail(path) for path in paths]

    thumbnails = load_thumbnails(filtered_media_paths)

    # Scrollable container for media grid
    selected_media = None
    with streamlit_component.container(height=400):
        if filtered_media_paths:
            selected_thumb = image_select(
                label="Choisis un média",
                images=thumbnails,
                captions=filtered_media_names,
                use_container_width=True,
            )
            if selected_thumb:
                selected_idx = thumbnails.index(selected_thumb)
                selected_media = filtered_media_paths[selected_idx]
        else:
            streamlit_component.write("Aucun média trouvé dans les répertoires sélectionnés.")

    return selected_media
