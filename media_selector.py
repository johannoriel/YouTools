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


def media_selector(media_dirs, extensions, suffix, streamlit_component=st):
    # Normalize input to a list of directories
    if isinstance(media_dirs, str):
        media_dirs = [media_dirs]
    elif not isinstance(media_dirs, list):
        raise ValueError("media_dirs must be a string or a list of strings")

    # List files from all selected directories (no recursion)
    def scan_media_files():
        media_files = []
        media_paths = []
        for dir_path in media_dirs:
            if not os.path.exists(dir_path):
                continue
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and
                     any(f.lower().endswith(ext.lower()) for ext in extensions)]
            media_files.extend(files)
            media_paths.extend(os.path.join(dir_path, f) for f in files)
        return media_files, media_paths

    # Initial scan
    media_files, media_paths = scan_media_files()
    media_names = [os.path.splitext(f)[0] for f in media_files]
    media_dates = [os.path.getmtime(path) for path in media_paths]

    # Filter and sort UI - now with 3 columns
    search_col, sort_col, refresh_col = streamlit_component.columns([4, 3, 1])
    with search_col:
        search_query = st.text_input(
            "Rechercher un média (ex. 'ru')", "", key=f"search_{suffix}")
    with sort_col:
        sort_options = [
            "Alphabétique (A-Z)",
            "Alphabétique (Z-A)",
            "Date (plus ancien au plus récent)",
            "Date (plus récent au plus ancien)"
        ]
        sort_choice = st.selectbox(
            "Trier par :", sort_options, key=f"sort_{suffix}")
    with refresh_col:
        st.write("")  # Espacement vertical
        if st.button("🔄", key=f"refresh_{suffix}", help="Rafraîchir la liste des médias"):
            # Effacer le cache des thumbnails
            st.cache_data.clear()
            # Rescanner les fichiers
            media_files, media_paths = scan_media_files()
            media_names = [os.path.splitext(f)[0] for f in media_files]
            media_dates = [os.path.getmtime(path) for path in media_paths]
            st.rerun()

    # Reste de la fonction inchangé...
    # Filter by search query
    if search_query:
        filtered_indices = [i for i, name in enumerate(
            media_names) if search_query.lower() in name.lower()]
    else:
        filtered_indices = list(range(len(media_files)))

    # Sort based on user choice
    if sort_choice == "Alphabétique (A-Z)":
        sorted_indices = sorted(
            filtered_indices, key=lambda i: media_names[i].lower())
    elif sort_choice == "Alphabétique (Z-A)":
        sorted_indices = sorted(
            filtered_indices, key=lambda i: media_names[i].lower(), reverse=True)
    elif sort_choice == "Date (plus ancien au plus récent)":
        sorted_indices = sorted(filtered_indices, key=lambda i: media_dates[i])
    else:  # "Date (plus récent au plus ancien)"
        sorted_indices = sorted(
            filtered_indices, key=lambda i: media_dates[i], reverse=True)

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
                label="Media",
                images=thumbnails,
                captions=filtered_media_names,
                use_container_width=True,
            )
            if selected_thumb:
                selected_idx = thumbnails.index(selected_thumb)
                selected_media = filtered_media_paths[selected_idx]
        else:
            streamlit_component.write(
                "Aucun média trouvé dans les répertoires sélectionnés.")

    return selected_media


def remote_media_selector(media_items, suffix, streamlit_component=st):
    """
    Sélectionneur de médias pour des ressources distantes (URLs ou images en mémoire)

    Args:
        media_items: Liste de dictionnaires avec:
            - 'url': URL de l'image/vignette
            - 'name': Nom/description du média
            - 'date': Date de création (optionnel)
            - 'type': Type de média (optionnel)
        streamlit_component: Composant Streamlit à utiliser (par défaut st)

    Returns:
        L'item sélectionné ou None
    """
    if not media_items:
        streamlit_component.write("Aucun média disponible.")
        return None

    # Extraire les informations nécessaires
    media_urls = [item['url'] for item in media_items]
    media_names = [item.get('name', f"Media {i+1}")
                   for i, item in enumerate(media_items)]
    media_dates = [item.get('date', 0) for item in media_items]

    # Filtre et tri UI
    search_col, sort_col = streamlit_component.columns(2)
    with search_col:
        search_query = streamlit_component.text_input(
            "Rechercher un média", "", key=f"search_remote_{suffix}")
    with sort_col:
        sort_options = [
            "Alphabétique (A-Z)",
            "Alphabétique (Z-A)",
            "Date (plus ancien au plus récent)",
            "Date (plus récent au plus ancien)"
        ]
        sort_choice = streamlit_component.selectbox(
            "Trier par :", sort_options, key=f"sort_remote_{suffix}")

    # Filtrer par requête de recherche
    if search_query:
        filtered_indices = [i for i, name in enumerate(media_names)
                            if search_query.lower() in name.lower()]
    else:
        filtered_indices = list(range(len(media_items)))

    # Trier selon le choix de l'utilisateur
    if sort_choice == "Alphabétique (A-Z)":
        sorted_indices = sorted(
            filtered_indices, key=lambda i: media_names[i].lower())
    elif sort_choice == "Alphabétique (Z-A)":
        sorted_indices = sorted(
            filtered_indices, key=lambda i: media_names[i].lower(), reverse=True)
    elif sort_choice == "Date (plus ancien au plus récent)":
        sorted_indices = sorted(filtered_indices, key=lambda i: media_dates[i])
    else:  # "Date (plus récent au plus ancien)"
        sorted_indices = sorted(
            filtered_indices, key=lambda i: media_dates[i], reverse=True)

    # Appliquer le tri
    filtered_media_urls = [media_urls[i] for i in sorted_indices]
    filtered_media_names = [media_names[i] for i in sorted_indices]
    filtered_media_items = [media_items[i] for i in sorted_indices]

    # Sélection d'image avec container scrollable
    selected_media = None
    with streamlit_component.container(height=400):
        if filtered_media_urls:
            selected_idx = image_select(
                label="Choisis un média",
                images=filtered_media_urls,
                captions=filtered_media_names,
                use_container_width=True,
                return_value="index"
            )
            if selected_idx is not None:
                selected_media = filtered_media_items[selected_idx]
        else:
            streamlit_component.write(
                "Aucun média ne correspond aux critères.")

    return selected_media
