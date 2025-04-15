import streamlit as st
import os
from PIL import Image
import cv2
from streamlit_image_select import image_select
from global_vars import translations, t

translations["en"].update({
    "media_selector_search_placeholder": "Search for media (e.g. 'ru')",
    "media_selector_sort_label": "Sort by:",
    "media_selector_sort_option_az": "Alphabetical (A-Z)",
    "media_selector_sort_option_za": "Alphabetical (Z-A)",
    "media_selector_sort_option_date_oldest": "Date (oldest to newest)",
    "media_selector_sort_option_date_newest": "Date (newest to oldest)",
    "media_selector_refresh_tooltip": "Refresh media list",
    "media_selector_no_media_found": "No media found in selected directories.",
    "media_selector_remote_search_placeholder": "Search for media",
    "media_selector_remote_choose_label": "Choose a media",
    "media_selector_remote_no_matching": "No media matching criteria.",
    "media_selector_remote_no_media": "No media available."
})

translations["fr"].update({
    "media_selector_search_placeholder": "Rechercher un média (ex. 'ru')",
    "media_selector_sort_label": "Trier par :",
    "media_selector_sort_option_az": "Alphabétique (A-Z)",
    "media_selector_sort_option_za": "Alphabétique (Z-A)",
    "media_selector_sort_option_date_oldest": "Date (plus ancien au plus récent)",
    "media_selector_sort_option_date_newest": "Date (plus récent au plus ancien)",
    "media_selector_refresh_tooltip": "Rafraîchir la liste des médias",
    "media_selector_no_media_found": "Aucun média trouvé dans les répertoires sélectionnés.",
    "media_selector_remote_search_placeholder": "Rechercher un média",
    "media_selector_remote_choose_label": "Choisis un média",
    "media_selector_remote_no_matching": "Aucun média ne correspond aux critères.",
    "media_selector_remote_no_media": "Aucun média disponible."
})

# Constantes pour les extensions de fichiers
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv')
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.ogg')
ALL_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + AUDIO_EXTENSIONS

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

def media_selector(media_dirs, extensions, suffix, streamlit_component=st, initial_search=None):
    # Normalize input to a list of directories
    if isinstance(media_dirs, str):
        media_dirs = [media_dirs]
    elif not isinstance(media_dirs, list):
        raise ValueError("media_dirs must be a string or a list of strings")

    # List files from all selected directories (no recursion)
    #@st.cache_data
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
    try:
        media_names = [os.path.splitext(f)[0] for f in media_files]
        media_dates = [os.path.getmtime(path) for path in media_paths]
    except Exception as e:
        #scan_media_files.clear()
        st.warning("Error scanning media files")

    # Filter and sort UI - now with 3 columns
    search_col, sort_col, refresh_col = streamlit_component.columns([4, 3, 1])
    with search_col:
        search_query = st.text_input(
            t("media_selector_search_placeholder"),
            initial_search if initial_search is not None else "",
            key=f"search_{suffix}")
    with sort_col:
        sort_options = [
            t("media_selector_sort_option_az"),
            t("media_selector_sort_option_za"),
            t("media_selector_sort_option_date_oldest"),
            t("media_selector_sort_option_date_newest")
        ]
        sort_choice = st.selectbox(
            t("media_selector_sort_label"), sort_options, key=f"sort_{suffix}")
    with refresh_col:
        st.write("")  # Espacement vertical
        if st.button("🔄", key=f"refresh_{suffix}", help=t("media_selector_refresh_tooltip")):
            # Effacer le cache des thumbnails
            st.cache_data.clear()
            #scan_media_files.clear()
            # Rescanner les fichiers
            media_files, media_paths = scan_media_files()
            media_names = [os.path.splitext(f)[0] for f in media_files]
            media_dates = [os.path.getmtime(path) for path in media_paths]
            st.rerun()

    # Reste de la fonction inchangé...
    # Filter by search query
    if search_query:
        # Split the search query by commas and strip whitespace
        search_terms = [term.strip().lower() for term in search_query.split(',') if term.strip()]
        filtered_indices = [
            i for i, name in enumerate(media_names)
            if any(term in name.lower() for term in search_terms)
        ]
    else:
        filtered_indices = list(range(len(media_files)))

    # Sort based on user choice
    if sort_choice == t("media_selector_sort_option_az"):
        sorted_indices = sorted(filtered_indices, key=lambda i: media_names[i].lower())
    elif sort_choice == t("media_selector_sort_option_za"):
        sorted_indices = sorted(filtered_indices, key=lambda i: media_names[i].lower(), reverse=True)
    elif sort_choice == t("media_selector_sort_option_date_oldest"):
        sorted_indices = sorted(filtered_indices, key=lambda i: media_dates[i])
    else:  # t("media_selector_sort_option_date_newest")
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
                label="Media",
                images=thumbnails,
                captions=filtered_media_names,
                use_container_width=True,
                key=f"image_select_{suffix}"
            )
            if selected_thumb:
                selected_idx = thumbnails.index(selected_thumb)
                selected_media = filtered_media_paths[selected_idx]
        else:
            streamlit_component.write(t("media_selector_no_media_found"))

    return selected_media


def remote_media_selector(media_items, suffix, streamlit_component=st, initial_search=None):
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
        streamlit_component.write(t("media_selector_remote_no_media"))
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
            t("media_selector_remote_search_placeholder"),
            initial_search if initial_search is not None else "",
            key=f"search_remote_{suffix}")
    with sort_col:
        sort_options = [
            t("media_selector_sort_option_az"),
            t("media_selector_sort_option_za"),
            t("media_selector_sort_option_date_oldest"),
            t("media_selector_sort_option_date_newest")
        ]
        sort_choice = streamlit_component.selectbox(
            t("media_selector_sort_label"), sort_options, key=f"sort_remote_{suffix}")

    # Filtrer par requête de recherche
    if search_query:
        # Split the search query by commas and strip whitespace
        search_terms = [term.strip().lower() for term in search_query.split(',') if term.strip()]
        filtered_indices = [
            i for i, name in enumerate(media_names)
            if any(term in name.lower() for term in search_terms)
        ]
    else:
        filtered_indices = list(range(len(media_items)))

    # Trier selon le choix de l'utilisateur
    if sort_choice == t("media_selector_sort_option_az"):
        sorted_indices = sorted(filtered_indices, key=lambda i: media_names[i].lower())
    elif sort_choice == t("media_selector_sort_option_za"):
        sorted_indices = sorted(filtered_indices, key=lambda i: media_names[i].lower(), reverse=True)
    elif sort_choice == t("media_selector_sort_option_date_oldest"):
        sorted_indices = sorted(filtered_indices, key=lambda i: media_dates[i])
    else:  # t("media_selector_sort_option_date_newest")
        sorted_indices = sorted(filtered_indices, key=lambda i: media_dates[i], reverse=True)

    # Appliquer le tri
    filtered_media_urls = [media_urls[i] for i in sorted_indices]
    filtered_media_names = [media_names[i] for i in sorted_indices]
    filtered_media_items = [media_items[i] for i in sorted_indices]

    # Sélection d'image avec container scrollable
    selected_media = None
    with streamlit_component.container(height=400):
        if filtered_media_urls:
            selected_idx = image_select(
                t("media_selector_remote_choose_label"),
                images=filtered_media_urls,
                captions=filtered_media_names,
                use_container_width=True,
                return_value="index"
            )
            if selected_idx is not None:
                selected_media = filtered_media_items[selected_idx]
        else:
            streamlit_component.write(t("media_selector_remote_no_matching"))

    return selected_media
