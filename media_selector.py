import streamlit as st
import os
from PIL import Image
import cv2
from streamlit_image_select import image_select

# Fonction pour générer une miniature


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
            img.thumbnail((200, 200))
        except:
            img = Image.new('RGB', (200, 200), color='gray')
    return img

# Fonction du sélecteur de médias


def media_selector(media_dir, extensions, recursive=False, streamlit_component=st):
    # Lister les fichiers selon récursivité
    media_files = []
    media_paths = []
    if recursive:
        for root, _, files in os.walk(media_dir):
            for f in files:
                if any(f.lower().endswith(ext.lower()) for ext in extensions):
                    media_files.append(f)
                    media_paths.append(os.path.join(root, f))
    else:
        media_files = [f for f in os.listdir(media_dir) if any(
            f.lower().endswith(ext.lower()) for ext in extensions)]
        media_paths = [os.path.join(media_dir, f) for f in media_files]

    media_names = [os.path.splitext(f)[0] for f in media_files]
    media_dates = [os.path.getmtime(path) for path in media_paths]

    # Filtre de recherche et tri sur une seule ligne
    search_col, sort_col = streamlit_component.columns(2)
    with search_col:
        search_query = st.text_input(
            "Rechercher un média (ex. 'ru')", "", key="search")
    with sort_col:
        sort_options = [
            "Alphabétique (A-Z)",
            "Alphabétique (Z-A)",
            "Date (plus ancien au plus récent)",
            "Date (plus récent au plus ancien)"
        ]
        sort_choice = st.selectbox("Trier par :", sort_options, key="sort")

    # Filtrer les médias selon la recherche
    if search_query:
        filtered_indices = [i for i, name in enumerate(
            media_names) if search_query.lower() in name.lower()]
    else:
        filtered_indices = list(range(len(media_files)))

    # Trier les médias selon l'option choisie
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

    # Appliquer le tri aux listes
    filtered_media_paths = [media_paths[i] for i in sorted_indices]
    filtered_media_names = [media_names[i] for i in sorted_indices]

    # Générer les miniatures (mise en cache pour performance)
    @st.cache_data
    def load_thumbnails(paths):
        return [get_thumbnail(path) for path in paths]

    # Charger les miniatures uniquement pour les médias filtrés et triés
    thumbnails = load_thumbnails(filtered_media_paths)

    # Conteneur scrollable pour la grille
    selected_media = None
    with streamlit_component.container(height=400):
        if filtered_media_paths:  # Vérifier s'il y a des médias après filtrage
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
            streamlit_component.write(
                "Aucun média ne correspond à votre recherche.")

    return selected_media
