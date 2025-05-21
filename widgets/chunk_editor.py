from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import pandas as pd
import os
from PIL import Image
import base64
from io import BytesIO

translations["en"].update({
    "chunk_editor": "Chunk Editor",
    "audio_filename": "Audio File",
    "video_filename": "Video File",
    "image": "Image",
    "text": "Text",
    "view_chunks": "View Selected Chunks",
    "chunk_details": "Chunk Details: {0}"
})

translations["fr"].update({
    "chunk_editor": "Éditeur de Chunks",
    "audio_filename": "Nom du fichier audio",
    "video_filename": "Nom du fichier vidéo",
    "image": "Image",
    "text": "Texte",
    "view_chunks": "Afficher les Chunks Sélectionnés",
    "chunk_details": "Détails du Chunk : {0}"
})

class ChunkEditorWidget(Widget):
    def __init__(self, name, prefix, plugin_manager, output_dir):
        super().__init__(name, prefix, plugin_manager)
        self.output_dir = os.path.expanduser(output_dir)
        self.chunks_key = f"{self.prefix}_chunks_data"

    def load_chunks(self):
        if self.chunks_key in st.session_state:
            return st.session_state[self.chunks_key]

        audio_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.wav')])
        text_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.txt')])
        image_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.png') and f.startswith('chunk_')])
        video_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.mp4') and f.startswith('chunk_')])

        data = []
        max_files = max(len(audio_files), len(text_files), len(image_files), len(video_files))
        for i in range(max_files):
            chunk_data = {
                'filename': f'chunk_{i}',
                'audio': os.path.join(self.output_dir, audio_files[i]) if i < len(audio_files) else None,
                'audio_filename': audio_files[i] if i < len(audio_files) else None,
                'text': None,
                'text_path': os.path.join(self.output_dir, text_files[i]) if i < len(text_files) else None,
                'image': os.path.join(self.output_dir, image_files[i]) if i < len(image_files) else None,
                'video': os.path.join(self.output_dir, video_files[i]) if i < len(video_files) else None,
                'video_filename': video_files[i] if i < len(video_files) else None,
                'image_base64': None
            }
            if chunk_data['text_path']:
                try:
                    with open(chunk_data['text_path'], 'r', encoding='utf-8') as f:
                        chunk_data['text'] = f.read()
                except Exception as e:
                    st.warning(f"Erreur lors de la lecture du fichier texte {chunk_data['text_path']}: {str(e)}")
            if chunk_data['image']:
                try:
                    with Image.open(chunk_data['image']) as img:
                        img.thumbnail((100, 100))
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        chunk_data['image_base64'] = f"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
                except Exception as e:
                    st.warning(f"Erreur lors du chargement de l'image {chunk_data['image']}: {str(e)}")
            data.append(chunk_data)

        df = pd.DataFrame(data)
        st.session_state[self.chunks_key] = df
        return df

    def display(self, config):
        st.header(t("chunk_editor"))
        df = self.load_chunks()

        # Slider pour la taille des vignettes et la hauteur des lignes
        col1, col2 = st.columns(2)
        thumbnail_size = col1.slider(
            t("marketyoutube_thumbnail_size"),
            min_value=60, max_value=200, value=130, step=10,
            key=f"{self.prefix}_thumbnail_size"
        )
        row_height = col2.slider(
            "Row Height (px)",
            min_value=60, max_value=200, value=80, step=10,
            key=f"{self.prefix}_row_height"
        )

        # JavaScript pour rendre les images
        image_renderer = JsCode(f"""
            class ImageRenderer {{
                init(params) {{
                    this.eGui = document.createElement('div');
                    this.eGui.style.height = '{row_height}px';
                    this.eGui.style.display = 'flex';
                    this.eGui.style.alignItems = 'center';
                    if (params.value) {{
                        let img = document.createElement('img');
                        img.src = params.value;
                        img.style.height = '{thumbnail_size}px';
                        img.style.width = '{thumbnail_size}px';
                        img.style.objectFit = 'contain';
                        this.eGui.appendChild(img);
                    }}
                }}
                getGui() {{
                    return this.eGui;
                }}
            }}
        """)

        # Configurer la grille
        gd = GridOptionsBuilder.from_dataframe(df)
        gd.configure_column("image_base64", t("image"), width=thumbnail_size + 20,
                            cellRenderer=image_renderer)
        gd.configure_column("text", t("text"), flex=2)
        gd.configure_column("audio_filename", t("audio_filename"), width=150, hide=True)
        gd.configure_column("video_filename", t("video_filename"), width=150, hide=True)
        gd.configure_column("filename", hide=False)
        gd.configure_column("audio", hide=True)
        gd.configure_column("text_path", hide=True)
        gd.configure_column("video", hide=True)
        gd.configure_column("image", hide=True)
        gd.configure_selection('multiple', use_checkbox=True)
        gd.configure_grid_options(rowHeight=row_height)
        gridOptions = gd.build()

        # Afficher la grille
        grid_response = AgGrid(
            df,
            gridOptions=gridOptions,
            allow_unsafe_jscode=True,
            height=400,
            fit_columns_on_grid_load=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            key=f"{self.prefix}_chunk_grid"
        )

        # Bouton pour afficher les chunks sélectionnés
        if st.button(t("view_chunks"), key=f"{self.prefix}_view_chunks"):
            selected_rows = grid_response['selected_rows']
            if  selected_rows is None or selected_rows.empty:
                st.warning("No chunks selected.")
                return

            for _, row in selected_rows.iterrows():
                with st.expander(t("chunk_details").format(row['filename'])):
                    cols = st.columns(4)
                    if row['audio']:
                        with cols[0]:
                            st.audio(row['audio'], format='audio/wav')
                    if row['video']:
                        with cols[1]:
                            st.video(row['video'])
                    if row['image']:
                        with cols[2]:
                            st.image(row['image'])
                    if row['text']:
                        with cols[3]:
                            st.text_area("Text", row['text'], height=100, disabled=True)
