from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import pandas as pd
import os
import shutil
from PIL import Image
import base64
from io import BytesIO
from widgets.image_generator import ImageGeneratorWidget
from pydub import AudioSegment

translations["en"].update({
    "chunk_editor": "Chunk Editor",
    "audio_filename": "Audio File",
    "video_filename": "Video File",
    "image": "Image",
    "text": "Text",
    "view_chunks": "View Selected Chunks",
    "chunk_details": "Chunk Details: {0}",
    "bulk_operation": "Bulk Operation",
    "delete_chunks": "Delete Chunks",
    "generate_image_prompts": "Generate Image Prompts",
    "generate_images": "Generate Images",
    "apply_operation": "Apply Operation",
    "chunks_deleted": "Deleted {0} chunks and renumbered remaining files.",
    "prompts_generated": "Generated image prompts for {0} chunks.",
    "images_generated": "Generated images for {0} chunks.",
    "duration": "Duration",
    "prompt": "Image Prompt",
})

translations["fr"].update({
    "chunk_editor": "Éditeur de Chunks",
    "audio_filename": "Nom du fichier audio",
    "video_filename": "Nom du fichier vidéo",
    "image": "Image",
    "text": "Texte",
    "view_chunks": "Afficher les Chunks Sélectionnés",
    "chunk_details": "Détails du Chunk : {0}",
    "bulk_operation": "Opération de masse",
    "delete_chunks": "Supprimer les Chunks",
    "generate_image_prompts": "Générer des Prompts d'Image",
    "generate_images": "Générer des Images",
    "apply_operation": "Appliquer l'Opération",
    "chunks_deleted": "Supprimé {0} chunks et renommé les fichiers restants.",
    "prompts_generated": "Prompts d'image générés pour {0} chunks.",
    "images_generated": "Images générées pour {0} chunks.",
    "duration": "Durée",
    "prompt": "Prompt d'Image",
})

class ChunkEditorWidget(Widget):
    def __init__(self, name, prefix, plugin_manager, output_dir):
        super().__init__(name, prefix, plugin_manager)
        self.output_dir = os.path.expanduser(output_dir)
        self.chunks_key = f"{self.prefix}_chunks_data"
        self.image_generator = ImageGeneratorWidget("image_generator", f"{self.prefix}_image", plugin_manager)

    def load_chunks(self):
        if self.chunks_key in st.session_state:
            return st.session_state[self.chunks_key]

        # Récupérer tous les fichiers avec leur numéro
        files_by_number = {}
        for f in os.listdir(self.output_dir):
            if f.startswith('chunk_'):
                parts = f.split('_')
                if len(parts) > 1:
                    try:
                        number = int(parts[1].split('.')[0])
                        ext = '.' + f.split('.')[-1]
                        if ext == '.txt' and f.endswith('.img_prompt'):
                            ext = '.img_prompt'
                        if number not in files_by_number:
                            files_by_number[number] = {}
                        files_by_number[number][ext] = f
                    except ValueError:
                        continue

        # Créer les données des chunks
        data = []
        max_number = max(files_by_number.keys()) + 1 if files_by_number else 0
        for i in range(max_number):
            chunk_data = {
                'filename': f'chunk_{i}',
                'audio': None,
                'audio_filename': None,
                'text': None,
                'text_path': None,
                'image': None,
                'video': None,
                'video_filename': None,
                'prompt': None,
                'prompt_path': None,
                'image_base64': None,
                'duration': None
            }
            if i in files_by_number:
                files = files_by_number[i]
                if '.wav' in files:
                    chunk_data['audio'] = os.path.join(self.output_dir, files['.wav'])
                    chunk_data['audio_filename'] = files['.wav']
                    try:
                        audio = AudioSegment.from_wav(chunk_data['audio'])
                        chunk_data['duration'] = f"{audio.duration_seconds:.1f}s"
                    except Exception as e:
                        st.warning(f"Erreur lors du calcul de la durée de l'audio {chunk_data['audio']}: {str(e)}")
                if '.txt' in files:
                    chunk_data['text_path'] = os.path.join(self.output_dir, files['.txt'])
                    try:
                        with open(chunk_data['text_path'], 'r', encoding='utf-8') as f:
                            chunk_data['text'] = f.read()
                    except Exception as e:
                        st.warning(f"Erreur lors de la lecture du fichier texte {chunk_data['text_path']}: {str(e)}")
                if '.png' in files:
                    chunk_data['image'] = os.path.join(self.output_dir, files['.png'])
                    try:
                        with Image.open(chunk_data['image']) as img:
                            img.thumbnail((100, 100))
                            buffered = BytesIO()
                            img.save(buffered, format="PNG")
                            chunk_data['image_base64'] = f"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
                    except Exception as e:
                        st.warning(f"Erreur lors du chargement de l'image {chunk_data['image']}: {str(e)}")
                if '.mp4' in files:
                    chunk_data['video'] = os.path.join(self.output_dir, files['.mp4'])
                    chunk_data['video_filename'] = files['.mp4']
                if '.img_prompt' in files:
                    chunk_data['prompt_path'] = os.path.join(self.output_dir, files['.img_prompt'])
                    try:
                        with open(chunk_data['prompt_path'], 'r', encoding='utf-8') as f:
                            chunk_data['prompt'] = f.read()
                    except Exception as e:
                        st.warning(f"Erreur lors de la lecture du fichier prompt {chunk_data['prompt_path']}: {str(e)}")
            data.append(chunk_data)

        df = pd.DataFrame(data)
        st.session_state[self.chunks_key] = df
        return df

    def delete_chunks(self, selected_rows):
        deleted_count = 0
        indices_to_delete = [int(row['filename'].split('_')[1]) for _, row in selected_rows.iterrows()]

        # Supprimer les fichiers des chunks sélectionnés
        for index in indices_to_delete:
            for ext in ['.wav', '.txt', '.png', '.mp4', '.img_prompt']:
                file_path = os.path.join(self.output_dir, f"chunk_{index}{ext}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1

        # Renommer les fichiers restants pour maintenir la continuité
        remaining_files = {
            'wav': sorted([f for f in os.listdir(self.output_dir) if f.endswith('.wav')]),
            'txt': sorted([f for f in os.listdir(self.output_dir) if f.endswith('.txt') and not f.endswith('.img_prompt')]),
            'png': sorted([f for f in os.listdir(self.output_dir) if f.endswith('.png') and f.startswith('chunk_')]),
            'mp4': sorted([f for f in os.listdir(self.output_dir) if f.endswith('.mp4') and f.startswith('chunk_')]),
            'img_prompt': sorted([f for f in os.listdir(self.output_dir) if f.endswith('.img_prompt')])
        }

        for ext in remaining_files:
            for i, old_name in enumerate(remaining_files[ext]):
                old_path = os.path.join(self.output_dir, old_name)
                new_path = os.path.join(self.output_dir, f"chunk_{i}{ext}")
                if old_path != new_path and os.path.exists(old_path):
                    shutil.move(old_path, new_path)

        # Recharger les chunks dans session_state
        del st.session_state[self.chunks_key]
        self.load_chunks()
        return deleted_count // 5  # Approximation du nombre de chunks supprimés

    def generate_image_prompts(self, selected_rows, config):
        total_prompts = len(selected_rows)
        processed = 0
        transcript = " ".join(row['text'] for _, row in selected_rows.iterrows() if row['text'])
        resume = self.process_with_llm(
            config['podcasttovideo']['summurize_transcript'].format(transcript=transcript),
            config['articletovideo']['image_sysprompt']
        )

        progress_bar = st.progress(0)
        for _, row in selected_rows.iterrows():
            if row['text']:
                context = config['podcasttovideo']['image_prompt'].format(text=row['text'], resume=resume)
                prompt = self.process_with_llm(context, config['articletovideo']['image_sysprompt'])
                prompt_path = os.path.join(self.output_dir, f"chunk_{row['filename'].split('_')[1]}.img_prompt")
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                processed += 1
            progress_bar.progress(processed / total_prompts)
        progress_bar.empty()
        # Recharger les chunks pour inclure les nouveaux prompts
        del st.session_state[self.chunks_key]
        self.load_chunks()
        return processed

    def generate_images(self, selected_rows):
        total_images = len(selected_rows)
        processed = 0
        progress_bar = st.progress(0)
        for _, row in selected_rows.iterrows():
            prompt_path = os.path.join(self.output_dir, f"chunk_{row['filename'].split('_')[1]}.img_prompt")
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                image_path = os.path.join(self.output_dir, f"chunk_{row['filename'].split('_')[1]}.png")
                self.image_generator.generate_image_direct(prompt, "16:9", output_path=image_path)
                processed += 1
            progress_bar.progress(processed / total_images)
        progress_bar.empty()
        # Recharger les chunks pour inclure les nouvelles images
        del st.session_state[self.chunks_key]
        self.load_chunks()
        return processed

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
        gd.configure_column("filename", t("filename"), headerCheckboxSelection=True,width=100)
        gd.configure_column("image_base64", t("image"), width=thumbnail_size + 20,
                            cellRenderer=image_renderer)
        gd.configure_column("text", t("text"), width=600)
        gd.configure_column("duration", t("duration"), width=100)
        gd.configure_column("prompt", t("prompt"), width=300)
        gd.configure_column("audio_filename", t("audio_filename"), width=150, hide=True)
        gd.configure_column("video_filename", t("video_filename"), width=150, hide=True)
        gd.configure_column("audio", hide=True)
        gd.configure_column("text_path", hide=True)
        gd.configure_column("video", hide=True)
        gd.configure_column("image", hide=True)
        gd.configure_column("prompt_path", hide=True)
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

        # Opérations de masse
        st.subheader(t("bulk_operation"))
        operation = st.selectbox(
            t("bulk_operation"),
            [t("delete_chunks"), t("generate_image_prompts"), t("generate_images")],
            key=f"{self.prefix}_bulk_operation"
        )
        if st.button(t("apply_operation"), key=f"{self.prefix}_apply_operation"):
            selected_rows = grid_response['selected_rows']
            if selected_rows is None or selected_rows.empty:
                st.warning("No chunks selected.")
                return

            if operation == t("delete_chunks"):
                deleted_count = self.delete_chunks(selected_rows)
                st.success(t("chunks_deleted").format(deleted_count))
                st.rerun()
            elif operation == t("generate_image_prompts"):
                processed_count = self.generate_image_prompts(selected_rows, config)
                st.success(t("prompts_generated").format(processed_count))
                st.rerun()
            elif operation == t("generate_images"):
                processed_count = self.generate_images(selected_rows)
                st.success(t("images_generated").format(processed_count))
                st.rerun()

        # Bouton pour afficher les chunks sélectionnés
        if st.button(t("view_chunks"), key=f"{self.prefix}_view_chunks"):
            selected_rows = grid_response['selected_rows']
            if selected_rows is None or selected_rows.empty:
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
