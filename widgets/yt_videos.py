from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from lib.youtube_api import YoutubeAPI
from lib.youtube_db import get_videos, sync_videos, get_video_transcript, save_transcript, reset_database, auto_upgrade_database, update_video_keywords, delete_video
from datetime import datetime
from typing import List, Dict, Any
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import pandas as pd
import os, json

translations["en"].update({
    "marketyoutube_header_videos": "YouTube Video Database",
    "marketyoutube_sync": "Sync Videos",
    "marketyoutube_syncing": "Synchronizing videos...",
    "marketyoutube_sync_complete": "Synchronization complete!",
    "marketyoutube_sync_transcripts": "Sync Transcripts",
    "marketyoutube_copy_transcripts": "Copy Transcripts",
    "marketyoutube_download_transcripts": "Download Transcripts",
    "marketyoutube_get_transcripts": "Get Transcripts",
    "marketyoutube_generate_transcript": "Generate Transcript",
    "marketyoutube_suggest_keywords": "Suggest Keywords",
    "marketyoutube_edit_keywords": "Edit Keywords",
    "marketyoutube_save_keywords": "Save Keywords",
    "marketyoutube_delete_videos": "Delete Videos",
    "marketyoutube_thumbnail_size": "Thumbnail Size (px)",
    "marketyoutube_video_count": "Total videos: {0}",
    "marketyoutube_filter_label": "Filter by",
    "marketyoutube_filter_title": "Title",
    "marketyoutube_filter_title_desc": "Title and Description",
    "marketyoutube_filter_all": "All",
    "marketyoutube_keyword": "Search keyword",
    "marketyoutube_filter_keywords": "Filter by keywords",
    "marketyoutube_export_keywords": "Export Keywords (JSON)",
    "marketyoutube_export_success": "Keywords exported to {path}",
    "marketyoutube_export_error": "Export error: {error}",
})

translations["fr"].update({
    "marketyoutube_header_videos": "Base de données des vidéos YouTube",
    "marketyoutube_sync": "Synchroniser les vidéos",
    "marketyoutube_syncing": "Synchronisation des vidéos...",
    "marketyoutube_sync_complete": "Synchronisation terminée !",
    "marketyoutube_sync_transcripts": "Synchroniser les transcriptions",
    "marketyoutube_copy_transcripts": "Copier les transcriptions",
    "marketyoutube_download_transcripts": "Télécharger les transcriptions",
    "marketyoutube_get_transcripts": "Récupérer les transcriptions",
    "marketyoutube_generate_transcript": "Générer la transcription",
    "marketyoutube_suggest_keywords": "Suggérer des mots-clés",
    "marketyoutube_edit_keywords": "Modifier les mots-clés",
    "marketyoutube_save_keywords": "Enregistrer les mots-clés",
    "marketyoutube_delete_videos": "Supprimer les vidéos",
    "marketyoutube_thumbnail_size": "Taille des miniatures (px)",
    "marketyoutube_video_count": "Total des vidéos : {0}",
    "marketyoutube_filter_label": "Filtrer par",
    "marketyoutube_filter_title": "Titre",
    "marketyoutube_filter_title_desc": "Titre et description",
    "marketyoutube_filter_all": "Tout",
    "marketyoutube_keyword": "Mot-clé de recherche",
    "marketyoutube_filter_keywords": "Filtrer par mots-clés",
    "marketyoutube_export_keywords": "Exporter mots-clés (JSON)",
    "marketyoutube_export_success": "Mots-clés exportés vers {path}",
    "marketyoutube_export_error": "Erreur export : {error}",
})


class VideoDatabaseWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.youtube_api = YoutubeAPI(self.plugin_manager.config)
        self.transcript_plugin = self.plugin_manager.get_plugin('transcript')

    def format_count(self, count: int) -> str:
        """Formats a number into K/M if > 1000."""
        if count >= 1_000_000:
            return f"{count/1_000_000:.1f}M"
        elif count >= 1000:
            return f"{count/1000:.1f}K"
        return str(count)

    def suggest_keywords(self, title: str, description: str, transcript: str) -> List[str]:
        """Suggests keywords via LLM."""
        prompt = """
        Suggest 5-10 relevant keywords for a YouTube video based on the following:
        Title: {title}
        Description: {description}
        Transcript: {transcript}
        Return the keywords as a comma-separated list.
        """
        context = f"Title: {title}\nDescription: {description}\nTranscript: {transcript}"
        llm_response = self.process_with_llm(
            prompt.format(title=title, description=description,
                          transcript=transcript),
            "",
            context
        )
        return [kw.strip() for kw in llm_response.split(",")]

    def export_keywords_to_json(self, config):
        """Export simple liste mots-clés avec poids"""
        work_dir = config['common']['work_directory']
        output_path = os.path.join(work_dir, "keywords.json")

        try:
            # Compter les occurrences
            keyword_counts = {}
            for video in get_videos():
                if 'keywords' in video:
                    for kw in video['keywords']:
                        keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

            # Trier par poids décroissant
            sorted_kw = dict(sorted(
                keyword_counts.items(),
                key=lambda x: -x[1]
            ))

            # Écrire le JSON simple
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_kw, f, indent=2, ensure_ascii=False)

            st.success(t("marketyoutube_export_success").format(path=output_path))
        except Exception as e:
            st.error(t("marketyoutube_export_error").format(error=str(e)))

    def generate_transcript(self, video_id: str, title: str) -> str:
        """Generate transcript by downloading video and processing it with transcript plugin."""
        try:
            # Initialize transcript plugin if not already done
            if self.transcript_plugin is None:
                raise Exception("Transcript plugin not available")

            # Get video URL
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            # Download video
            work_directory = self.plugin_manager.config['common']['work_directory']
            from lib.video_utils import download_audio_with_auth
            video_path = download_audio_with_auth(video_url, work_directory, 'www.youtube.com_cookies.txt') # use "Get cookies.txt" extension

            # Transcribe video
            transcript = self.transcript_plugin.transcribe_video(video_path, "txt")

            # Save transcript to database
            if transcript:
                save_transcript(video_id, transcript)
                return True
            return False

        except Exception as e:
            st.error(f"Error generating transcript for {title}: {str(e)}")
            return False
        finally:
            # Nettoyage: supprimer le fichier vidéo temporaire s'il existe
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception as e:
                    st.error(f"Error deleting temporary video file: {str(e)}")

    def global_operations(self, config):
        """Display global operation buttons for video database management."""
        st.header(t("marketyoutube_header_videos"))
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button(t("marketyoutube_sync"), key=f"{self.prefix}_sync_videos"):
                with st.spinner(t("marketyoutube_syncing")):
                    sync_videos(config['common']['channel_id'], self.youtube_api)
                    st.success(t("marketyoutube_sync_complete"))
        with col2:
            if st.button(t("marketyoutube_export_keywords"), key=f"{self.prefix}_export_keywords"):
                self.export_keywords_to_json(config)
        with col3:
            if st.button("Reset Database Structure", key=f"{self.prefix}_reset_database"):
                with st.spinner("Resetting database..."):
                    reset_database()
                    st.success("Database structure reset successfully!")
        with col4:
            if st.button("Upgrade Database Structure", key=f"{self.prefix}_upgrade_database"):
                try:
                    auto_upgrade_database()
                    st.info("Database upgraded")
                except Exception as e:
                    print(f"Database error: {str(e)}")

    def display_video_database(self):
        """Display the video database with filters and return selected rows."""
        col1, col2, col3 = st.columns(3)
        filter_options = {
            t("marketyoutube_filter_title"): "title",
            t("marketyoutube_filter_title_desc"): "title_description",
            t("marketyoutube_filter_all"): "all"
        }

        filter_type = col1.selectbox(
            t("marketyoutube_filter_label"),
            options=list(filter_options.keys()),
            key=f"{self.prefix}_filter_type_videos"
        )
        search_keyword = col2.text_input(
            t("marketyoutube_keyword"),
            key=f"{self.prefix}_keyword_videos"
        )

        # Compter les occurrences de chaque mot-clé
        keyword_counts = {}
        for video in get_videos():
            for keyword in video['keywords']:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        # Trier d'abord par occurrence décroissante, puis par ordre alphabétique
        sorted_keywords = sorted(
            keyword_counts.items(),
            key=lambda item: (-item[1], item[0])  # -item[1] pour ordre décroissant
        )

        # Créer les options avec le format "mot-clé (occurrences)"
        keyword_options = [f"{keyword} ({count})" for keyword, count in sorted_keywords]
        raw_keywords = [keyword for keyword, count in sorted_keywords]

        selected_keyword_filter = col3.multiselect(
            t("marketyoutube_filter_keywords"),
            options=keyword_options,
            key=f"{self.prefix}_keyword_filter_videos"
        )

        # Pour récupérer les mots-clés sans les occurrences dans le filtre
        selected_keywords = [kw.split(" (")[0] for kw in selected_keyword_filter] if selected_keyword_filter else None

        videos = get_videos(filter_options[filter_type], search_keyword, 0,
                            keyword_filter=selected_keywords)

        col1, col2, col3 = st.columns(3)
        total_videos = len(videos)
        col1.write(t("marketyoutube_video_count").format(total_videos))
        if total_videos == 0:
            st.warning("No videos to display.")
            return None

        # Slider pour la taille des vignettes et la hauteur des lignes
        thumbnail_size = col2.slider(
            t("marketyoutube_thumbnail_size"),
            min_value=60, max_value=200, value=130, step=10,
            key=f"{self.prefix}_thumbnail_size"
        )
        row_height = col3.slider(
            "Row Height (px)",
            min_value=60, max_value=200, value=80, step=10,
            key=f"{self.prefix}_row_height"
        )

        # Prepare DataFrame for AgGrid
        df = pd.DataFrame(videos)
        df['keywords'] = df['keywords'].apply(
            lambda x: ", ".join(x) if x else "--")
        df['transcript_available'] = df['video_id'].apply(
            lambda x: "Yes" if get_video_transcript(x) else "No")
        df['published_at'] = df['published_at'].apply(
            lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else x)

        # JavaScript pour rendre les images et les liens
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

        link_renderer = JsCode("""
            class LinkRenderer {
                init(params) {
                    this.eGui = document.createElement('a');
                    this.eGui.href = params.data.url;
                    this.eGui.target = '_blank';
                    this.eGui.innerText = params.value;
                }
                getGui() {
                    return this.eGui;
                }
            }
        """)

        # Configure AgGrid
        grid_options = {
            "rowSelection": "multiple",
            "rowHeight": row_height,
            "columnDefs": [
                {
                    "field": "video_id",
                    "headerName": "Select",
                    "checkboxSelection": True,
                    "headerCheckboxSelection": True,
                    "width": 100,
                    "pinned": "left"
                },
                {
                    "field": "thumbnail_url",
                    "headerName": "Thumbnail",
                    "cellRenderer": image_renderer,
                    "width": thumbnail_size + 20
                },
                {
                    "field": "title",
                    "headerName": "Title",
                    "cellRenderer": link_renderer,
                    "flex": 2
                },
                {
                    "field": "published_at",
                    "headerName": "Published",
                    "width": 120
                },
                {
                    "field": "status",
                    "headerName": "Status",
                    "width": 100
                },
                {
                    "field": "keywords",
                    "headerName": "Keywords",
                    "flex": 1
                },
                {
                    "field": "transcript_available",
                    "headerName": "Transcript",
                    "width": 100
                }
            ],
            "defaultColDef": {
                "flex": 1,
                "sortable": True,
                "filter": True,
                "resizable": True
            }
        }

        # Display AgGrid
        response = AgGrid(
            df,
            gridOptions=grid_options,
            height=400,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            key=f"{self.prefix}_video_grid"
        )

        return response['selected_rows']

    def video_db_edit(self, selected_rows):
        """Handle editing operations for selected videos."""
        if selected_rows is None or selected_rows.empty:
            st.write("No videos selected.")
            return

        st.subheader("Actions on Selected Videos")
        st.write(f"Selected videos: {len(selected_rows)}")

        # Initialize session state for keyword input and selected video IDs if not present
        if f"{self.prefix}_edit_keywords_selected" not in st.session_state:
            st.session_state[f"{self.prefix}_edit_keywords_selected"] = ""
        if f"{self.prefix}_selected_video_ids" not in st.session_state:
            st.session_state[f"{self.prefix}_selected_video_ids"] = []

        # Get current selected video IDs
        current_video_ids = selected_rows['video_id'].tolist()

        # Check if the selection has changed
        selection_changed = current_video_ids != st.session_state[f"{self.prefix}_selected_video_ids"]

        # Update selected video IDs in session state
        st.session_state[f"{self.prefix}_selected_video_ids"] = current_video_ids

        # If selection has changed, update the keyword input with the first selected video's keywords
        if selection_changed and len(selected_rows) > 0:
            first_row_keywords = selected_rows.iloc[0]['keywords']
            st.session_state[f"{self.prefix}_edit_keywords_selected"] = first_row_keywords if first_row_keywords != "--" else ""

        # Boutons pour les actions
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            if st.button(t("marketyoutube_copy_transcripts"), key=f"{self.prefix}_copy_transcripts"):
                transcripts = []
                for _, row in selected_rows.iterrows():
                    transcript = get_video_transcript(row['video_id'])
                    if transcript:
                        transcripts.append(
                            f"Transcript for {row['title']}:\n{transcript}\n")
                if transcripts:
                    st.code("\n".join(transcripts))
                else:
                    st.warning("No transcripts available for selected videos.")

        with col2:
            if st.button(t("marketyoutube_download_transcripts"), key=f"{self.prefix}_download_transcripts"):
                transcripts = []
                for _, row in selected_rows.iterrows():
                    transcript = get_video_transcript(row['video_id'])
                    if transcript:
                        transcripts.append(
                            f"Transcript for {row['title']}:\n{transcript}\n")
                if transcripts:
                    st.download_button(
                        label="Download All Transcripts",
                        data="\n".join(transcripts),
                        file_name="selected_transcripts.txt",
                        mime="text/plain",
                        key=f"{self.prefix}_download_all_transcripts"
                    )
                else:
                    st.warning("No transcripts available for selected videos.")

        with col3:
            if st.button(t("marketyoutube_get_transcripts"), key=f"{self.prefix}_get_transcripts"):
                total = len(selected_rows)
                processed = 0
                successes = 0
                errors = []
                with st.spinner("Generating transcripts..."):
                    progress_bar = st.progress(0)
                    for _, row in selected_rows.iterrows():
                        if not get_video_transcript(row['video_id']):
                            try:
                                transcript, lang = self.youtube_api.get_transcript(
                                    row['video_id'], self.plugin_manager.config['common']['language']
                                )
                                if transcript:
                                    save_transcript(row['video_id'], transcript)
                                    successes += 1
                                else:
                                    errors.append(
                                        f"{row['title']} ({row['video_id']}): No transcript available")
                            except Exception as e:
                                errors.append(
                                    f"{row['title']} ({row['video_id']}): {str(e)}")
                        processed += 1
                        progress_bar.progress(processed / total)
                    progress_bar.empty()
                st.success(f"Transcripts generated: {successes}/{total}")
                if errors:
                    with st.expander("Errors"):
                        for error in errors:
                            st.write(error)
                st.rerun()

        with col4:
            if st.button(t("marketyoutube_generate_transcript"), key=f"{self.prefix}_generate_transcript"):
                total = len(selected_rows)
                processed = 0
                successes = 0
                errors = []
                with st.spinner("Generating transcripts from video..."):
                    progress_bar = st.progress(0)
                    for _, row in selected_rows.iterrows():
                        if not get_video_transcript(row['video_id']):
                            try:
                                if self.generate_transcript(row['video_id'], row['title']):
                                    successes += 1
                                else:
                                    errors.append(f"{row['title']} ({row['video_id']}): Generation failed")
                            except Exception as e:
                                errors.append(f"{row['title']} ({row['video_id']}): {str(e)}")
                        processed += 1
                        progress_bar.progress(processed / total)
                    progress_bar.empty()
                st.success(f"Transcripts generated: {successes}/{total}")
                if errors:
                    with st.expander("Errors"):
                        for error in errors:
                            st.write(error)

        with col5:
            if st.button(t("marketyoutube_suggest_keywords"), key=f"{self.prefix}_suggest_keywords"):
                total = len(selected_rows)
                processed = 0
                with st.spinner("Suggesting keywords..."):
                    progress_bar = st.progress(0)
                    for _, row in selected_rows.iterrows():
                        transcript = get_video_transcript(
                            row['video_id']) or ""
                        suggested_keywords = self.suggest_keywords(
                            row['title'], row['description'], transcript
                        )
                        update_video_keywords(
                            row['video_id'], suggested_keywords)
                        processed += 1
                        progress_bar.progress(processed / total)
                    progress_bar.empty()
                st.success(f"Keywords suggested for {total} videos.")
                st.rerun()

        with col6:
            if st.button(t("marketyoutube_delete_videos"), key=f"{self.prefix}_delete_videos"):
                total = len(selected_rows)
                with st.spinner("Deleting videos..."):
                    for _, row in selected_rows.iterrows():
                        delete_video(row['video_id'])
                    st.success(f"Deleted {total} videos.")
                    st.rerun()

        col1, col2 = st.columns([4, 1])
        with col1:
            new_keywords = st.text_input(
                t("marketyoutube_edit_keywords"),
                key=f"{self.prefix}_edit_keywords_selected",
                placeholder="Enter keywords (comma-separated)"
            )
        with col2:
            if st.button(t("marketyoutube_save_keywords"), key=f"{self.prefix}_save_keywords_selected"):
                if new_keywords:
                    updated_keywords = [
                        kw.strip() for kw in new_keywords.split(",") if kw.strip()]
                    for _, row in selected_rows.iterrows():
                        update_video_keywords(
                            row['video_id'], updated_keywords)
                    st.success(
                        f"Keywords updated for {len(selected_rows)} videos.")
                    st.rerun()
                else:
                    st.warning("Please enter keywords to update.")

    def display(self):
        """Main display method coordinating global operations, database display, and editing."""
        config = self.plugin_manager.config
        self.global_operations(config)
        selected_rows = self.display_video_database()
        self.video_db_edit(selected_rows)
