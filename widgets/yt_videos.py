from global_vars import translations, t
from app import Widget
import streamlit as st
from global_vars import translations, t
from youtube_api import YoutubeAPI
from youtube_db import get_videos, sync_videos, get_video_transcript, save_transcript, reset_database, auto_upgrade_database, update_video_keywords, delete_video
from datetime import datetime
from typing import List, Dict, Any
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import pandas as pd

translations["en"].update({
    "marketyoutube_header_videos": "YouTube Video Database",
    "marketyoutube_sync": "Sync Videos",
    "marketyoutube_syncing": "Synchronizing videos...",
    "marketyoutube_sync_complete": "Synchronization complete!",
    "marketyoutube_sync_transcripts": "Sync Transcripts",
    "marketyoutube_copy_transcripts": "Copy Transcripts",
    "marketyoutube_download_transcripts": "Download Transcripts",
    "marketyoutube_generate_transcripts": "Generate Transcripts",
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
    "marketyoutube_filter_keywords": "Filter by keywords"
})

translations["fr"].update({
    "marketyoutube_header_videos": "Base de données des vidéos YouTube",
    "marketyoutube_sync": "Synchroniser les vidéos",
    "marketyoutube_syncing": "Synchronisation des vidéos...",
    "marketyoutube_sync_complete": "Synchronisation terminée !",
    "marketyoutube_sync_transcripts": "Synchroniser les transcriptions",
    "marketyoutube_copy_transcripts": "Copier les transcriptions",
    "marketyoutube_download_transcripts": "Télécharger les transcriptions",
    "marketyoutube_generate_transcripts": "Générer les transcriptions",
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
    "marketyoutube_filter_keywords": "Filtrer par mots-clés"
})


class VideoDatabaseWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        self.youtube_api = YoutubeAPI(self.plugin_manager.config)

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

    def sync_transcripts(self, channel_id: str, youtube_api, config):
        """Synchronizes transcripts for all videos in the channel that lack them."""
        videos = get_videos()  # Retrieve all videos from the database
        total_videos = len(videos)
        processed = 0
        successes = 0
        errors = []

        with st.spinner(t("marketyoutube_syncing")):
            progress_bar = st.progress(0)

            for video in videos:
                current_transcript = get_video_transcript(video['video_id'])
                if not current_transcript:
                    try:
                        transcript, lang = youtube_api.get_transcript(
                            video['video_id'],
                            config['common']['language']
                        )
                        if transcript:
                            save_transcript(video['video_id'], transcript)
                            successes += 1
                        else:
                            errors.append(
                                f"{video['title']} ({video['video_id']}): No transcript available")
                    except Exception as e:
                        error_msg = f"{video['title']} ({video['video_id']}): {str(e)}"
                        errors.append(error_msg)

                processed += 1
                progress_bar.progress(processed / total_videos)

            progress_bar.empty()

            if total_videos > 0:
                st.success(t("marketyoutube_sync_complete"))
                st.write(
                    f"Transcripts synchronized successfully: {successes}/{total_videos}")
                if errors:
                    with st.expander("Error details"):
                        for error in errors:
                            st.write(error)
            else:
                st.info("No videos to synchronize.")

    def display_video_database(self, config, filter_type: str, keyword: str, keyword_filter: List[str] = None):
        videos = get_videos(filter_type, keyword, 0,
                            keyword_filter=keyword_filter)

        col1, col2, col3 = st.columns(3)
        total_videos = len(videos)
        col1.write(t("marketyoutube_video_count").format(total_videos))

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

        # Handle selected rows
        selected_rows = response['selected_rows']
        st.subheader("Actions on Selected Videos")

        # Initialize session state for keyword input
        if f"{self.prefix}_edit_keywords_selected" not in st.session_state:
            st.session_state[f"{self.prefix}_edit_keywords_selected"] = ""

        # Update keyword input with first selected row's keywords
        if selected_rows is not None and not selected_rows.empty:
            first_row_keywords = selected_rows.iloc[0]['keywords']
            if first_row_keywords != "--":
                st.session_state[f"{self.prefix}_edit_keywords_selected"] = first_row_keywords

        if selected_rows is not None and not selected_rows.empty:
            st.write(f"Selected videos: {len(selected_rows)}")

            # Boutons pour les actions
            col1, col2, col3, col4, col5 = st.columns(5)

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
                        st.warning(
                            "No transcripts available for selected videos.")

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
                        st.warning(
                            "No transcripts available for selected videos.")

            with col3:
                if st.button(t("marketyoutube_generate_transcripts"), key=f"{self.prefix}_generate_transcripts"):
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
                                        row['video_id'], config['common']['language']
                                    )
                                    if transcript:
                                        save_transcript(
                                            row['video_id'], transcript)
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

            with col5:
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

        else:
            st.write("No videos selected.")

    def display(self):
        config = self.plugin_manager.config
        st.header(t("marketyoutube_header_videos"))
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button(t("marketyoutube_sync"), key=f"{self.prefix}_sync_videos"):
                with st.spinner(t("marketyoutube_syncing")):
                    sync_videos(config['common']
                                ['channel_id'], self.youtube_api)
                    st.success(t("marketyoutube_sync_complete"))
        with col2:
            if st.button(t("marketyoutube_sync_transcripts"), key=f"{self.prefix}_sync_transcripts"):
                self.sync_transcripts(
                    config['common']['channel_id'], self.youtube_api, config)
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
        keyword = col2.text_input(
            t("marketyoutube_keyword"),
            key=f"{self.prefix}_keyword_videos"
        )

        all_keywords = set()
        for video in get_videos():
            all_keywords.update(video['keywords'])
        all_keywords = sorted(list(all_keywords))
        selected_keyword_filter = col3.multiselect(
            t("marketyoutube_filter_keywords"),
            options=all_keywords,
            key=f"{self.prefix}_keyword_filter_videos"
        )

        self.display_video_database(
            config,
            filter_options[filter_type],
            keyword,
            keyword_filter=selected_keyword_filter if selected_keyword_filter else None
        )
