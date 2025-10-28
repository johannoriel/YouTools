from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import os, re
import subprocess
from plugins.common import list_video_files
from plugins.transcript import TranscriptPlugin
from moviepy import VideoFileClip, TextClip, CompositeVideoClip


# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "shortextractor_tab": "Short Extractor",
    "shortextractor_header": "Extract Shorts from Videos",
    "shortextractor_select_video": "Select a video to extract shorts from",
    "shortextractor_transcribe": "Transcribe Video",
    "shortextractor_transcribing": "Transcribing video...",
    "shortextractor_select_range": "Select time range for short",
    "shortextractor_extract": "Extract Short",
    "shortextractor_extracting": "Extracting short...",
    "shortextractor_preview": "Short Preview",
    "shortextractor_zoom": "Zoom Factor",
    "shortextractor_center_x": "Center X (0-1)",
    "shortextractor_center_y": "Center Y (0-1)",
    "shortextractor_error": "Error during short extraction: ",
    "click_to_select": "Click on the text to select start and end times:",
    "set_as_start": "Set as Start Time",
    "set_as_end": "Set as End Time",
    "shortextractor_start_time": "Start time",
    "shortextractor_end_time": "End time",
    "shortextractor_select_start": "Select start time",
    "shortextractor_select_end": "Select end time",
    "shortextractor_suggest_timecode": "Suggest Timecode",
    "shortextractor_suggesting": "Suggesting timecode...",
    "shortextractor_extract_timecode": "Extract Timecode",
    "shortextractor_llm_response": "LLM Response:",
    "shortextractor_no_timecode": "No valid timecode found in the LLM response.",
    "shortextractor_sugestion" : "Suggest a thematic or a subject",
    "searchable_transcript" : "Search in transcript",
    "search_in_transcript" : "Term to search for",
    "set_as_start_time" : "Set as start time",
    "start_time_set" : "Start time set",
    "set_as_end_time" : "Set as end time",
    "start_end_set" : "End time set",
    "full_transcript" : "Transcript",
    "shortextractor_format916": "Convert to 9/16 format",
    "shortextractor_use_old_mode": "Use old zoom mode (instead of 9:16 stack)",
    "shortextractor_preview_short": "Preview Short",
    "shortextractor_previewer": "Previewing short...",
    "shortextractor_suggest_timecode_prompt": """Analyze the following video transcript and suggest a short, interesting segment (15-60 seconds) that could be extracted as a standalone short video.

Provide the start and end timecodes in the format HH:MM:SS,mmm.

Please respond with two timecodes: a start time and an end time, along with a brief explanation of why this segment would make a good short video.
""",
    "shortextractor_searchfor" : "Search speifically around the thematic or following subject : '{suggestion}'",
    "shortextractor_add_subtitles": "Add subtitles",
    "shortextractor_subtitle_position": "Subtitles position",
    "shortextractor_subtitle_top": "Top",
    "shortextractor_subtitle_bottom": "Bottom",
    "shortextractor_subtitle_size": "Subtitles size",
    "shortextractor_subtitle_bold": "Bold subtitles",
})

translations["fr"].update({
    "shortextractor_tab": "Extracteur de Shorts",
    "shortextractor_header": "Extraire des Shorts à partir de Vidéos",
    "shortextractor_select_video": "Sélectionner une vidéo pour extraire des shorts",
    "shortextractor_transcribe": "Transcrire la Vidéo",
    "shortextractor_transcribing": "Transcription de la vidéo en cours...",
    "shortextractor_select_range": "Sélectionner la plage de temps pour le short",
    "shortextractor_extract": "Extraire le Short",
    "shortextractor_extracting": "Extraction du short en cours...",
    "shortextractor_preview": "Aperçu du Short",
    "shortextractor_zoom": "Facteur de Zoom",
    "shortextractor_center_x": "Centre X (0-1)",
    "shortextractor_center_y": "Centre Y (0-1)",
    "shortextractor_error": "Erreur lors de l'extraction du short : ",
    "click_to_select": "Cliquez sur le texte pour sélectionner les temps de début et de fin :",
    "set_as_start": "Définir comme temps de début",
    "set_as_end": "Définir comme temps de fin",
    "shortextractor_start_time": "Temps de début",
    "shortextractor_end_time": "Temps de fin",
    "shortextractor_select_start": "Sélectionner le temps de début",
    "shortextractor_select_end": "Sélectionner le temps de fin",
    "shortextractor_suggest_timecode": "Suggérer un timecode",
    "shortextractor_suggesting": "Suggestion de timecode en cours...",
    "shortextractor_extract_timecode": "Extraire le Timecode",
    "shortextractor_llm_response": "Réponse du LLM :",
    "shortextractor_no_timecode": "Aucun timecode valide trouvé dans la réponse du LLM.",
    "shortextractor_sugestion" : "Suggérer une thématique ou un sujet",
    "searchable_transcript" : "Rechercher dans le transcript",
    "search_in_transcript" : "Terme à rechercher",
    "set_as_start_time" : "Définir comme temps de début",
    "start_time_set" : "Temps de début défini",
    "set_as_end_time" : "Définir comme temps de fin",
    "start_end_set" : "Temps de fin définis",
    "full_transcript" : "Transcription de la vidéo",
    "shortextractor_format916": "Conversion au format 9/16",
    "shortextractor_use_old_mode": "Utiliser l'ancien mode zoom (au lieu du stack 9:16)",
    "shortextractor_preview_short": "Prévisualiser le Short",
    "shortextractor_previewer": "Prévisualisation du short en cours...",
    "shortextractor_suggest_timecode_prompt": """Analyse la transcription vidéo suivante et suggérez un court segment intéressant (15-60 secondes) qui pourrait être extrait comme une courte vidéo autonome.

Fournis les codes temporels de début et de fin au format HH:MM,mmm.

Réponds avec deux codes temporels : un code temporel de début et un code temporel de fin, accompagnés d'une brève explication de pourquoi ce segment ferait une bonne courte vidéo.
""",
    "shortextractor_searchfor" : "Recherche spécifiquement autour des thématiques suivantes : '{suggestion}'",
    "shortextractor_add_subtitles": "Ajouter les sous-titres",
    "shortextractor_subtitle_position": "Position des sous-titres",
    "shortextractor_subtitle_top": "Haut",
    "shortextractor_subtitle_bottom": "Bas",
    "shortextractor_subtitle_size": "Taille des sous-titres",
    "shortextractor_subtitle_bold": "Sous-titres en gras",
})


class ShortextractorPlugin(Plugin):
    def get_config_fields(self):
        return {
            "zoom_factor": {
                "type": "number",
                "label": t("shortextractor_zoom"),
                "default": 1
            },
            "center_x": {
                "type": "number",
                "label": t("shortextractor_center_x"),
                "default": 0
            },
            "center_y": {
                "type": "number",
                "label": t("shortextractor_center_y"),
                "default": 0
            }
        }

    def get_tabs(self):
        return [{"name": t("shortextractor_tab"), "plugin": "shortextractor"}]

    def convert_srt_time_to_seconds(self, time_str):
        """Convert SRT time format to seconds with millisecond precision."""
        if isinstance(time_str, (int, float)):
            return time_str
        hours, minutes, seconds = time_str.split(':')
        seconds, milliseconds = seconds.split(',')
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
        return total_seconds

    def seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)
        secs = int(secs)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def extract_timecodes_from_llm(self, llm_response):
        """Extract two timecodes from LLM response using robust regex."""
        # Regex for HH:MM:SS,mmm format
        timecode_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})'
        timecodes = re.findall(timecode_pattern, llm_response)
        if len(timecodes) >= 2:
            return timecodes[0], timecodes[1]
        elif len(timecodes) == 1:
            return timecodes[0], timecodes[0]
        return None, None

    def parse_transcript(self, transcript):
        lines = transcript.split('\n')
        parsed = []
        current_entry = {}
        for line in lines:
            if ' --> ' in line:
                if current_entry:
                    parsed.append(current_entry)
                    current_entry = {}
                start, end = line.split(' --> ')
                current_entry['start'] = start
                current_entry['end'] = end
            elif line.strip() and not line[0].isdigit():
                if 'text' not in current_entry:
                    current_entry['text'] = line
                else:
                    current_entry['text'] += ' ' + line
        if current_entry:
            parsed.append(current_entry)
        return parsed

    def display_searchable_transcript(self, transcript):
        st.subheader(t("searchable_transcript"))
        if not isinstance(transcript, list):
            return
        if not "start" in transcript[0]:
            return
        col1, col2 = st.columns([1, 3])

        with col1:
            search_term = st.text_input(t("search_in_transcript"), "")

            if st.button(t("set_as_start_time")):
                self.set_time_from_search(search_term, 'start', transcript)

            if st.button(t("set_as_end_time")):
                self.set_time_from_search(search_term, 'end', transcript)

        with col2:
            full_transcript = "\n\n".join([f"{entry['start']} - {entry['end']}\n{entry['text']}" for entry in transcript])

            if search_term:
                pattern = re.compile(re.escape(search_term), re.IGNORECASE)
                matches = []

                for entry in transcript:
                    if pattern.search(entry['text']):
                        matches.append(entry)

                if matches:
                    st.write(f"{len(matches)} occurrence(s) found:")
                    for i, match in enumerate(matches, 1):
                        highlighted_text = pattern.sub(lambda m: f"**{m.group()}**", match['text'])
                        st.markdown(f"{match['start']} - {match['end']} {highlighted_text}")
                        st.markdown("---")
                else:
                    st.warning("Term not found")
                    st.text_area(t("full_transcript"), full_transcript, height=400)
            else:
                st.text_area(t("full_transcript"), full_transcript, height=400)

    def set_time_from_search(self, search_term, time_type, transcript):
        if not search_term:
            st.warning("Enter search term")
            return

        for i, entry in enumerate(transcript):
            if search_term.lower() in entry['text'].lower():
                if time_type == 'start':
                    st.session_state.start_index = i
                    st.session_state.start_time = entry['start']
                    st.success(f"Start time set: {entry['start']}")
                else:
                    st.session_state.end_index = i
                    st.session_state.end_time = entry['end']
                    st.success(f"End time set: {entry['end']}")
                return

        st.warning("Search term not found")

    def build_short_clip(self, input_file, start_time, end_time, zoom_factor, center_x, center_y, use_old_mode, format_916, add_subtitles, subtitle_position, subtitle_size, subtitle_bold, subclip):
        w, h = subclip.size
        duration = subclip.duration

        if use_old_mode:
            clip = subclip
            if zoom_factor != 1:
                zoom_w = int(w / zoom_factor)
                zoom_h = int(h / zoom_factor)
                cx_offset = int(center_x * (w - zoom_w) / 2)
                cy_offset = int(center_y * (h - zoom_h) / 2)
                clip = clip.cropped(x1=cx_offset, y1=cy_offset, x2=cx_offset + zoom_w, y2=cy_offset + zoom_h)
                clip = clip.resized(width=w, height=h)
            if format_916:
                target_w = int(h * 9 / 16)
                crop_x = int((w - target_w) / 2)
                clip = clip.cropped(x1=crop_x, y1=0, x2=crop_x + target_w, y2=h)
        else:
            # New 9:16 stack mode
            crop_w = int(h * 9 / 8)
            left = subclip.cropped(x1=0, y1=0, x2=min(crop_w, w), y2=h)
            right = subclip.cropped(x1=max(0, w - crop_w), y1=0, x2=w, y2=h)
            target_w = h  # portrait width = original height
            left = left.resized(width=target_w).without_audio()
            right = right.resized(width=target_w).without_audio()
            block_h = left.h
            total_h = 2 * block_h
            stacked = CompositeVideoClip([
                left.with_position((0, 0)),
                right.with_position((0, block_h))
            ], size=(target_w, total_h)).with_audio(subclip.audio)
            clip = stacked

        # Add subtitles
        if add_subtitles and 'transcript' in st.session_state:
            parsed_transcript = self.parse_transcript(st.session_state.transcript)
            subtitle_clips = []
            for entry in parsed_transcript:
                entry_start_abs = self.convert_srt_time_to_seconds(entry['start'])
                entry_end_abs = self.convert_srt_time_to_seconds(entry['end'])
                if entry_start_abs >= self.convert_srt_time_to_seconds(start_time) and entry_end_abs <= self.convert_srt_time_to_seconds(end_time):
                    adjusted_start = entry_start_abs - self.convert_srt_time_to_seconds(start_time)
                    adjusted_dur = entry_end_abs - entry_start_abs
                    txt = entry['text']
                    font_name = 'Arial-Bold' if subtitle_bold else 'Arial'
                    txt_clip = TextClip(
                        text=txt,
                        font=font_name,
                        font_size=subtitle_size,
                        color='white',
                        stroke_color='black',
                        stroke_width=4,
                        method='caption',
                        size=clip.size,
                        #align='center',
                        transparent=True
                    ).with_start(adjusted_start).with_duration(adjusted_dur)
                    if subtitle_position == "top":
                        txt_clip = txt_clip.with_position(('center', 'top'))
                    else:
                        txt_clip = txt_clip.with_position(('center', 'bottom'))
                    subtitle_clips.append(txt_clip)
            if subtitle_clips:
                clip = CompositeVideoClip([clip] + subtitle_clips)

        return clip

    def preview_short(self, input_file, start_time, end_time, zoom_factor, center_x, center_y, use_old_mode, format_916, add_subtitles, subtitle_position, subtitle_size, subtitle_bold):
        start_seconds = self.convert_srt_time_to_seconds(start_time)
        end_seconds = self.convert_srt_time_to_seconds(end_time)
        subclip = VideoFileClip(input_file).subclipped(start_seconds, end_seconds)
        clip = self.build_short_clip(input_file, start_time, end_time, zoom_factor, center_x, center_y, use_old_mode, format_916, add_subtitles, subtitle_position, subtitle_size, subtitle_bold, subclip)
        # Use preview as in movied
        st.write("Preview start: 0")
        st.write("Preview end: " + str(clip.duration))
        preview_clip = clip.subclipped(0, clip.duration)
        preview_clip.preview()
        preview_clip.close()
        clip.close()
        subclip.close()

    def extract_short(self, input_file, start_time, end_time, output_file, zoom_factor, center_x, center_y, use_old_mode, format_916,
                      add_subtitles=False, subtitle_position="top", subtitle_size=24, subtitle_bold=False):
        start_seconds = self.convert_srt_time_to_seconds(start_time)
        end_seconds = self.convert_srt_time_to_seconds(end_time)
        subclip = VideoFileClip(input_file).subclipped(start_seconds, end_seconds)
        clip = self.build_short_clip(input_file, start_time, end_time, zoom_factor, center_x, center_y, use_old_mode, format_916, add_subtitles, subtitle_position, subtitle_size, subtitle_bold, subclip)
        clip.write_videofile(output_file, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a",
                             remove_temp=True, logger=None)
        clip.close()
        subclip.close()
        return output_file

    def run(self, config):
        st.header(t("shortextractor_header"))

        # Initialize session state if needed
        if 'transcript' not in st.session_state:
            st.session_state.transcript = None
        if 'start_time' not in st.session_state:
            st.session_state.start_time = "00:00:00,000"
        if 'end_time' not in st.session_state:
            st.session_state.end_time = "00:01:00,000"
        if 'start_index' not in st.session_state:
            st.session_state.start_index = 0
        if 'end_index' not in st.session_state:
            st.session_state.end_index = 0
        if 'llm_response' not in st.session_state:
            st.session_state.llm_response = ""

        # Video selection
        work_directory = os.path.expanduser(config['common']['work_directory'])
        l1, l2, l3, _ = list_video_files(work_directory)
        videos = l1 + l2 + l3

        if not videos:
            st.info(f"No videos in {work_directory}")
            return

        selected_video = st.selectbox(t("shortextractor_select_video"), options=[v[0] for v in videos])
        selected_video_path = next(v[1] for v in videos if v[0] == selected_video)

        if st.button(t("shortextractor_transcribe")):
            with st.spinner(t("shortextractor_transcribing")):
                transcript_plugin = TranscriptPlugin("transcript", self.plugin_manager)
                st.session_state.transcript = transcript_plugin.transcribe_video(selected_video_path, "srt")

        if st.session_state.transcript:
            parsed_transcript = self.parse_transcript(st.session_state.transcript)
            if not isinstance(parsed_transcript, list) or not parsed_transcript:
                return
            if "start" not in parsed_transcript[0]:
                return

            self.display_searchable_transcript(parsed_transcript)

            options = [f"{entry['start']} - {entry['text'][:50]}..." for entry in parsed_transcript]

            col1, col2 = st.columns(2)
            with col1:
                start_index = st.selectbox(t("shortextractor_select_start"),
                                           options=options,
                                           index=st.session_state.start_index,
                                           key='start_select')
            with col2:
                end_index = st.selectbox(t("shortextractor_select_end"),
                                         options=options,
                                         index=st.session_state.end_index,
                                         key='end_select')

            # Update session state
            st.session_state.start_index = options.index(start_index)
            st.session_state.end_index = options.index(end_index)
            st.session_state.start_time = parsed_transcript[st.session_state.start_index]['start']
            st.session_state.end_time = parsed_transcript[st.session_state.end_index]['end']

            # Ensure end >= start
            if st.session_state.end_index < st.session_state.start_index:
                st.session_state.end_index = st.session_state.start_index
                st.session_state.end_time = parsed_transcript[st.session_state.end_index]['end']

            suggestion = st.text_input(t("shortextractor_sugestion"))

            col_suggest, col_extract = st.columns(2)
            with col_suggest:
                if st.button(t("shortextractor_suggest_timecode")):
                    with st.spinner(t("shortextractor_suggesting")):
                        suggest_theme = t("shortextractor_searchfor").format(suggestion=suggestion) if suggestion else ""
                        prompt = t("shortextractor_suggest_timecode_prompt") + suggest_theme
                        st.session_state.llm_response = self.process_with_llm(prompt, config['llm']['llm_sys_prompt'], st.session_state.transcript)

            st.text(t("shortextractor_llm_response"))
            st.text(st.session_state.llm_response)

            with col_extract:
                if st.button(t("shortextractor_extract_timecode")):
                    start_tc, end_tc = self.extract_timecodes_from_llm(st.session_state.llm_response)
                    if start_tc and end_tc:
                        st.session_state.start_time = start_tc
                        st.session_state.end_time = end_tc
                        st.success(f"Timecodes extracted: {start_tc} -> {end_tc}")
                        st.rerun()
                    else:
                        st.warning(t("shortextractor_no_timecode"))

            col3, col4 = st.columns(2)
            st.session_state.start_time = col3.text_input(t("shortextractor_start_time"), value=st.session_state.start_time, key='display_start_time')
            st.session_state.end_time = col4.text_input(t("shortextractor_end_time"), value=st.session_state.end_time, key='display_end_time')

            # Controls
            default_zoom = float(config['shortextractor'].get('zoom_factor', 1.5))
            default_center_x = float(config['shortextractor'].get('center_x', 0))
            default_center_y = float(config['shortextractor'].get('center_y', 0))

            col1, col2, col3 = st.columns([1, 1, 1])
            zoom_factor = col1.slider(t("shortextractor_zoom"), min_value=1.0, max_value=3.0, value=default_zoom, step=0.1)
            center_x = col2.slider(t("shortextractor_center_x"), min_value=-1.0, max_value=1.0, value=default_center_x, step=0.1)
            center_y = col3.slider(t("shortextractor_center_y"), min_value=-1.0, max_value=1.0, value=default_center_y, step=0.1)

            use_old_mode = st.checkbox(t("shortextractor_use_old_mode"), value=False)

            add_subtitles = col1.checkbox(t("shortextractor_add_subtitles"), value=True)
            subtitle_position = col2.selectbox(
                t("shortextractor_subtitle_position"),
                options=["top", "bottom"],
                format_func=lambda x: t(f"shortextractor_subtitle_{x}"),
                disabled=not add_subtitles
            )
            format_916 = col3.checkbox(t("shortextractor_format916"), value=True)
            if add_subtitles:
                subtitle_size = col2.slider(
                    t("shortextractor_subtitle_size"),
                    min_value=12,
                    max_value=192,
                    value=96,
                    step=2
                )
                subtitle_bold = col3.checkbox(t("shortextractor_subtitle_bold"), value=False)
            else:
                subtitle_size = 18
                subtitle_bold = False

            col_preview, col_extract = st.columns(2)
            with col_preview:
                if st.button(t("shortextractor_preview_short")):
                    with st.spinner(t("shortextractor_previewer")):
                        self.preview_short(selected_video_path, st.session_state.start_time, st.session_state.end_time,
                                           zoom_factor, center_x, center_y, use_old_mode, format_916,
                                           add_subtitles, subtitle_position, subtitle_size, subtitle_bold)

            with col_extract:
                if st.button(t("shortextractor_extract")):
                    with st.spinner(t("shortextractor_extracting")):
                        output_file = os.path.join(work_directory, f"short_{os.path.splitext(selected_video)[0]}.mp4")
                        result = self.extract_short(selected_video_path, st.session_state.start_time, st.session_state.end_time,
                                                    output_file, zoom_factor, center_x, center_y, use_old_mode, format_916,
                                                    add_subtitles, subtitle_position, subtitle_size, subtitle_bold)
                        if result == output_file:
                            st.success("Short extracted successfully!")
                            _, center, _ = st.columns([1, 1, 1])
                            center.video(output_file)
                        else:
                            st.error(result)
