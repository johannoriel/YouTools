from global_vars import translations, t
from app import Plugin
import streamlit as st
import os, re
import subprocess
from plugins.common import list_video_files
from plugins.transcript import TranscriptPlugin
from plugins.ragllm import RagllmPlugin


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
        hours, minutes, seconds = time_str.split(':')
        seconds, milliseconds = seconds.split(',')
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
        return total_seconds

    def ffmpeg(self, ffmpeg_command):
        try:
            print("Commande ffmpeg:", " ".join(ffmpeg_command))
            result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
            print("Sortie STDOUT:", result.stdout)
            print("Sortie STDERR:", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Erreur ffmpeg:", e.stderr)
            return f"{t('shortextractor_error')}{e.stderr}"

    def extract_timecodes(self, llm_response, options):
        timecode_pattern = r'(\d{2}:\d{2}:\d{2}(?:,\d{3})?)'
        timecodes = re.findall(timecode_pattern, llm_response)
        if len(timecodes) >= 2:
            start_index = self.find_closest_index(options, timecodes[0])
            end_index = self.find_closest_index(options, timecodes[1])
            return start_index, end_index
        elif len(timecodes) == 1:
            index = self.find_closest_index(options, timecodes[0])
            return index, index
        else:
            return 0, len(options) - 1

    def find_closest_index(self, options, target):
        def extract_time(option):
            return option.split(' - ')[0] if isinstance(option, str) else option

        target_time = extract_time(target)

        closest_index = 0
        min_diff = float('inf')
        for i, option in enumerate(options):
            option_time = extract_time(option)

            # Convert times to seconds for comparison
            target_seconds = self.convert_srt_time_to_seconds(target_time) if isinstance(target_time, str) else target_time
            option_seconds = self.convert_srt_time_to_seconds(option_time)

            diff = abs(option_seconds - target_seconds)
            if diff < min_diff:
                min_diff = diff
                closest_index = i
        return closest_index

    def extract_short(self, input_file, start_time, end_time, output_file, zoom_factor, center_x, center_y, format_916,
        add_subtitles=False, subtitle_position="top", subtitle_size=24, subtitle_bold=False):
        start_seconds = self.convert_srt_time_to_seconds(start_time)
        end_seconds = self.convert_srt_time_to_seconds(end_time)
        duration = end_seconds - start_seconds

        videos_dir = os.path.dirname(input_file)
        input_filename = os.path.basename(input_file)
        zoom_filename = f"zoom_{input_filename}"
        zoom_file = os.path.join(videos_dir, zoom_filename)

        def build_ffmpeg_command(input, output, extra_options=None):
            command = [
                "ffmpeg", "-y",
                "-i", input,
                "-c:a", "aac",
            ]
            if extra_options:
                command.extend(extra_options)
            command.append(output)
            return command

        # Créer un fichier SRT temporaire pour le segment
        if add_subtitles and 'transcript' in st.session_state:
            temp_srt = os.path.join(videos_dir, "temp_segment.srt")
            parsed_transcript = self.parse_transcript(st.session_state.transcript)

            # Filtrer et ajuster les sous-titres pour le segment
            with open(temp_srt, 'w', encoding='utf-8') as f:
                subtitle_index = 1
                for entry in parsed_transcript:
                    entry_start = self.convert_srt_time_to_seconds(entry['start'])
                    entry_end = self.convert_srt_time_to_seconds(entry['end'])

                    if entry_start >= start_seconds and entry_end <= end_seconds:
                        # Ajuster les temps relatifs au début du segment
                        adjusted_start = self.format_srt_time(entry_start - start_seconds)
                        adjusted_end = self.format_srt_time(entry_end - start_seconds)

                        f.write(f"{subtitle_index}\n")
                        f.write(f"{adjusted_start} --> {adjusted_end}\n")
                        f.write(f"{entry['text']}\n\n")
                        subtitle_index += 1

        # Commande de zoom
        zoom_options = [
            "-ss", f"{start_seconds:.3f}",
            "-t", f"{duration:.3f}",
            "-vf", f"scale=iw/{zoom_factor}:ih/{zoom_factor}, pad=iw*{zoom_factor}:ih*{zoom_factor}:(ow-iw)/2:(oh-ih)/2",
        ]
        ffmpeg_command = build_ffmpeg_command(input_file, zoom_file, zoom_options)
        self.ffmpeg(ffmpeg_command)

        # Commande finale avec sous-titres si nécessaire
        if format_916:
            # Pour la position "top", on utilise Alignment=8 (haut-centre)
            # Pour la position "bottom", on utilise Alignment=2 (bas-centre)
            alignment = "6" if subtitle_position == "top" else "2"
            subtitle_y = "0" if subtitle_position == "top" else "(h-th-20)"
            bold_style = ",Bold=1" if subtitle_bold else ""
            subtitle_filter = f"subtitles='{temp_srt}':force_style='FontSize={subtitle_size},Alignment={alignment},MarginV={subtitle_y}{bold_style}'" if add_subtitles else ""
            crop_filter = f"crop='min(iw,ih)*9/16:min(iw,ih):((iw-min(iw,ih)*9/16)/2+iw/(4*{zoom_factor})*{center_x}):ih/2'"

            vf_filters = [crop_filter]
            if add_subtitles:
                vf_filters.append(subtitle_filter)

            final_options = ["-vf", ",".join(vf_filters)]
            ffmpeg_command = build_ffmpeg_command(zoom_file, output_file, final_options)
        else:
            alignment = "6" if subtitle_position == "top" else "2"
            subtitle_y = "0" if subtitle_position == "top" else "(h-th-20)"
            bold_style = ",Bold=1" if subtitle_bold else ""
            if add_subtitles:
                final_options = ["-vf", f"subtitles='{temp_srt}':force_style='FontSize={subtitle_size},Alignment={alignment}{bold_style}'"]
                ffmpeg_command = build_ffmpeg_command(zoom_file, output_file, final_options)
            else:
                ffmpeg_command = build_ffmpeg_command(zoom_file, output_file)

        result = self.ffmpeg(ffmpeg_command)

        # Nettoyage
        os.remove(zoom_file)
        if add_subtitles:
            os.remove(temp_srt)

        return output_file

    def format_srt_time(self, seconds):
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


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
                matches = list(pattern.finditer(full_transcript))

                if matches:
                    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
                    matches = []

                    for entry in transcript:
                        if pattern.search(entry['text']):
                            matches.append(entry)

                    if matches:
                        st.write(f"{len(matches)} occurrence(s) found:")
                        for i, match in enumerate(matches, 1):
                            highlighted_text = pattern.sub(lambda m: f"**{m.group()}**", match['text'])
                            #st.markdown(f"**Occurrence {i}:**")
                            st.markdown(f"{match['start']} - {match['end']} {highlighted_text}")
                            st.markdown("---")
                else:
                    st.warning(t("search_term_not_found").format(search_term=search_term))
                    st.text_area(t("full_transcript"), full_transcript, height=400)
            else:
                st.text_area(t("full_transcript"), full_transcript, height=400)

    def set_time_from_search(self, search_term, time_type, transcript):
        if not search_term:
            st.warning(t("enter_search_term"))
            return

        for i, entry in enumerate(transcript):
            if search_term.lower() in entry['text'].lower():
                if time_type == 'start':
                    st.session_state.start_index = i
                    st.session_state.start_time = entry['start']
                    st.success(t("start_time_set").format(time=entry['start']))
                else:
                    st.session_state.end_index = i
                    st.session_state.end_time = entry['end']
                    st.success(t("end_time_set").format(time=entry['end']))
                return

        st.warning(t("search_term_not_found").format(search_term=search_term))

    def run(self, config):
        st.header(t("shortextractor_header"))

        # Sélection de la vidéo
        work_directory = os.path.expanduser(config['common']['work_directory'])
        l1, l2, l3, _ = list_video_files(work_directory)
        videos = l1+l2+l3

        if not videos:
            st.info(f"{t('transcript_no_videos')} {work_directory}")
            return

        selected_video = st.selectbox(t("shortextractor_select_video"), options=[v[0] for v in videos])
        selected_video_path = next(v[1] for v in videos if v[0] == selected_video)

        if st.button(t("shortextractor_transcribe")):
            with st.spinner(t("shortextractor_transcribing")):
                transcript_plugin = TranscriptPlugin("transcript", self.plugin_manager)
                transcript = transcript_plugin.transcribe_video(selected_video_path, "srt", config['transcript']['whisper_path'], config['transcript']['whisper_model'], config['transcript']['ffmpeg_path'], config['common']['language'])
                st.session_state.transcript = transcript

        if 'transcript' in st.session_state:
            parsed_transcript = self.parse_transcript(st.session_state.transcript)
            if not isinstance(parsed_transcript, list):
                return
            if not parsed_transcript:
                return
            if not "start" in parsed_transcript[0]:
                return

            self.display_searchable_transcript(parsed_transcript)

            # Création des options pour les select boxes
            options = [f"{entry['start']} - {entry['text']}" for entry in parsed_transcript]

            # Initialisation des valeurs de session si elles n'existent pas
            if 'start_index' not in st.session_state:
                st.session_state.start_index = 0
            if 'end_index' not in st.session_state:
                st.session_state.end_index = len(options) - 1
            if 'start_time' not in st.session_state:
                st.session_state.start_time = parsed_transcript[st.session_state.start_index]['start']
            if 'end_time' not in st.session_state:
                st.session_state.end_time = parsed_transcript[st.session_state.end_index]['end']


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

            # Mise à jour des indices et des temps dans st.session_state
            st.session_state.start_index = options.index(start_index)
            st.session_state.end_index = options.index(end_index)
            st.session_state.start_time = parsed_transcript[st.session_state.start_index]['start']
            st.session_state.end_time = parsed_transcript[st.session_state.end_index]['end']

            # Assurer que l'index de fin n'est pas avant l'index de début
            if st.session_state.end_index < st.session_state.start_index:
                st.session_state.end_index = st.session_state.start_index
                st.session_state.end_time = parsed_transcript[st.session_state.end_index]['end']


            # Mise à jour des valeurs de timecode
            start_time = parsed_transcript[st.session_state.start_index]['start']
            end_time = parsed_transcript[st.session_state.end_index]['end']

            suggestion = st.text_input(t("shortextractor_sugestion"))

            if st.button(t("shortextractor_suggest_timecode")):
                with st.spinner(t("shortextractor_suggesting")):
                    ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)

                    suggest_theme = ""
                    if suggestion != "":
                        suggest_theme = t("shortextractor_searchfor").format(suggestion=suggestion)


                    prompt = t("shortextractor_suggest_timecode_prompt") + suggest_theme
                    print(prompt)
                    llm_response = ragllm_plugin.process_with_llm(prompt, config['ragllm']['llm_sys_prompt'], st.session_state.transcript)

                    st.text(t("shortextractor_llm_response"))
                    st.text(llm_response)

                    options = [f"{entry['start']} - {entry['text']}" for entry in parsed_transcript]

                    suggested_start_index, suggested_end_index = self.extract_timecodes(llm_response, options)
                    st.session_state.start_index = suggested_start_index
                    st.session_state.end_index = suggested_end_index
                    st.session_state.start_time = parsed_transcript[suggested_start_index]['start']
                    st.session_state.end_time = parsed_transcript[suggested_end_index]['end']
                    st.write(f"{st.session_state.start_time} -> {st.session_state.end_time}")

            col3,col4 = st.columns(2)
            # Affichage des timecodes sélectionnés
            st.session_state.start_time = col3.text_input(t("shortextractor_start_time"), value=st.session_state.start_time, key='display_start_time')
            st.session_state.end_time = col4.text_input(t("shortextractor_end_time"), value=st.session_state.end_time, key='display_end_time')



            # Assurez-vous que les valeurs sont des floats
            default_zoom = float(config['shortextractor'].get('zoom_factor', 1.5))
            default_center_x = float(config['shortextractor'].get('center_x', 0))
            default_center_y = float(config['shortextractor'].get('center_y', 0))

            col1, col2, col3 = st.columns([1, 1, 1])
            zoom_factor = col1.slider(t("shortextractor_zoom"), min_value=1.0, max_value=3.0, value=default_zoom, step=0.1)
            center_x = col2.slider(t("shortextractor_center_x"), min_value=-1.0, max_value=1.0, value=default_center_x, step=0.1)
            center_y = col3.slider(t("shortextractor_center_y"), min_value=-1.0, max_value=1.0, value=default_center_y, step=0.1)

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
                    max_value=36,
                    value=18,
                    step=2
                )
                subtitle_bold = col3.checkbox(t("shortextractor_subtitle_bold"), value=False)
            else:
                subtitle_size = 18
                subtitle_bold = True

            if st.button(t("shortextractor_extract")):
                with st.spinner(t("shortextractor_extracting")):
                    output_file = os.path.join(work_directory, f"short_{os.path.splitext(selected_video)[0]}.mp4")
                    st.write(f"Extracting {st.session_state.start_time} -> {st.session_state.end_time}")
                    result = self.extract_short(selected_video_path, st.session_state.start_time, st.session_state.end_time,
                        output_file, zoom_factor, center_x, center_y, format_916, add_subtitles, subtitle_position, subtitle_size, subtitle_bold)
                    _, center, _ = st.columns([1, 1, 1])
                    if result == output_file:
                        st.success("Short extracted successfully!")
                        center.video(output_file, muted=False)
                    else:
                        st.error(result)
