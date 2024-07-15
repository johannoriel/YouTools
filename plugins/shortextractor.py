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
})

class ShortextractorPlugin(Plugin):
    def get_config_fields(self):
        return {
            "zoom_factor": {
                "type": "number",
                "label": t("shortextractor_zoom"),
                "default": 1.2
            },
            "center_x": {
                "type": "number",
                "label": t("shortextractor_center_x"),
                "default": 0.5
            },
            "center_y": {
                "type": "number",
                "label": t("shortextractor_center_y"),
                "default": 0.5
            }
        }

    def get_tabs(self):
        return [{"name": t("shortextractor_tab"), "plugin": "shortextractor"}]

    def convert_srt_time_to_seconds(self, time_str):
        """Convert SRT time format to seconds."""
        hours, minutes, seconds = time_str.replace(',', '.').split(':')
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

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

    def extract_short(self, input_file, start_time, end_time, output_file, zoom_factor, center_x, center_y):
        start_seconds = self.convert_srt_time_to_seconds(start_time)
        end_seconds = self.convert_srt_time_to_seconds(end_time)
        duration = end_seconds - start_seconds

        videos_dir =  os.path.dirname(input_file)
        input_filename = os.path.basename(input_file)
        zoom_filename = f"zoom_{input_filename}"
        zoom_file = os.path.join(videos_dir, zoom_filename)
        print(zoom_file)

        ffmpeg_command = [
            "ffmpeg", "-y",
            "-ss", f"{start_seconds:.3f}",
            "-i", input_file,
            "-t", f"{duration:.3f}",
            "-vf", f"scale=iw/{zoom_factor}:ih/{zoom_factor}, pad=iw*{zoom_factor}:ih*{zoom_factor}:(ow-iw)/2:(oh-ih)/2",
            "-c:a", "copy",
            zoom_file
        ]
        self.ffmpeg(ffmpeg_command)
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-ss", f"{start_seconds:.3f}",
            "-i", zoom_file,
            "-t", f"{duration:.3f}",
            "-vf", f"crop='min(iw,ih)*9/16:min(iw,ih):((iw-min(iw,ih)*9/16)/2+iw/(4*{zoom_factor})*{center_x}):ih/2'",
            "-c:a", "copy",
            output_file
        ]
        self.ffmpeg(ffmpeg_command)
        os.remove(zoom_file)
        return output_file


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

            # Création des options pour les select boxes
            options = [f"{entry['start']} - {entry['text']}" for entry in parsed_transcript]

            col1, col2 = st.columns(2)
            with col1:
                start_index = st.selectbox(t("shortextractor_select_start"), options=options, key='start_select')

            # Mise à jour automatique de l'index de fin
            if 'start_select' in st.session_state:
                if options.index(st.session_state.get('end_select', options[-1])) < options.index(start_index):
                    end_index = options.index(start_index)
                else:
                    end_index = options.index(st.session_state.get('end_select', options[-1]))
            else:
                end_index = len(options) - 1

            with col2:
                end_index = st.selectbox(t("shortextractor_select_end"), options=options, key='end_select')

            # Extraction des timecodes sélectionnés
            start_time = parsed_transcript[options.index(start_index)]['start']
            end_time = parsed_transcript[options.index(end_index)]['end']

            st.session_state.start_time = start_time
            st.session_state.end_time = end_time

            if st.button(t("shortextractor_suggest_timecode")):
                with st.spinner(t("shortextractor_suggesting")):
                    ragllm_plugin = RagllmPlugin("ragllm", self.plugin_manager)

                    prompt = f"""Analyze the following video transcript and suggest a short, interesting segment (15-60 seconds) that could be extracted as a standalone short video. Provide the start and end timecodes in the format HH:MM:SS,mmm.

            Transcript:
            {st.session_state.transcript}

            Please respond with two timecodes: a start time and an end time, along with a brief explanation of why this segment would make a good short video."""

                    llm_response = ragllm_plugin.process_with_llm(prompt, config['ragllm']['llm_sys_prompt'], "", config['ragllm']['llm_model'])

                    st.text(t("shortextractor_llm_response"))
                    st.text(llm_response)

                    options = [f"{entry['start']} - {entry['text']}" for entry in parsed_transcript]
                    start_index, end_index = self.extract_timecodes(llm_response, options)
                    #st.write(f"{start_index} -> {end_index}")
                    #st.write(options)
                    start_time = parsed_transcript[start_index]['start']
                    end_time = parsed_transcript[end_index]['end']
                    #st.write(f"{start_time} -> {end_time}")

            col3,col4 = st.columns(2)
            # Affichage des timecodes sélectionnés
            col3.text_input(t("shortextractor_start_time"), value=start_time, key='display_start_time')
            col4.text_input(t("shortextractor_end_time"), value=end_time, key='display_end_time')



            # Assurez-vous que les valeurs sont des floats
            default_zoom = float(config['shortextractor'].get('zoom_factor', 1.2))
            default_center_x = float(config['shortextractor'].get('center_x', 0))
            default_center_y = float(config['shortextractor'].get('center_y', 0))

            col1, col2, col3 = st.columns([1, 1, 1])
            zoom_factor = col1.slider(t("shortextractor_zoom"), min_value=1.0, max_value=2.0, value=default_zoom, step=0.1)
            center_x = col2.slider(t("shortextractor_center_x"), min_value=-1.0, max_value=1.0, value=default_center_x, step=0.1)
            center_y = col3.slider(t("shortextractor_center_y"), min_value=-1.0, max_value=1.0, value=default_center_y, step=0.1)

            if st.button(t("shortextractor_extract")):
                with st.spinner(t("shortextractor_extracting")):
                    output_file = os.path.join(work_directory, f"short_{os.path.splitext(selected_video)[0]}.mp4")
                    result = self.extract_short(selected_video_path, start_time, end_time, output_file, zoom_factor, center_x, center_y)
                    if result == output_file:
                        st.success("Short extracted successfully!")
                        col2.video(output_file)
                    else:
                        st.error(result)
