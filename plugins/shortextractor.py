from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import os, re
from plugins.common import list_video_files
from plugins.transcript import TranscriptPlugin
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from lib.video_utils import extract_and_reformat_subclip, generate_karaoke_ass, burn_ass_subtitles  # New imports
import requests
from plugins.common import upload_video

# Import du composant
try:
    from code_editor import code_editor
except ImportError:
    st.error("Installez streamlit-code-editor : pip install streamlit-code-editor")
    st.stop()

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
    "shortextractor_use_old_subtitle": "Use old subtitle mode (MoviePy instead of ASS)",
    "shortextractor_preview_short": "Preview Short",
    "shortextractor_previewer": "Previewing short...",
    "shortextractor_suggest_timecode_prompt": """Analyze the following video transcript and suggest {num} short, interesting segments (15-60 seconds each) that could be extracted as standalone short videos.

For each segment, provide the start and end timecodes in the format HH:MM:SS,mmm.

Please respond with {num} pairs of timecodes: a start time and an end time for each, along with a brief explanation of why this segment would make a good short video.
""",
    "shortextractor_searchfor" : "Search speifically around the thematic or following subject : '{suggestion}'",
    "shortextractor_add_subtitles": "Add subtitles",
    "shortextractor_subtitle_position": "Subtitles position",
    "shortextractor_subtitle_top": "Top",
    "shortextractor_subtitle_bottom": "Bottom",
    "shortextractor_subtitle_size": "Subtitles size",
    "shortextractor_subtitle_bold": "Bold subtitles",
    "shortextractor_suggest_filename": "Suggest filename",
    "shortextractor_generating_title": "Generating title...",
    "shortextractor_edit_filename": "Edit filename:",
    "shortextractor_apply": "Apply",
    "shortextractor_regenerate": "Regenerate suggestion",
    "shortextractor_suggest_title_prompt": """Suggest a catchy, engaging title for a YouTube Short based on the following transcript segment:

{segment_text}

The title should be concise, click-optimized, with Title Case and spaces (example: Amazing AI Trick You Need To Try!). Respond only with the title, no extra text.""",
    "shortextractor_number_suggestions": "Number of suggestions",
    "shortextractor_current_suggestion": "Current suggestion",
    "shortextractor_of": "of",
    "preview_full": "Preview Full",
    "preview_start": "Preview Start",
    "preview_end": "Preview End",
    "preview_duration": "Preview duration (seconds)",
    "shortextractor_extract_all": "Extract All",
    "shortextractor_extracting_full": "Extracting full video...",
    "shortextractor_full_extracted": "Full video extracted successfully!",
    "directpublish_processing" : "Automatic publishing processing...",
    "publish_without_subtitles": "Publish without subtitles",
    "publish_with_subtitles": "Publish with subtitles",
    "auto_transcribing": "Automatic transcription in progress...",
    "removing_silences": "Removing silences...",
    "formatting_short": "Formatting to Short (9:16)...",
    "generating_content": "Generating title, description and tags...",
    "uploading_youtube": "Uploading to YouTube...",
    "directpublish_success": "Upload successful!",
    "directpublish_desc_prompt": "Generate an engaging and optimized description for a YouTube Short based on the following transcript. Add relevant hashtags at the end.",
    "shortextractor_custom_desc": "Custom description to insert:",
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
    "shortextractor_use_old_subtitle": "Utiliser l'ancien mode sous-titres (MoviePy au lieu d'ASS)",
    "shortextractor_preview_short": "Prévisualiser le Short",
    "shortextractor_previewer": "Prévisualisation du short en cours...",
    "shortextractor_suggest_timecode_prompt": """Analyse la transcription vidéo suivante et suggérez {num} courts segments intéressants (15-60 secondes chacun) qui pourraient être extraits comme des courtes vidéos autonomes.

Pour chaque segment, fournis les codes temporels de début et de fin au format HH:MM:SS,mmm.

Réponds avec {num} paires de codes temporels : un code temporel de début et un code temporel de fin pour chacun, accompagnés d'une brève explication de pourquoi ce segment ferait une bonne courte vidéo.
""",
    "shortextractor_searchfor" : "Recherche spécifiquement autour des thématiques suivantes : '{suggestion}'",
    "shortextractor_add_subtitles": "Ajouter les sous-titres",
    "shortextractor_subtitle_position": "Position des sous-titres",
    "shortextractor_subtitle_top": "Haut",
    "shortextractor_subtitle_bottom": "Bas",
    "shortextractor_subtitle_size": "Taille des sous-titres",
    "shortextractor_subtitle_bold": "Sous-titres en gras",
    "shortextractor_suggest_filename": "Suggérer un nom de fichier",
    "shortextractor_generating_title": "Génération du titre...",
    "shortextractor_edit_filename": "Modifier le nom de fichier:",
    "shortextractor_apply": "Appliquer",
    "shortextractor_regenerate": "Régénérer la suggestion",
    "shortextractor_suggest_title_prompt": """Suggérez un titre accrocheur et engageant pour un YouTube Short basé sur le segment de transcription suivant :

{segment_text}

Le titre doit être concis, optimisé pour les clics, en français, avec des majuscules et des espaces (exemple : Astuce IA Incroyable à Essayer !). Répondez uniquement avec le titre, sans texte supplémentaire.""",
    "shortextractor_number_suggestions": "Nombre de suggestions",
    "shortextractor_current_suggestion": "Suggestion actuelle",
    "shortextractor_of": "sur",
    "preview_full": "Prévisualiser Complet",
    "preview_start": "Prévisualiser Début",
    "preview_end": "Prévisualiser Fin",
    "preview_duration": "Durée de prévisualisation (secondes)",
    "shortextractor_extract_all": "Extraire tout",
    "shortextractor_extracting_full": "Extraction de la vidéo complète...",
    "shortextractor_full_extracted": "Vidéo complète extraite avec succès!",
    "publish_without_subtitles": "Publier sans sous-titres",
    "directpublish_processing" : "Publication automatique en cours...",
    "publish_with_subtitles": "Publier avec sous-titres",
    "auto_transcribing": "Transcription automatique en cours...",
    "removing_silences": "Suppression des silences...",
    "formatting_short": "Formatage en Short (9:16)...",
    "generating_content": "Génération du titre, description et tags...",
    "uploading_youtube": "Téléversement sur YouTube...",
    "directpublish_success": "Publication réussie !",
    "directpublish_desc_prompt": "Génère une description engageante et optimisée pour un YouTube Short basée sur cette transcription. Ajoute des hashtags pertinents à la fin.",
    "shortextractor_custom_desc": "Description personnalisée à insérer :",
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
        """Convert SRT time format to seconds with millisecond precision. Handles both ',' and '.' for ms."""
        if isinstance(time_str, (int, float)):
            return time_str
        # Normalize to ',' for ms
        time_str = time_str.replace('.', ',')
        hours, minutes, seconds = time_str.split(':')
        seconds, milliseconds = seconds.split(',')
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
        return total_seconds

    def seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT time format (uses ',' for ms)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)
        secs = int(secs)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def extract_timecodes_from_llm(self, llm_response, num_suggestions):
        """Extract multiple pairs of timecodes from LLM response using robust regex."""
        # Regex for HH:MM:SS,mmm format
        timecode_pattern = r'(\d{2}:\d{2}:\d{2},\d{3})'
        timecodes = re.findall(timecode_pattern, llm_response)
        # Group them into pairs: assume even number, start-end pairs
        pairs = []
        for i in range(0, len(timecodes), 2):
            if i + 1 < len(timecodes):
                pairs.append((timecodes[i], timecodes[i+1]))
            else:
                pairs.append((timecodes[i], timecodes[i]))  # If odd, duplicate last
        # Take up to num_suggestions
        return pairs[:num_suggestions]

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

    def format_compact_transcript(self, transcript):
        """Format the transcript compactly: timecodes and text on same line, no empty lines or numbers."""
        parsed = self.parse_transcript(transcript)
        compact_lines = []
        for entry in parsed:
            text = entry['text'].strip()
            if text:  # Skip empty text
                line = f"{entry['start']} - {entry['end']}: {text}"
                compact_lines.append(line)
        return '\n'.join(compact_lines)

    def compute_adjusted_times_from_selection(self, selected_text, full_compact):
        """Compute adjusted start and end times from potentially partial selection."""
        if not selected_text or not selected_text.strip():
            return None, None

        full_text = full_compact  # the whole string with \n
        index = full_text.find(selected_text)
        if index == -1:
            st.warning("Selection not found in transcript.")
            return None, None

        sel_end = index + len(selected_text)

        # For start
        # Find start of the line: last \n before index, or 0
        line_start = full_text.rfind('\n', 0, index) + 1 if full_text.rfind('\n', 0, index) != -1 else 0
        next_nl = full_text.find('\n', line_start)
        if next_nl == -1:
            next_nl = len(full_text)
        full_line = full_text[line_start:next_nl].rstrip('\n').strip()

        # Parse full_line
        match = re.match(r'^(\d{2}:\d{2}:\d{2},\d{3}) - (\d{2}:\d{2}:\d{2},\d{3}): (.*)$', full_line)
        if not match:
            st.warning("Could not parse start line.")
            return None, None

        start_srt, end_srt, text = match.groups()
        # text_start pos in full_line: len(start_srt + " - " + end_srt + ": ")
        prefix_len = len(start_srt) + len(" - ") + len(end_srt) + len(": ")
        text_start_in_line = line_start + prefix_len
        # selection start pos in text
        sel_pos_in_text = index - text_start_in_line
        text_len = len(text)
        if sel_pos_in_text < 0:
            prop = 0.0
        else:
            prop = sel_pos_in_text / text_len if text_len > 0 else 0.0

        start_sec = self.convert_srt_time_to_seconds(start_srt)
        end_sec = self.convert_srt_time_to_seconds(end_srt)
        dur = end_sec - start_sec
        adj_start_sec = start_sec + prop * dur
        adjusted_start_srt = self.seconds_to_srt_time(adj_start_sec)

        chars_before = full_text[text_start_in_line : index]

        # For end
        # Find start of the end line: last \n before sel_end, or 0
        last_line_start = full_text.rfind('\n', 0, sel_end) + 1 if full_text.rfind('\n', 0, sel_end) != -1 else 0
        line_end = full_text.find('\n', sel_end)
        if line_end == -1:
            line_end = len(full_text)
        full_last_line = full_text[last_line_start:line_end].rstrip('\n').strip()

        # Parse full_last_line
        match_end = re.match(r'^(\d{2}:\d{2}:\d{2},\d{3}) - (\d{2}:\d{2}:\d{2},\d{3}): (.*)$', full_last_line)
        if not match_end:
            st.warning("Could not parse end line.")
            return adjusted_start_srt, None

        start_srt_e, end_srt_e, text_e = match_end.groups()
        prefix_len_e = len(start_srt_e) + len(" - ") + len(end_srt_e) + len(": ")
        text_start_in_line_e = last_line_start + prefix_len_e
        sel_end_pos_in_text = sel_end - text_start_in_line_e
        text_len_e = len(text_e)
        if sel_end_pos_in_text < 0:
            prop_e = 0.0
        elif sel_end_pos_in_text > text_len_e:
            prop_e = 1.0
        else:
            prop_e = sel_end_pos_in_text / text_len_e if text_len_e > 0 else 0.0

        start_sec_e = self.convert_srt_time_to_seconds(start_srt_e)
        end_sec_e = self.convert_srt_time_to_seconds(end_srt_e)
        dur_e = end_sec_e - start_sec_e
        adj_end_sec = start_sec_e + prop_e * dur_e
        adjusted_end_srt = self.seconds_to_srt_time(adj_end_sec)

        chars_up_to_sel = full_text[text_start_in_line_e : sel_end]

        return adjusted_start_srt, adjusted_end_srt

    def build_short_clip(self, input_file, start_time, end_time, zoom_factor, center_x, center_y, use_old_mode, format_916, add_subtitles, subtitle_position, subtitle_size, subtitle_bold, subclip):
        w, h = subclip.size
        duration = subclip.duration
        clip_w, clip_h = w, h  # Sera ajusté si 9:16

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
                clip_w, clip_h = target_w, h
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
            clip_w, clip_h = target_w, total_h

        # Add subtitles (MoviePy fallback for preview/old mode)
        subtitle_clips = []
        start_sec = self.convert_srt_time_to_seconds(start_time)
        end_sec = self.convert_srt_time_to_seconds(end_time)

        if add_subtitles:
            if 'transcript' in st.session_state:
                # Phrase-level (basique)
                parsed_transcript = self.parse_transcript(st.session_state.transcript)
                for entry in parsed_transcript:
                    entry_start_abs = self.convert_srt_time_to_seconds(entry['start'])
                    entry_end_abs = self.convert_srt_time_to_seconds(entry['end'])
                    if entry_start_abs >= start_sec and entry_end_abs <= end_sec:
                        adjusted_start = entry_start_abs - start_sec
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
                            size=(clip_w, clip_h),
                            transparent=True
                        ).with_start(adjusted_start).with_duration(adjusted_dur)
                        if subtitle_position == "top":
                            txt_clip = txt_clip.with_position(('center', 'top'))
                        else:
                            txt_clip = txt_clip.with_position(('center', 'bottom'))
                        subtitle_clips.append(txt_clip)

            if subtitle_clips:
                clip = CompositeVideoClip([clip] + subtitle_clips, size=(clip_w, clip_h))

        return clip

    def preview_short(self, input_file, start_time, end_time, zoom_factor, center_x, center_y, use_old_mode, format_916, add_subtitles, subtitle_position, subtitle_size, subtitle_bold):
        start_seconds = self.convert_srt_time_to_seconds(start_time)
        end_seconds = self.convert_srt_time_to_seconds(end_time)
        subclip = VideoFileClip(input_file).subclipped(start_seconds, end_seconds)
        clip = self.build_short_clip(input_file, start_time, end_time, zoom_factor, center_x, center_y, use_old_mode, format_916, add_subtitles, subtitle_position, subtitle_size, subtitle_bold, subclip)
        st.write("Preview start: 0")
        st.write("Preview end: " + str(clip.duration))
        preview_clip = clip.subclipped(0, clip.duration)
        preview_clip.preview()
        preview_clip.close()
        clip.close()
        subclip.close()

    def extract_short(self, input_file, start_time, end_time, output_file, zoom_factor, center_x, center_y, use_old_mode, format_916,
                      add_subtitles=False, subtitle_position="bottom", subtitle_size=24, subtitle_bold=False, use_old_subtitle=False, lang="fr"):
        start_seconds = self.convert_srt_time_to_seconds(start_time)
        end_seconds = self.convert_srt_time_to_seconds(end_time)

        # Step 1: Extract and reformat subclip using utils
        temp_short = "temp_short.mp4"
        extract_and_reformat_subclip(input_file, start_seconds, end_seconds, temp_short, zoom_factor, center_x, center_y, use_old_mode, format_916)

        # Step 2: Handle subtitles
        if add_subtitles and not use_old_subtitle:
            ass_file = "temp_subs.ass"
            generate_karaoke_ass(temp_short, ass_file, subtitle_size, subtitle_bold, "middle", lang, max_line_chars=35)
            try:
                burn_ass_subtitles(temp_short, ass_file, output_file, subtitle_size, "middle")
            except Exception as e:
                st.error(str(e))
                # Fallback: rename temp to output
                os.rename(temp_short, output_file)
            finally:
                if os.path.exists(ass_file):
                    os.remove(ass_file)
                if os.path.exists(temp_short):
                    os.remove(temp_short)
        else:
            # Fallback: Load temp and add MoviePy subtitles
            clip_temp = VideoFileClip(temp_short)
            # Recreate subclip for build (since we have temp, but build expects original subclip; approximate)
            # Note: For old mode, we re-apply subtitles here
            original_subclip = VideoFileClip(input_file).subclipped(start_seconds, end_seconds)
            clip = self.build_short_clip(input_file, start_time, end_time, zoom_factor, center_x, center_y, use_old_mode, format_916, add_subtitles, subtitle_position, subtitle_size, subtitle_bold, original_subclip)
            clip.write_videofile(output_file, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a",
                                 remove_temp=True, logger=None)
            clip.close()
            clip_temp.close()
            original_subclip.close()
            os.remove(temp_short)

        return output_file

    def process_and_publish(self, video_path: str, work_directory: str, config: dict, with_subtitles: bool, custom_desc: str = ""):
        """Fonction commune pour les deux types de publication (Short complet)."""
        with st.spinner(t("directpublish_processing")):

            video_to_process = video_path

            # 1. Suppression des silences
            trimsilences_plugin = self.plugin_manager.get_plugin('trimsilences')
            if trimsilences_plugin:
                st.info(t("removing_silences"))
                result, reduction, orig_dur, final_dur = trimsilences_plugin.remove_silence(
                    video_to_process, work_directory)
                if isinstance(result, str) and ("erreur" in result.lower() or "error" in result.lower()):
                    st.error(result)
                    return
                video_to_process = result

            # 2. Transcription automatique
            st.info(t("auto_transcribing"))
            transcript_plugin = self.plugin_manager.get_plugin('transcript')
            st.session_state.transcript = transcript_plugin.transcribe_video(video_to_process, "srt")

            # 3. Durée totale
            clip = VideoFileClip(video_to_process)
            total_duration = clip.duration
            clip.close()
            start_time = "00:00:00,000"
            end_time = self.seconds_to_srt_time(total_duration)

            # 4. Paramètres optimisés pour Short complet
            zoom_factor = 1.2
            center_x = 0.5
            center_y = 0.4
            use_old_mode = False
            format_916 = True
            subtitle_position = "bottom"
            subtitle_size = 80
            subtitle_bold = True
            use_old_subtitle = False
            lang = config['common'].get('language', 'fr')

            base_name = os.path.splitext(os.path.basename(video_path))[0]
            temp_output = os.path.join(work_directory, f"temp_short_full_{base_name}.mp4")

            st.info(t("formatting_short"))
            self.extract_short(
                video_to_process, start_time, end_time, temp_output,
                zoom_factor, center_x, center_y, use_old_mode, format_916,
                add_subtitles=with_subtitles,
                subtitle_position=subtitle_position,
                subtitle_size=subtitle_size,
                subtitle_bold=subtitle_bold,
                use_old_subtitle=use_old_subtitle,
                lang=lang
            )

            # 5. Génération titre / description / tags
            st.info(t("generating_content"))
            parsed = self.parse_transcript(st.session_state.transcript)
            plain_transcript = " ".join([e['text'].strip() for e in parsed if e['text'].strip()])

            title_prompt = t("shortextractor_suggest_title_prompt").format(segment_text=plain_transcript[:1500])
            title = self.process_with_llm(title_prompt, config['llm']['llm_sys_prompt'], plain_transcript).strip()

            desc_prompt = t("directpublish_desc_prompt") + "\n\n" + plain_transcript[:2000]
            description = self.process_with_llm(desc_prompt, config['llm']['llm_sys_prompt'], plain_transcript)

            tag_prompt = t("directpublish_tag_generator")
            tags = self.process_with_llm(tag_prompt, config['llm']['llm_sys_prompt'], plain_transcript)
            extra_keywords = config.get('directpublish', {}).get('keywords', '').strip()
            if extra_keywords:
                tags = extra_keywords + ", " + tags

            introduction = config.get('directpublish', {}).get('introduction', '')
            signature = config.get('directpublish', {}).get('signature', '')
            parts = [introduction, description, custom_desc, signature]
            full_description = "\n\n".join([p for p in parts if p]).strip()

            # 6. Renommage final
            final_video = temp_output
            if with_subtitles:
                # Sanitization minimale des caractères interdits, conservation des espaces et de la casse
                safe_title = re.sub(r'[\\/:*?"<>|]', '-', title.strip())
                new_name = f"{safe_title}.mp4"
                new_path = os.path.join(work_directory, new_name)
                i = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(work_directory, f"{safe_title} ({i}).mp4")
                    i += 1
                os.rename(temp_output, new_path)
                final_video = new_path
            else:
                generic_path = os.path.join(work_directory, f"short_full_{base_name}.mp4")
                os.rename(temp_output, generic_path)
                final_video = generic_path

            # 7. Upload YouTube
            st.info(t("uploading_youtube"))
            category_id = "24"
            tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

            try:
                video_id = upload_video(
                    final_video, title, full_description, category_id, tags_list, "unlisted")
            except Exception:
                st.warning(t("directpublish_notags"))
                video_id = upload_video(
                    final_video, title, full_description, category_id, [], "unlisted")

            st.success(t("directpublish_success"))
            st.markdown(f"**Publication :** https://www.youtube.com/shorts/{video_id}")

            # 8. Webhooks
            webhook_urls = config.get('directpublish', {}).get('webhook_urls', '').strip().split('\n')
            webhook_urls = [u.strip() for u in webhook_urls if u.strip()]
            if webhook_urls:
                for webhook in webhook_urls:
                    try:
                        response = requests.post(webhook, json={"video_id": video_id})
                        if response.status_code == 200:
                            st.success(t("directpublish_webhook_triggered").format(webhook=webhook))
                        else:
                            st.warning(t("directpublish_webhook_not_triggered").format(
                                webhook=webhook, status_code=response.status_code))
                    except Exception as e:
                        st.error(t("directpublish_webhook_error").format(webhook=webhook, error=str(e)))

    def run(self, config):
        st.header(t("shortextractor_header"))

        # Initialize session state if needed
        if 'transcript' not in st.session_state:
            st.session_state.transcript = None
        if 'start_time' not in st.session_state:
            st.session_state.start_time = "00:00:00,000"
        if 'end_time' not in st.session_state:
            st.session_state.end_time = "00:01:00,000"
        if 'llm_response' not in st.session_state:
            st.session_state.llm_response = ""
        if 'texte_actuel' not in st.session_state:
            st.session_state.texte_actuel = ""
        if 'texte_selectionne' not in st.session_state:
            st.session_state.texte_selectionne = ""
        if 'short_generated' not in st.session_state:
            st.session_state.short_generated = False
        if 'output_file' not in st.session_state:
            st.session_state.output_file = ""
        if 'suggested_filename' not in st.session_state:
            st.session_state.suggested_filename = ""
        if 'suggested_timecodes' not in st.session_state:
            st.session_state.suggested_timecodes = []
        if 'current_timecode_index' not in st.session_state:
            st.session_state.current_timecode_index = 0
        if 'num_suggestions' not in st.session_state:
            st.session_state.num_suggestions = 3
        if 'preview_duration' not in st.session_state:
            st.session_state.preview_duration = 5.0

        # Video selection
        work_directory = os.path.expanduser(config['common']['work_directory'])
        l1, l2, l3, _ = list_video_files(work_directory)
        videos = l1 + l2 + l3

        if not videos:
            st.info(f"No videos in {work_directory}")
            return

        selected_video = st.selectbox(t("shortextractor_select_video"), options=[v[0] for v in videos], key="video_selector")
        selected_video_path = next(v[1] for v in videos if v[0] == selected_video)

        custom_desc = st.text_area(t("shortextractor_custom_desc"), "")

        col_transcribe, col_pub_no_sub, col_pub_sub = st.columns(3)

        with col_transcribe:
            if st.button(t("shortextractor_transcribe")):
                with st.spinner(t("shortextractor_transcribing")):
                    transcript_plugin = TranscriptPlugin("transcript", self.plugin_manager)
                    st.session_state.transcript = transcript_plugin.transcribe_video(selected_video_path, "srt")

        with col_pub_no_sub:
            if st.button(t("publish_without_subtitles")):
                self.process_and_publish(selected_video_path, work_directory, config, with_subtitles=False, custom_desc=custom_desc)

        with col_pub_sub:
            if st.button(t("publish_with_subtitles")):
                self.process_and_publish(selected_video_path, work_directory, config, with_subtitles=True, custom_desc=custom_desc)

        if st.session_state.transcript:
            compact_transcript = self.format_compact_transcript(st.session_state.transcript)
            if st.session_state.texte_actuel != compact_transcript:
                st.session_state.texte_actuel = compact_transcript

            st.subheader("Transcript Editor")
            # Configuration de l'éditeur
            response_dict = code_editor(
                st.session_state.texte_actuel,
                lang="text",
                theme="default",
                height=30,
                response_mode=["select", "blur"],  # Met à jour sur sélection ET perte de focus
                allow_reset=True,
                key="transcript_editor"
            )

            # Mise à jour du texte actuel si changé (via submit ou blur)
            if response_dict.get('type') in ['submit', 'blur'] and response_dict.get('text'):
                st.session_state.texte_actuel = response_dict['text']

            # Mise à jour de la sélection si changée et extraction des timecodes
            if response_dict.get('type') == 'selection' and response_dict.get('selected'):
                st.session_state.texte_selectionne = response_dict['selected']
                adjusted_start, adjusted_end = self.compute_adjusted_times_from_selection(
                    st.session_state.texte_selectionne, st.session_state.texte_actuel
                )
                if adjusted_start and adjusted_end:
                    st.session_state.start_time = adjusted_start
                    st.session_state.end_time = adjusted_end
                    st.success(f"Selected range: {adjusted_start} to {adjusted_end}")
                else:
                    st.warning("Could not parse selection times.")

            suggestion = st.text_input(t("shortextractor_sugestion"))

            # Slider pour le nombre de suggestions
            num_suggestions = st.slider(t("shortextractor_number_suggestions"), min_value=1, max_value=5, value=st.session_state.num_suggestions, key="num_suggestions_slider")
            st.session_state.num_suggestions = num_suggestions

            col_suggest, col_extract = st.columns(2)
            with col_suggest:
                if st.button(t("shortextractor_suggest_timecode")):
                    with st.spinner(t("shortextractor_suggesting")):
                        suggest_theme = t("shortextractor_searchfor").format(suggestion=suggestion) if suggestion else ""
                        prompt = t("shortextractor_suggest_timecode_prompt").format(num=num_suggestions) + suggest_theme
                        st.session_state.llm_response = self.process_with_llm(prompt, config['llm']['llm_sys_prompt'], st.session_state.transcript)
                        # Extract timecodes after suggestion
                        st.session_state.suggested_timecodes = self.extract_timecodes_from_llm(st.session_state.llm_response, num_suggestions)
                        st.session_state.current_timecode_index = 0
                        if st.session_state.suggested_timecodes:
                            start_tc, end_tc = st.session_state.suggested_timecodes[0]
                            st.session_state.start_time = start_tc
                            st.session_state.end_time = end_tc
                            st.success(f"First suggestion applied: {start_tc} -> {end_tc}")

            st.text(t("shortextractor_llm_response"))
            st.text(st.session_state.llm_response)

            with col_extract:
                if st.button(t("shortextractor_extract_timecode")):
                    if not st.session_state.suggested_timecodes:
                        st.warning(t("shortextractor_no_timecode"))
                    else:
                        # Cycle to next
                        current_index = st.session_state.current_timecode_index
                        next_index = (current_index + 1) % len(st.session_state.suggested_timecodes)
                        st.session_state.current_timecode_index = next_index
                        start_tc, end_tc = st.session_state.suggested_timecodes[next_index]
                        st.session_state.start_time = start_tc
                        st.session_state.end_time = end_tc
                        st.success(f"Suggestion {next_index + 1} {t('shortextractor_of')} {len(st.session_state.suggested_timecodes)} applied: {start_tc} -> {end_tc}")
                        st.rerun()

            # Display current suggestion info
            if st.session_state.suggested_timecodes:
                current_index = st.session_state.current_timecode_index
                st.info(f"{t('shortextractor_current_suggestion')} {current_index + 1} {t('shortextractor_of')} {len(st.session_state.suggested_timecodes)}")

            col3, col4 = st.columns(2)
            col3.text_input(t("shortextractor_start_time"), key="start_time")
            col4.text_input(t("shortextractor_end_time"), key="end_time")

            # Controls
            default_zoom = float(config['shortextractor'].get('zoom_factor', 1.5))
            default_center_x = float(config['shortextractor'].get('center_x', 0))
            default_center_y = float(config['shortextractor'].get('center_y', 0))

            col1, col2, col3 = st.columns([1, 1, 1])
            zoom_factor = col1.slider(t("shortextractor_zoom"), min_value=1.0, max_value=3.0, value=default_zoom, step=0.1)
            center_x = col2.slider(t("shortextractor_center_x"), min_value=-1.0, max_value=1.0, value=default_center_x, step=0.1)
            center_y = col3.slider(t("shortextractor_center_y"), min_value=-1.0, max_value=1.0, value=default_center_y, step=0.1)

            col_old_zoom, col_old_sub = st.columns(2)
            use_old_mode = col_old_zoom.checkbox(t("shortextractor_use_old_mode"), value=False)
            use_old_subtitle = col_old_sub.checkbox(t("shortextractor_use_old_subtitle"), value=False)

            col1, col2, col3 = st.columns(3)
            add_subtitles = col1.checkbox(t("shortextractor_add_subtitles"), value=True)
            subtitle_position = col2.selectbox(
                t("shortextractor_subtitle_position"),
                options=["top", "bottom"],
                format_func=lambda x: t(f"shortextractor_subtitle_{x}"),
                disabled=not add_subtitles
            )
            format_916 = col3.checkbox(t("shortextractor_format916"), value=True)
            if add_subtitles:
                col_size, col_bold = st.columns(2)
                subtitle_size = col_size.slider(
                    t("shortextractor_subtitle_size"),
                    min_value=12,
                    max_value=192,
                    value=96,
                    step=2
                )
                subtitle_bold = col_bold.checkbox(t("shortextractor_subtitle_bold"), value=False)
            else:
                subtitle_size = 18
                subtitle_bold = False

            # Preview duration slider
            preview_duration = st.slider(t("preview_duration"), min_value=1.0, max_value=30.0, value=st.session_state.preview_duration, key="preview_dur_slider")
            st.session_state.preview_duration = preview_duration

            # Preview and extract buttons
            col_full, col_start, col_end, col_extract, col_extract_all = st.columns(5)  # Changé à 5 colonnes
            with col_full:
                if st.button(t("preview_full")):
                    with st.spinner(t("shortextractor_previewer")):
                        self.preview_short(selected_video_path, st.session_state.start_time, st.session_state.end_time,
                                           zoom_factor, center_x, center_y, use_old_mode, format_916,
                                           add_subtitles, subtitle_position, subtitle_size, subtitle_bold)
            with col_start:
                if st.button(t("preview_start")):
                    with st.spinner(t("shortextractor_previewer")):
                        start_sec = self.convert_srt_time_to_seconds(st.session_state.start_time)
                        end_sec = start_sec + preview_duration
                        temp_end = self.seconds_to_srt_time(end_sec)
                        self.preview_short(selected_video_path, st.session_state.start_time, temp_end,
                                           zoom_factor, center_x, center_y, use_old_mode, format_916,
                                           add_subtitles, subtitle_position, subtitle_size, subtitle_bold)
            with col_end:
                if st.button(t("preview_end")):
                    with st.spinner(t("shortextractor_previewer")):
                        end_sec = self.convert_srt_time_to_seconds(st.session_state.end_time)
                        start_sec = end_sec - preview_duration
                        temp_start = self.seconds_to_srt_time(max(0, start_sec))
                        self.preview_short(selected_video_path, temp_start, st.session_state.end_time,
                                           zoom_factor, center_x, center_y, use_old_mode, format_916,
                                           add_subtitles, subtitle_position, subtitle_size, subtitle_bold)
            with col_extract:
                if st.button(t("shortextractor_extract")):
                    with st.spinner(t("shortextractor_extracting")):
                        output_file = os.path.join(work_directory, f"short_{os.path.splitext(selected_video)[0]}.mp4")
                        result = self.extract_short(selected_video_path, st.session_state.start_time, st.session_state.end_time,
                                                    output_file, zoom_factor, center_x, center_y, use_old_mode, format_916,
                                                    add_subtitles, subtitle_position, subtitle_size, subtitle_bold, use_old_subtitle, config['common'].get('language', 'fr'))
                        if result == output_file:
                            st.session_state.short_generated = True
                            st.session_state.output_file = output_file
                            st.session_state.suggested_filename = ""
                            st.success("Short extracted successfully!")
                        else:
                            st.error(result)

            # Ajout du bouton "Extraire tout"
            with col_extract_all:
                if st.button(t("shortextractor_extract_all")):
                    with st.spinner(t("shortextractor_extracting_full")):
                        # Récupérer la durée totale de la vidéo
                        video_clip = VideoFileClip(selected_video_path)
                        total_duration = video_clip.duration
                        video_clip.close()

                        # Définir le début à 0 et la fin à la durée totale
                        full_start_time = "00:00:00,000"
                        full_end_time = self.seconds_to_srt_time(total_duration)

                        output_file = os.path.join(work_directory, f"full_{os.path.splitext(selected_video)[0]}.mp4")
                        result = self.extract_short(selected_video_path, full_start_time, full_end_time,
                                                    output_file, zoom_factor, center_x, center_y, use_old_mode, format_916,
                                                    add_subtitles, subtitle_position, subtitle_size, subtitle_bold, use_old_subtitle, config['common'].get('language', 'fr'))
                        if result == output_file:
                            st.session_state.short_generated = True
                            st.session_state.output_file = output_file
                            st.session_state.suggested_filename = ""
                            st.success(t("shortextractor_full_extracted"))
                        else:
                            st.error(result)

            # Display generated short and filename suggestion
            if st.session_state.short_generated:
                st.subheader("Generated Short")
                _, center, _ = st.columns([1, 1, 1])
                center.video(st.session_state.output_file)

                if not st.session_state.suggested_filename:
                    if st.button(t("shortextractor_suggest_filename")):
                        with st.spinner(t("shortextractor_generating_title")):
                            parsed = self.parse_transcript(st.session_state.transcript)
                            start_sec = self.convert_srt_time_to_seconds(st.session_state.start_time)
                            end_sec = self.convert_srt_time_to_seconds(st.session_state.end_time)
                            segment_text = " ".join([
                                entry['text']
                                for entry in parsed
                                if self.convert_srt_time_to_seconds(entry['start']) >= start_sec
                                and self.convert_srt_time_to_seconds(entry['end']) <= end_sec
                            ])
                            prompt = t("shortextractor_suggest_title_prompt").format(segment_text=segment_text[:1000])
                            system_prompt = config['llm']['llm_sys_prompt']
                            suggested_title = self.process_with_llm(prompt, system_prompt, segment_text)
                            filename = suggested_title.strip() + ".mp4"
                            st.session_state.suggested_filename = filename
                        st.rerun()
                else:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        edited_filename = st.text_input(
                            t("shortextractor_edit_filename"),
                            value=st.session_state.suggested_filename,
                            key="edit_filename"
                        )
                    with col2:
                        if st.button(t("shortextractor_apply")):
                            new_path = os.path.join(
                                os.path.dirname(st.session_state.output_file),
                                edited_filename
                            )
                            os.rename(st.session_state.output_file, new_path)
                            st.session_state.output_file = new_path
                            st.session_state.suggested_filename = ""
                            st.success("Filename applied!")
                            st.rerun()

                    if st.button(t("shortextractor_regenerate")):
                        st.session_state.suggested_filename = ""
                        st.rerun()
