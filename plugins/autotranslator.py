from global_vars import translations, t
from app import Plugin
from plugins.common import upload_video, get_credentials, list_all_video_files
from plugins.ragllm import RagllmPlugin

import streamlit as st
import os
import subprocess
import yt_dlp
from yt_dlp.utils import download_range_func
from googleapiclient.discovery import build
import ffmpeg
import math


from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

from plugins.common import get_category_id

def debug_youtube_object(yt):
    print("Available attributes and methods in the YouTube object:")
    for attr in dir(yt):
        try:
            value = getattr(yt, attr)
            #print(f"{attr}: {value}")
            print(f"{attr}")
        except Exception as e:
            print(f"{attr}: Could not retrieve value - {str(e)}")

def convert_time_to_seconds(time_str):
    # Convert a time string "mm:ss" to seconds
    if isinstance(time_str, str):
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds
    elif isinstance(time_str, int):
        return time_str
    else:
        return 0

def extract_video_section(input_path, output_dir, start_time=None, end_time=None, video_length=None):
    # If no start_time and end_time are provided, return the full video path
    if start_time is None and end_time is None:
        return input_path

    # Convert time from "mm:ss" to seconds
    start_sec = convert_time_to_seconds(start_time) if start_time else 0
    end_sec = convert_time_to_seconds(end_time) if end_time else video_length

    # Extract the video section using moviepy
    new_file_name = os.path.join(output_dir, "section_" + os.path.basename(input_path))
    print(f"Extracting section {start_sec} -> {end_sec}...")
    ffmpeg_extract_subclip(input_path, start_sec, end_sec, targetname=new_file_name)
    print(f"Section extracted {new_file_name}")

    # Delete the full downloaded video to save space
    #os.remove(input_path) #debug only

    return new_file_name

def download_video_pytube(url, output_dir, start_time=None, end_time=None):
    # Download the full video using pytube
    yt = YouTube(url)
    # Extract video information
    video_info = {
        "title": yt.title,
        "description": yt.description,
        "tags": yt.keywords,
        "category": 'Unknown'
    }

    # Check if metadata is available
    if hasattr(yt, 'metadata') and yt.metadata:
        # Since yt.metadata is an object, try to find 'category' if it exists
        # Note: YouTubeMetadata may not have 'category', adapt as needed based on actual content
        for data in yt.metadata.raw_metadata:
            if 'category' in data:
                video_info['category'] = data['category']
                break

    video = yt.streams.filter(file_extension='mp4').first()
    filename = video.download(output_path=output_dir)

    new_file_name = extract_video_section(filename, output_dir, start_time, end_time, video_length=yt.length)

    return new_file_name, video_info

def download_video_dlp(url, output_dir, start_time=None, end_time=None):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, 'downloaded_video.%(ext)s')
    }

    #if start_time is not None or end_time is not None:
    #    ydl_opts['download_ranges'] = download_range_func(None, [(start_time, end_time)])
    #    ydl_opts['force_keyframes_at_cuts'] = True

    print(ydl_opts)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    new_file_name = extract_video_section(filename, output_dir, start_time, end_time)
    return new_file_name, info

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "autotranslator_tab": "Auto Translator",
    "autotranslator_header": "YouTube Video Auto Translator",
    "autotranslator_url_input": "Enter YouTube Video URL:",
    "autotranslator_enhance_checkbox": "Enhance video (zoom and rotate)",
    "autotranslator_zoom_factor": "Max zoom factor (1.0 to 1.1):",
    "autotranslator_rotation_angle": "Max rotation angle (degrees):",
    "autotranslator_process_button": "Process Video",
    "autotranslator_download_success": "Video downloaded successfully : ",
    "autotranslator_translation_success": "Video translated successfully : ",
    "autotranslator_enhancement_success": "Video enhanced successfully!",
    "autotranslator_upload_success": "Video successfully published! Video URL: https://www.youtube.com/watch?v={video_id}, Edition : https://studio.youtube.com/video/{video_id}/edit",
    "autotranslator_error": "An error occurred: ",
    "autotranslator_translation_sonitranslate": "SoniTranslate directory:",
    "autotranslator_intro_video": "Select Intro Video (optional):",
    "autotranslator_outro_video": "Select Outro Video (optional):",
    "autotranslator_concat_success": "Videos concatenated successfully!",
    "autotranslator_processing": "Processing video...",
    "autotranslator_start_timecode": "Start Timecode (mm : ss) (option):",
    "autotranslator_end_timecode": "End Timecode (mm : ss) (option):",
    "autotranslator_translate_prompt": "Traduis depuis l'anglais vers le français:\n\n",
    "autotranslator_no_comment": "Ne commente pas, n'ajoute rien à part ce qui est demandé.",
    "autotranslator_direct_upload" : "Direct uplaod",
    "autotranslator_uploading" : "Uploading..."
})

translations["fr"].update({
    "autotranslator_tab": "Traducteur Auto",
    "autotranslator_header": "Traducteur Automatique de Vidéo YouTube",
    "autotranslator_url_input": "Entrez l'URL de la vidéo YouTube :",
    "autotranslator_enhance_checkbox": "Améliorer la vidéo (zoom et rotation)",
    "autotranslator_zoom_factor": "Facteur de zoom max (1.0 à 1.1) :",
    "autotranslator_rotation_angle": "Angle de rotation max (degrés) :",
    "autotranslator_process_button": "Traiter la vidéo",
    "autotranslator_download_success": "Vidéo téléchargée avec succès : ",
    "autotranslator_translation_success": "Vidéo traduite avec succès : ",
    "autotranslator_enhancement_success": "Vidéo améliorée avec succès !",
    "autotranslator_upload_success": "Vidéo publiée avec succès ! URL de la vidéo : https://www.youtube.com/watch?v={video_id}, édition : https://studio.youtube.com/video/{video_id}/edit",
    "autotranslator_error": "Une erreur s'est produite : ",
    "autotranslator_translation_sonitranslate": "Répertoire de SoniTranslate :",
    "autotranslator_intro_video": "Sélectionner une vidéo d'introduction (optionnel) :",
    "autotranslator_outro_video": "Sélectionner une vidéo de conclusion (optionnel) :",
    "autotranslator_concat_success": "Vidéos concaténées avec succès !",
    "autotranslator_processing": "Traitement en cours...",
    "autotranslator_start_timecode": "Timecode de début (mm : ss) (option) :",
    "autotranslator_end_timecode": "Timecode de fin (mm : ss) (option):",
    "autotranslator_translate_prompt": "Translate from English to French:\n\n",
    "autotranslator_no_comment": "Do not comment, do not add anything beside what you are instructed to do.",
    "autotranslator_direct_upload" : "Upload direct",
    "autotranslator_uploading": "Téléversement..."
})

class AutotranslatorPlugin(Plugin):
    def get_tabs(self):
        return [{"name": t("autotranslator_tab"), "plugin": "autotranslator"}]

    def get_config_fields(self):
        return {
            "translation_sonitranslate": {
                "type": "text",
                "label": t("autotranslator_translation_sonitranslate"),
                "default": ""
            }
        }

    def timecode_to_seconds(self, timecode):
        parts = timecode.split(':')
        return int(parts[0]) * 60 + int(parts[1])

    def download_video(self, url, config, start_time=None, end_time=None):
        work_directory = config['common']['work_directory']
        #return download_video_pytube(url, work_directory, start_time, end_time)
        return download_video_dlp(url, work_directory, start_time, end_time)

    def translate_video(self, input_file, config):
        work_directory = config['common']['work_directory']
        output_file = os.path.join(work_directory,'translated_video.mp4')
        translation_sonitranslate = config['autotranslator']['translation_sonitranslate']
        command = f"./autotranslate.sh {input_file} {output_file} {translation_sonitranslate}"
        subprocess.run(command, shell=True, check=True)
        return output_file

    def enhance_video_complex(self, input_file, max_zoom, max_rotation):
        output_file = 'enhanced_video.mp4'

        # Créer un filtre complexe pour le zoom et la rotation progressifs
        filter_complex = (
            f"[0:v]scale=iw*1.1:ih*1.1,zoompan=z='min(max(zoom,pzoom)+0.0015,{max_zoom})':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720:fps=30,"
            f"rotate=a='if(ld(1),ld(1),0)+0.01*sin(2*PI*t/60)*{max_rotation}':c=none[v]"
        )

        # Appliquer le filtre complexe
        stream = ffmpeg.input(input_file)
        video = stream.video.filter('fps', fps=30, round='up').filter('scale', 1280, 720)
        audio = stream.audio

        out = ffmpeg.output(
            video.filter(filter_complex),
            audio,
            output_file,
            vcodec='libx264',
            acodec='aac'
        )

        ffmpeg.run(out)
        return output_file

    def enhance_video(self, input_file, zoom_factor, rotation_angle, config):
        work_directory = config['common']['work_directory']
        output_file = os.path.join(work_directory, 'enhanced_video.mp4')

        # Appliquer le zoom et la rotation fixes
        stream = ffmpeg.input(input_file)

        # Vérifier si la vidéo principale a des sous-titres
        probe = ffmpeg.probe(input_file)
        subtitle_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'subtitle'), None)

        video = (
            stream.video
            .filter('scale', f'{zoom_factor}*iw', f'{zoom_factor}*ih')
            .filter('crop', 'iw', 'ih')
            .filter('rotate', math.radians(rotation_angle))
        )

        if subtitle_stream:
            #subtitle_filter = f"subtitles={input_file}:force_style='FontName=Arial,FontSize=24,Bold=1'"
            #video = video.filter('subtitles', subtitle_filter)
            video = video.filter('subtitles', input_file)

        audio = stream.audio

        out = ffmpeg.output(video, audio, output_file,
            acodec='aac', scodec='mov_text',
            #vcodec='libx264',
            vcodec='h264_nvenc',
            preset='fast'
            )
        ffmpeg.run(out, overwrite_output=True)
        return output_file

    def translate_text(self, text, ragllm_plugin, config):
        prompt = t("autotranslator_translate_prompt") + text
        translated_text = ragllm_plugin.process_with_llm(
            prompt,
            t("autotranslator_no_comment"),
            ""
        )
        return translated_text

    def concatenate_videos_without_subtitles(self, intro_video, main_video, outro_video, config):
        work_directory = config['common']['work_directory']
        output_file = os.path.join(work_directory, 'final_video.mp4')

        # Créer une liste de fichiers d'entrée
        input_files = [f for f in [intro_video, main_video, outro_video] if f]

        if len(input_files) == 1:
            # S'il n'y a qu'une seule vidéo, pas besoin de concaténer
            return input_files[0]

        # Obtenir les propriétés de la vidéo principale
        probe = ffmpeg.probe(main_video)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        # Créer les objets FilterableStream pour chaque fichier d'entrée
        streams = []
        for file in input_files:
            input_stream = ffmpeg.input(file)
            video = input_stream.video.filter('scale', width, height).filter('fps', fps=30).filter('format', 'yuv420p')
            audio = input_stream.audio.filter('aresample', 44100)
            streams.extend([video, audio])

        # Concaténer les vidéos
        joined = ffmpeg.concat(*streams, v=1, a=1).node
        output = ffmpeg.output(joined[0], joined[1], output_file, scodec='copy')

        ffmpeg.run(output, overwrite_output=True)
        return output_file

    def concatenate_videos(self, intro_video, main_video, outro_video, config):
        work_directory = config['common']['work_directory']
        output_file = os.path.join(work_directory, 'final_video.mp4')

        # Créer une liste de fichiers d'entrée
        input_files = [f for f in [intro_video, main_video, outro_video] if f]

        if len(input_files) == 1:
            # S'il n'y a qu'une seule vidéo, pas besoin de concaténer
            return main_video

        # Obtenir les propriétés de la vidéo principale
        probe = ffmpeg.probe(main_video)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        # Vérifier si la vidéo principale a des sous-titres
        subtitle_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'subtitle'), None)

        # Créer les entrées pour ffmpeg concat
        input_streams = [ffmpeg.input(file) for file in input_files]

        # Prépare les filtres vidéo et audio
        video_filters = []
        audio_filters = []

        for i, input in enumerate(input_streams):
            video = input.video.filter('scale', width, height).filter('fps', fps=30).filter('format', 'yuv420p')
            audio = input.audio.filter('aresample', 44100)

            if i == 1 and subtitle_stream:  # Ajoute les sous-titres pour la vidéo principale si présents
                video = video.filter('subtitles', main_video)

            video_filters.append(video)
            audio_filters.append(audio)

        # Concaténer les flux vidéo et audio
        joined_video = ffmpeg.concat(*video_filters, v=1, a=0).node
        joined_audio = ffmpeg.concat(*audio_filters, v=0, a=1).node

        # Sortie finale
        output_args = {
            'acodec': 'aac',
            #'vcodec': 'libx264',
            'vcodec':'h264_nvenc',
            'preset':'fast'
        }
        if subtitle_stream:
            output_args['scodec'] = 'mov_text'

        output = ffmpeg.output(joined_video[0], joined_audio[0], output_file, **output_args)


        # Exécuter la commande ffmpeg
        ffmpeg.run(output, overwrite_output=True)
        return output_file

    def run(self, config):
        st.header(t("autotranslator_header"))

        if 'video_to_translate' not in st.session_state:
            st.session_state['video_to_translate'] = ''

        col1, col2, col3 = st.columns(3)
        url = col1.text_input(t("autotranslator_url_input"), value=st.session_state.video_to_translate)
        st.session_state.video_to_translate = url
        start_timecode = col2.text_input(t("autotranslator_start_timecode"), "")
        end_timecode = col3.text_input(t("autotranslator_end_timecode"), "")
        enhance_video = st.checkbox(t("autotranslator_enhance_checkbox"))
        do_translate = st.checkbox("Dubb")
        do_upload_video = st.checkbox("Upload")

        max_zoom = 1.05
        max_rotation = 5
        if enhance_video:
            max_zoom = st.slider(t("autotranslator_zoom_factor"), 1.0, 1.1, 1.05, 0.01)
            max_rotation = st.slider(t("autotranslator_rotation_angle"), 0, 10, 5, 1)


        # Récupérer la liste des fichiers vidéo
        video_files = list_all_video_files(config['common']['work_directory'])
        video_options = ["None"] + [file[0] for file in video_files]

        # Sélecteurs pour les vidéos d'intro et de conclusion
        intro_video = st.selectbox(t("autotranslator_intro_video"), video_options)
        outro_video = st.selectbox(t("autotranslator_outro_video"), video_options)

        if st.button(t("autotranslator_process_button")):
            with st.spinner(t("autotranslator_processing")):
                #try:
                    start_time = self.timecode_to_seconds(start_timecode) if start_timecode else None
                    end_time = self.timecode_to_seconds(end_timecode) if end_timecode else None

                    # Download video
                    input_file, video_info = self.download_video(url, config, start_time, end_time)
                    st.success(t("autotranslator_download_success")+input_file)

                    # Translate video
                    if do_translate:
                        translated_file = self.translate_video(input_file, config)
                    else:
                        work_directory = config['common']['work_directory']
                        translated_file = os.path.join(work_directory,input_file)
                    st.success(t("autotranslator_translation_success") + translated_file)

                    # Enhance video if option is selected
                    if enhance_video:
                        enhanced_file = self.enhance_video(translated_file, max_zoom, max_rotation, config)
                        st.success(t("autotranslator_enhancement_success"))
                    else:
                        enhanced_file = translated_file

                    # Concatenate videos
                    intro_path = next((file[1] for file in video_files if file[0] == intro_video), None)
                    outro_path = next((file[1] for file in video_files if file[0] == outro_video), None)
                    final_file = self.concatenate_videos(intro_path, enhanced_file, outro_path, config)
                    if final_file != enhanced_file:
                        st.success(t("autotranslator_concat_success"))

                    # Upload video
                    if do_upload_video :
                        # Get and translate video info
                        title = video_info['title']
                        description = video_info['description']
                        tags = video_info.get('tags', [])
                        category = get_category_id(video_info.get('categories')[0])  # Use original category or default to "22" (People & Blogs)
                        print(category)

                        ragllm_plugin = self.plugin_manager.get_plugin('ragllm')
                        translated_title = self.translate_text(title, ragllm_plugin, config)
                        translated_description = self.translate_text(description, ragllm_plugin, config)+f"\nSource : {url}"
                        translated_tags = [self.translate_text(tag, ragllm_plugin, config) for tag in tags]

                        st.write(f"{translated_title}\n{translated_description}\n{translated_tags}")
                        st.info(t("autotranslator_uploading"))
                        video_id = upload_video(
                            final_file,
                            translated_title,
                            translated_description,
                            category,
                            translated_tags,
                            "unlisted"
                        )
                    col1, _ = st.columns([1, 2])
                    col1.video(final_file)
                    st.success(t("autotranslator_upload_success").format(video_id=video_id))

                #except Exception as e:
                #    st.error(t("autotranslator_error") + str(e))
