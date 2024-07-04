import os
import json
import streamlit as st
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptAvailable
import subprocess
from datetime import datetime

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API depuis les variables d'environnement
API_KEY = os.getenv("YOUTUBE_API_KEY")
LLM_KEY = os.getenv("LLM_API_KEY")
DEFAULT_MODEL = "ollama-dolphin"

# Nom du fichier de configuration
CONFIG_FILE = "config.json"


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Valeurs par défaut
    defaults = {
        'channel_id': '',
        'language': 'fr',
        'llm_prompt': "Résume les grandes lignes du transcript, sous forme de liste à puce, sans commenter",
        'llm_sys_prompt': "Tu es un assistant YouTuber qui écrit du contenu viral. Tu exécute les instructions fidèlement. Tu réponds en français sauf si on te demande de traduire explicitement en anglais.",
        'llm_url': "http://localhost:4000",
        'llm_model': DEFAULT_MODEL,
        'llm_key': LLM_KEY,
        'work_directory': os.path.expanduser("~/Vidéos"),
        'silence_threshold': -35,
        'silence_duration': 0.5
    }
    
    # Mettre à jour avec les valeurs par défaut si elles n'existent pas
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

####################################################
# Youtube Transcript summurize

def get_channel_videos(channel_id):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    try:
        channel_response = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        ).execute()
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        playlist_response = youtube.playlistItems().list(
            part='snippet',
            playlistId=uploads_playlist_id,
            maxResults=5
        ).execute()
        
        videos = []
        for item in playlist_response['items']:
            video = {
                'title': item['snippet']['title'],
                'video_id': item['snippet']['resourceId']['videoId'],
                'thumbnail': item['snippet']['thumbnails']['default']['url']
            }
            videos.append(video)
        
        return videos
    except HttpError as e:
        st.error(f"Une erreur s'est produite : {e}")
        return []

def get_transcript(video_id, language):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
    except NoTranscriptAvailable:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except TranscriptsDisabled:
            return "Les transcriptions sont désactivées pour cette vidéo.", "N/A"
        except NoTranscriptAvailable:
            return "Aucune transcription n'est disponible pour cette vidéo.", "N/A"
        except Exception as e:
            return f"Une erreur s'est produite lors de la récupération du transcript : {str(e)}", "N/A"
    
    full_transcript = " ".join([entry['text'] for entry in transcript])
    
    return full_transcript, language
    
def get_available_models(llm_url, llm_key):
    try:
        headers = {'Authorization': f'Bearer {llm_key}'}
        response = requests.get(f"{llm_url}/models", headers=headers)
        response.raise_for_status()
        models = response.json()['data']
        return [model['id'] for model in models]
    except requests.RequestException as e:
        st.error(f"Erreur lors de la récupération des modèles : {str(e)}")
        return []    

def process_with_llm(prompt, sysprompt, transcript, llm_url, llm_model, llm_key):
    try:
        headers = {'Authorization': f'Bearer {llm_key}'}
        response = requests.post(
            f"{llm_url}/chat/completions", headers=headers,
            json={
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": sysprompt},
                    {"role": "user", "content": f"{prompt} : \n {transcript}"}
                ]
            }
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        return f"Erreur lors de l'appel au LLM : {str(e)}"

###################################################
# Trim silence

def list_video_files(directory):
    video_files = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.mkv', '.mp4')) and not file.startswith('outfile'):
            full_path = os.path.join(directory, file)
            mod_time = os.path.getmtime(full_path)
            video_files.append((file, full_path, mod_time))
    
    # Trier par date de modification, la plus récente en premier
    video_files.sort(key=lambda x: x[2], reverse=True)
    return video_files

def remove_silence(input_file, threshold, duration, videos_dir):
    # Obtenir le chemin absolu du script actuel (app.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construire le chemin complet vers remsi.py
    remsi_path = os.path.join(current_dir, 'remsi.py')
    
    # Construire le chemin complet vers output.sh
    output_sh_path = os.path.join(current_dir, 'output.sh')
    input_filename = os.path.basename(input_file)
    output_filename = f"outfile_{input_filename}"
    output_file = os.path.join(videos_dir, output_filename)
    ffmpeg_command = f"ffmpeg -i '{input_file}' -hide_banner -af silencedetect=n={threshold}dB:d={duration} -f null - 2>&1 | python '{remsi_path}' > '{output_sh_path}'"
    
    try:
        # Exécuter la commande ffmpeg
        subprocess.run(ffmpeg_command, shell=True, check=True, cwd=current_dir)
        
        # Vérifier si output.sh a été créé et n'est pas vide
        if not os.path.exists(output_sh_path) or os.path.getsize(output_sh_path) == 0:
            raise subprocess.CalledProcessError(1, ffmpeg_command, "output.sh n'a pas été créé ou est vide")
        
        # Rendre le script exécutable
        subprocess.run(f"chmod +x '{output_sh_path}'", shell=True, check=True)
        
        # Exécuter output.sh
        subprocess.run(output_sh_path, shell=True, check=True)
        
        # Supprimer output.sh en cas de succès
        os.remove(output_sh_path)
        
        return output_file
    except subprocess.CalledProcessError as e:
        return f"Erreur lors de l'exécution de la commande : {str(e)}"
    except Exception as e:
        return f"Une erreur inattendue s'est produite : {str(e)}"

###################################################

def main():
    st.set_page_config(page_title="YouTube Dashboard", layout="wide")
    
    st.title("YoutTools : YouTube Channel helpers")
    
    if not API_KEY:
        st.error("La clé API YouTube n'est pas définie. Veuillez vérifier votre fichier .env")
        return
    
    # Charger la configuration
    config = load_config()
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["Configuration", "Résumé des vidéos", "Retrait des silences"])
    
    # Onglet Configuration
    with tab1:
        # Onglet Configuration
        with st.expander("Configuration Transcript"):
            channel_id = st.text_input("ID de la chaîne YouTube", value=config.get('channel_id', ''))
            language = st.selectbox("Langue préférée pour les transcriptions", 
                                    options=['fr', 'en'], 
                                    format_func=lambda x: 'Français' if x == 'fr' else 'Anglais',
                                    index=0 if config.get('language', 'fr') == 'fr' else 1)
            llm_prompt = st.text_area("Prompt pour le LLM", value=config.get('llm_prompt', ''))
            llm_sys_prompt = st.text_area("Prompt système pour le LLM", value=config.get('llm_sys_prompt', ''))
            llm_url = st.text_input("URL du LLM", value=config.get('llm_url', ''))
            llm_key = st.text_input("Clé d'API du LLM", value=config.get('llm_key', ''))
            available_models = get_available_models(llm_url, llm_key)
            if available_models:
                llm_model = st.selectbox("Modèle LLM", options=available_models, index=available_models.index(config.get('llm_model', DEFAULT_MODEL)) if config.get('llm_model', DEFAULT_MODEL) in available_models else 0)
            else:
                llm_model = st.text_input("Modèle LLM", value=config.get('llm_model', DEFAULT_MODEL))
            if st.button("Sauvegarder transcript"):
                config['channel_id'] = channel_id
                config['language'] = language
                config['llm_prompt'] = llm_prompt
                config['llm_sys_prompt'] = llm_sys_prompt
                config['llm_url'] = llm_url
                config['llm_model'] = llm_model
                config['llm_key'] = llm_key
                save_config(config)
                st.success("Configuration sauvegardée !")

        with st.expander("Configuration silence"):
            work_directory = st.text_input("Répertoire de travail", value=config.get('work_directory', ''))
            silence_threshold = st.slider("Seuil de silence (dB)", min_value=-60, max_value=0, value=config.get('silence_threshold', -35))
            silence_duration = st.slider("Durée minimale du silence (secondes)", min_value=0.1, max_value=2.0, value=config.get('silence_duration', 0.5), step=0.1)
            if st.button("Sauvegarder silence"):
                config['work_directory'] = work_directory
                config['silence_threshold'] = silence_threshold
                config['silence_duration'] = silence_duration
                save_config(config)
                st.success("Configuration sauvegardée !")

    
    # Onglet Résumé des vidéos
    with tab2:
        if config.get('channel_id'):
            st.header("5 dernières vidéos")
            videos = get_channel_videos(config['channel_id'])
            
            for video in videos:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.image(video['thumbnail'])
                with col2:
                    st.subheader(video['title'])
                    st.markdown(f"[Voir la vidéo](https://www.youtube.com/watch?v={video['video_id']})")
                with col3:
                    if st.button("Transcript", key=f"transcript_{video['video_id']}"):
                        transcript, lang = get_transcript(video['video_id'], config['language'])
                        st.session_state.transcript = transcript
                        st.session_state.transcript_lang = lang
                        st.session_state.show_transcript = True
                        st.session_state.current_video_id = video['video_id']
        else:
            st.info("Veuillez configurer l'ID de la chaîne dans l'onglet Configuration.")
        
        # Affichage du transcript
        if st.session_state.get('show_transcript', False):
            st.header("Transcript")
            st.write(f"Langue de la transcription : {st.session_state.transcript_lang}")
            st.text_area("Contenu du transcript", st.session_state.transcript, height=300)
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Copier le transcript"):
                    st.code(st.session_state.transcript)
                    st.success("Transcript copié ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour le copier depuis le bloc de code ci-dessus.")
            with col2:
                st.download_button(
                    label="Télécharger le transcript",
                    data=st.session_state.transcript,
                    file_name=f"transcript_{st.session_state.transcript_lang}.txt",
                    mime="text/plain"
                )
            with col3:
                if st.button("Traiter avec LLM"):
                    llm_response = process_with_llm(config['llm_prompt'], config['llm_prompt'], st.session_state.transcript, config['llm_url'], config['llm_model'], config['llm_key'])
                    st.session_state.llm_response = llm_response
                    st.session_state.show_llm_response = True
                    
        # Affichage de la réponse du LLM
        if st.session_state.get('show_llm_response', False):
            st.header("Réponse du LLM")
            st.text_area("Contenu de la réponse", st.session_state.llm_response, height=300)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Copier la réponse du LLM"):
                    st.code(st.session_state.llm_response)
                    st.success("Réponse du LLM copiée ! Utilisez Ctrl+C (ou Cmd+C sur Mac) pour la copier depuis le bloc de code ci-dessus.")
            with col2:
                st.download_button(
                    label="Télécharger la réponse du LLM",
                    data=st.session_state.llm_response,
                    file_name=f"llm_response_{st.session_state.current_video_id}.txt",
                    mime="text/plain"
                )

    # Nouvel onglet pour le retrait des silences
    with tab3:
        st.header("Retrait des silences")
        video_files = list_video_files(config['work_directory'])
        
        for file, full_path, _ in video_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button("Retirer les silences", key=f"remove_silence_{file}"):
                    with st.spinner(f"Traitement de {file} en cours..."):
                        result = remove_silence(full_path, config['silence_threshold'], config['silence_duration'], config['work_directory'])
                    if result.startswith("Erreur"):
                        st.error(result)
                    else:
                        st.success(f"Traitement terminé. Fichier de sortie : {result}")
            
if __name__ == "__main__":
    main()
