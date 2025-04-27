from lib.global_vars import translations, t
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from lib.youtube_db import cache_campaign_response

def generate_response_for_content(
    widget, config, content_dict, prompt_template, context, keyword
):
    """
    Génère une réponse pour un contenu donné (commentaire, vidéo, etc.) en utilisant un prompt LLM.

    Args:
        widget: Instance du widget appelant (pour process_with_llm)
        config: Configuration du plugin
        content_dict: Dictionnaire contenant les informations du contenu
        prompt_template: Modèle de prompt pour le LLM
        context: Contexte supplémentaire (ex: transcription, description)
        url: URL à promouvoir
        keyword: Mot-clé associé

    Returns:
        Dictionnaire contenant la réponse générée et les métadonnées
    """
    content_with_context = (
        f"{context}\nContent: {content_dict.get('comment_text', '')}"
    )
    #st.write(content_dict)
    prompt = prompt_template.format(url=content_dict['product_url'], context=context, keywords=content_dict['keywords'], video_title=content_dict['video_title'], )
    try:
        llm_response = widget.process_with_llm(
            prompt,
            config.get('llm', {}).get('llm_sys_prompt', ''),
            content_with_context
        )
        clean_response = remove_quotes(llm_response.strip())
    except Exception as e:
        clean_response = f"Error: {str(e)}"

    return {
        'comment_id': content_dict.get('comment_id', ''),
        'response': clean_response,
        'target_video_id': content_dict.get('video_id', ''),
        'channel_id': content_dict.get('channel_id', ''),
        'keyword': keyword,
        'comment_text': content_dict.get('comment_text', ''),
        'author': content_dict.get('author', ''),
        'video_title': content_dict.get('video_title', ''),
        'channel_title': content_dict.get('channel_title', ''),
        'product_title': content_dict.get('product_title', ''),
    }

def generate_responses_for_list(
    widget, config, content_list, prompt_template, context, keyword
):
    """
    Génère des réponses pour une liste de contenus.

    Args:
        widget: Instance du widget appelant
        config: Configuration du plugin
        content_list: Liste de dictionnaires de contenus
        prompt_template: Modèle de prompt pour le LLM
        context: Contexte supplémentaire
        url: URL à promouvoir
        keyword: Mot-clé associé

    Returns:
        Liste de réponses générées
    """
    responses = []
    total_items = len(content_list)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for idx, content in enumerate(content_list):
        progress = (idx + 1) / total_items
        progress_bar.progress(progress)
        progress_text.text(f"Processing item {idx + 1} of {total_items}")

        response = generate_response_for_content(
            widget, config, content, prompt_template, context, keyword
        )
        responses.append(response)

    progress_bar.empty()
    progress_text.empty()
    return responses

def export_responses(responses, work_dir, base_filename, overwrite):
    """
    Exporte les réponses dans un fichier CSV.

    Args:
        responses: Liste de dictionnaires contenant les réponses
        work_dir: Répertoire de travail
        base_filename: Nom de base du fichier
        overwrite: Si True, écrase le fichier existant

    Returns:
        Chemin du fichier exporté
    """
    try:
        output_path = os.path.join(work_dir, base_filename)
        if not overwrite and os.path.exists(output_path):
            i = 1
            while True:
                new_filename = f"{base_filename.split('.')[0]}_{i:03d}.csv"
                new_output_path = os.path.join(work_dir, new_filename)
                if not os.path.exists(new_output_path):
                    output_path = new_output_path
                    break
                i += 1

        responses_data = [
            {
                'comment_id': resp['comment_id'],
                'response_text': resp['response'],
                'comment_text': resp['comment_text'],
                'video_id': resp['target_video_id'],
                'channel_id': resp['channel_id'],
                'author': resp['author'],
                'video_title': resp['video_title'],
                'channel_title': resp['channel_title'],
                'keyword': resp['keyword']
            }
            for resp in responses
        ]
        df = pd.DataFrame(responses_data)
        df.to_csv(output_path, index=False)
        st.success(t("export_success").format(filename=output_path))
        return output_path
    except Exception as e:
        st.error(t("export_error").format(error=str(e)))
        return None

def remove_quotes(text: str) -> str:
    """
    Supprime les guillemets entourant un texte.

    Args:
        text: Texte à nettoyer

    Returns:
        Texte sans guillemets
    """
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text
