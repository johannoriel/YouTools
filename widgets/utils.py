from lib.global_vars import translations, t
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from lib.youtube_db import cache_campaign_response

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
