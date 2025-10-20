from app import Widget
import streamlit as st
from lib.youtube_db import get_posted_responses
from datetime import datetime

class ResponseDBDisplayWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def display(self):
        """
        Displays existing responses from the database in a dataframe.
        """
        st.header("Réponses Existantes")
        st.markdown(
            f"[Lien vers mes commentaires](https://myactivity.google.com/page?hl=fr&utm_medium=web&utm_source=youtube&page=youtube_comments)",
        )
        responses = get_posted_responses()
        if responses:
            df_data = []
            for r in responses:
                # Try parsing with UTC offset, fallback to Z format
                try:
                    posted_at = datetime.strptime(r['posted_at'], "%Y-%m-%dT%H:%M:%S.%f%z")
                except ValueError:
                    posted_at = datetime.strptime(r['posted_at'], "%Y-%m-%dT%H:%M:%S.%fZ")
                df_data.append({
                    "Mot-clé": r['keyword'],
                    "Date": posted_at.strftime("%Y-%m-%d"),
                    "Chaîne": f"https://www.youtube.com/channel/{r['channel_id']}",
                    "Vidéo": f"https://www.youtube.com/watch?v={r['video_id']}&lc={r['comment_id']}",
                    "Réponse": r['response_text'],
                    "Statut": r['moderation_status']
                })
            st.dataframe(
                df_data,
                column_config={
                    "Chaîne": st.column_config.LinkColumn(
                        label="Chaîne",
                        width="small",
                        display_text="Chaîne"
                    ),
                    "Vidéo": st.column_config.LinkColumn(
                        label="Vidéo",
                        width="small",
                        display_text="Vidéo"
                    ),
                    "Date": st.column_config.TextColumn(width="medium"),
                    "Mot-clé": st.column_config.TextColumn(width="medium"),
                    "Réponse": st.column_config.TextColumn(width="large"),
                    "Statut": st.column_config.TextColumn(width="medium")
                },
                width='stretch',
                key=f"{self.prefix}_responses_dataframe"
            )
        else:
            st.info("Aucune réponse postée trouvée dans la base.", key=f"{self.prefix}_no_responses_info")
