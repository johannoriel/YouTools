from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import os

# Traductions spécifiques au widget
translations["en"].update({
    "social_preview": "Preview and Edit Posts",
    "social_manual_post_start": "Add Post at Start",
    "social_manual_post_end": "Add Post at End",
    "social_clear_all": "Clear All Posts",
    "social_reload_thread": "Reload thread.txt"
})

translations["fr"].update({
    "social_preview": "Prévisualiser et Éditer",
    "social_manual_post_start": "Ajouter un post au début",
    "social_manual_post_end": "Ajouter un post à la fin",
    "social_clear_all": "Effacer tous les posts",
    "social_reload_thread": "Recharger thread.txt"
})

class ThreadEditorWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def display(self, config):
        st.subheader(t("social_preview"))
        work_dir = config['common']['work_directory']
        thread_path = os.path.join(work_dir, "thread.txt")
        edited_thread_path = os.path.join(work_dir, "thread_edited.txt")

        # Initialize session state if not present
        if f"{self.prefix}_posts" not in st.session_state:
            st.session_state[f"{self.prefix}_posts"] = []

        # Load thread.txt if exists, and update session state
        if os.path.exists(thread_path):
            with open(thread_path, 'r') as f:
                posts = [post.strip() for post in f.read().split('---') if post.strip()]
            # Only load if session state is empty or reload is explicitly requested
            if not st.session_state[f"{self.prefix}_posts"]:
                st.session_state[f"{self.prefix}_posts"] = posts

        # Display and edit posts
        for i, post in enumerate(st.session_state[f"{self.prefix}_posts"]):
            edited_post = st.text_area(f"Post {i+1}", post, key=f"{self.prefix}_post_{i}", height=100)
            st.session_state[f"{self.prefix}_posts"][i] = edited_post

        # Buttons for adding posts
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("social_manual_post_start"), key=f"{self.prefix}_add_start"):
                st.session_state[f"{self.prefix}_posts"].insert(0, "")
                st.rerun()  # Force rerender to update UI
        with col2:
            if st.button(t("social_manual_post_end"), key=f"{self.prefix}_add_end"):
                st.session_state[f"{self.prefix}_posts"].append("")
                st.rerun()  # Force rerender to update UI

        # Buttons for clearing and reloading
        col3, col4 = st.columns(2)
        with col3:
            if st.button(t("social_clear_all"), key=f"{self.prefix}_clear"):
                st.session_state[f"{self.prefix}_posts"] = ['']
                st.rerun()  # Force rerender to update UI
        with col4:
            if st.button(t("social_reload_thread"), key=f"{self.prefix}_reload"):
                if os.path.exists(thread_path):
                    with open(thread_path, 'r') as f:
                        st.session_state[f"{self.prefix}_posts"] = [
                            post.strip() for post in f.read().split('---') if post.strip()
                        ]
                    st.rerun()  # Force rerender to update UI
                else:
                    st.warning("thread.txt not found")

        # Save edited posts
        if st.button("Save Edited Thread", key=f"{self.prefix}_save"):
            with open(edited_thread_path, 'w') as f:
                f.write('\n---\n'.join(st.session_state[f"{self.prefix}_posts"]))
            st.success("Thread saved to thread_edited.txt")
