from global_vars import translations, t
from app import Plugin
import streamlit as st
import requests
import json
import os
import time
from plugins.ragllm import RagllmPlugin

translations["en"].update({
    "meme_generator_tab_manual": "Manual Meme Generator",
    "meme_generator_tab_auto": "Automatic Meme Generator",
    "meme_generator_header": "Create Your Meme",
    "meme_search_label": "Search memes by keyword",
    "meme_size_label": "Thumbnail Size",
    "meme_template_id_label": "Selected Template ID",
    "meme_text0_label": "Top Text",
    "meme_text1_label": "Bottom Text",
    "meme_font_label": "Font",
    "meme_font_size_label": "Font Size (px)",
    "meme_generate_button": "Generate Meme",
    "meme_generating": "Generating your meme...",
    "meme_success": "Meme generated successfully! Saved to: {path}",
    "meme_error": "Error generating meme: {error}",
    "username_config": "Imgflip Username",
    "password_config": "Imgflip Password",
    "working_dir_config": "Working Directory",
    "auto_subject_label": "Meme Subject",
    "auto_prompt_label": "LLM Sub-Prompt",
    "auto_selected_meme": "Selected Meme",
    "auto_result_label": "LLM Reasoning",
})

translations["fr"].update({
    "meme_generator_tab_manual": "Générateur de Mèmes Manuel",
    "meme_generator_tab_auto": "Générateur de Mèmes Automatique",
    "meme_generator_header": "Créez Votre Mème",
    "meme_search_label": "Rechercher des mèmes par mot-clé",
    "meme_size_label": "Taille des vignettes",
    "meme_template_id_label": "ID du modèle sélectionné",
    "meme_text0_label": "Texte du haut",
    "meme_text1_label": "Texte du bas",
    "meme_font_label": "Police",
    "meme_font_size_label": "Taille de police (px)",
    "meme_generate_button": "Générer le mème",
    "meme_generating": "Génération de votre mème...",
    "meme_success": "Mème généré avec succès ! Sauvegardé à : {path}",
    "meme_error": "Erreur lors de la génération du mème : {error}",
    "username_config": "Nom d'utilisateur Imgflip",
    "password_config": "Mot de passe Imgflip",
    "working_dir_config": "Répertoire de travail",
    "auto_subject_label": "Sujet du mème",
    "auto_prompt_label": "Sous-prompt LLM",
    "auto_selected_meme": "Mème sélectionné",
    "auto_result_label": "Raisonnement du LLM",
})


class MemegenPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.memes = self.fetch_memes()
        self.ragllm_plugin = self.plugin_manager.get_plugin('ragllm')

    def fetch_memes(self):
        """Fetch and cache popular memes from Imgflip API."""
        try:
            response = requests.get("https://api.imgflip.com/get_memes")
            data = response.json()
            if data["success"]:
                return data["data"]["memes"]
            return []
        except Exception:
            return []

    def get_config_fields(self):
        """Define configuration fields with memegen_ prefix."""
        return {
            "memegen_username": {"type": "text", "label": t("username_config"), "default": ""},
            "memegen_password": {"type": "password", "label": t("password_config"), "default": ""},
            "memegen_working_dir": {"type": "text", "label": t("working_dir_config"), "default": "~/Vidéos"},
            "memegen_auto_prompt": {
                "type": "textarea",
                "label": t("auto_prompt_label"),
                "default": "Based on the subject '{subject}', select an appropriate meme template from this list: {meme_list}. Return a plain JSON string (no ```json markers) with: 1) 'meme_name' (exact name of the chosen meme), 2) 'text0' (short top text), 3) 'text1' (short optionnal bottom text)."
            }
        }

    def get_tabs(self):
        """Define plugin tabs."""
        return [
            {"name": t("meme_generator_tab_manual"),
             "plugin": "memegenplugin"},
            {"name": t("meme_generator_tab_auto"), "plugin": "memegenplugin"}
        ]

    def generate_meme(self, config, template_id, text0, text1, font, font_size):
        """Generate and save a meme directly in working_dir with full path expansion."""
        username = config.get(self.name, {}).get("memegen_username", "")
        password = config.get(self.name, {}).get("memegen_password", "")
        working_dir = os.path.expanduser(config.get(
            self.name, {}).get("memegen_working_dir", "~/Vidéos"))

        if not username or not password:
            return None, "Please configure Imgflip username and password in settings."

        payload = {
            "template_id": template_id,
            "username": username,
            "password": password,
            "text0": text0,
            "text1": text1,
            "font": font,
            "max_font_size": str(font_size)
        }

        try:
            response = requests.post(
                "https://api.imgflip.com/caption_image", data=payload)
            result = response.json()

            if result["success"]:
                meme_url = result["data"]["url"]
                os.makedirs(working_dir, exist_ok=True)
                meme_path = os.path.join(
                    working_dir, f"meme_{template_id}_{int(time.time())}.jpg")
                with open(meme_path, "wb") as f:
                    f.write(requests.get(meme_url).content)
                return meme_path, None
            return None, result["error_message"]
        except Exception as e:
            return None, str(e)

    def run(self, config):
        """Main plugin logic with two tabs."""
        tabs = st.tabs([t("meme_generator_tab_manual"),
                       t("meme_generator_tab_auto")])

        # Tab 1: Manual Meme Generation
        with tabs[0]:
            st.header(t("meme_generator_header"))

            search_term = st.text_input(t("meme_search_label"), "")
            thumbnail_size = st.slider(t("meme_size_label"), 50, 200, 100)
            num_cols = max(1, min(10, 1000 // thumbnail_size))

            filtered_memes = [meme for meme in self.memes
                              if search_term.lower() in meme["name"].lower()] if search_term else self.memes

            if filtered_memes:
                cols = st.columns(num_cols)
                template_id = st.text_input(
                    t("meme_template_id_label"),
                    value=st.session_state.get("selected_template", ""),
                    key="template_id_manual"
                )

                for idx, meme in enumerate(filtered_memes[:100]):
                    col = cols[idx % num_cols]
                    with col:
                        st.image(meme["url"], width=thumbnail_size)
                        if st.button(meme["name"], key=f"select_{meme['id']}"):
                            st.session_state.selected_template = meme["id"]
                            st.rerun()
            else:
                st.write("No memes found.")

            text0 = st.text_input(t("meme_text0_label"),
                                  "", key="text0_manual")
            text1 = st.text_input(t("meme_text1_label"),
                                  "", key="text1_manual")
            font = st.selectbox(t("meme_font_label"), [
                                "impact", "arial"], index=0, key="font_manual")
            font_size = st.number_input(
                t("meme_font_size_label"), min_value=10, max_value=100, value=50, key="font_size_manual")

            if st.button(t("meme_generate_button"), key="generate_manual"):
                with st.spinner(t("meme_generating")):
                    meme_path, error = self.generate_meme(
                        config, template_id, text0, text1, font, font_size)
                    if meme_path:
                        st.image(meme_path, caption="Generated Meme")
                        st.success(t("meme_success").format(path=meme_path))
                    elif error:
                        st.error(t("meme_error").format(error=error))

        # Tab 2: Automatic Meme Generation
        with tabs[1]:
            st.header(t("meme_generator_header"))

            subject = st.text_input(
                t("auto_subject_label"), "", key="auto_subject")

            # Initialize session state for auto generation
            if "auto_suggestion" not in st.session_state:
                st.session_state.auto_suggestion = None
            if "auto_meme_path" not in st.session_state:
                st.session_state.auto_meme_path = None

            if st.button("Suggest Meme", key="suggest_auto"):
                with st.spinner(t("meme_generating")):
                    if not self.ragllm_plugin:
                        st.error("RAG LLM plugin not available.")
                        return

                    meme_list = json.dumps(
                        [{m["id"]: m["name"]} for m in self.memes])
                    auto_prompt = config.get(self.name, {}).get(
                        "memegen_auto_prompt", "")
                    prompt = auto_prompt.format(
                        subject=subject, meme_list=meme_list)
                    llm_sys_prompt = config['ragllm']['llm_sys_prompt']
                    llm_response = self.ragllm_plugin.process_with_llm(
                        prompt, llm_sys_prompt, subject)

                    st.subheader(t("auto_result_label"))
                    st.write(llm_response)

                    # Trim ```json markers if present
                    cleaned_response = llm_response.strip()
                    if cleaned_response.startswith("```json"):
                        cleaned_response = cleaned_response[7:].strip()
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3].strip()

                    try:
                        suggestion = json.loads(cleaned_response)
                        st.session_state.auto_suggestion = suggestion
                        meme_name = suggestion.get(
                            "meme_name", self.memes[0]["name"])
                        text0 = suggestion.get("text0", "AUTO GENERATED")
                        text1 = suggestion.get("text1", "")
                        template_id = next(
                            (m["id"] for m in self.memes if m["name"] == meme_name), self.memes[0]["id"])

                        # Generate first shot
                        font = "impact"
                        font_size = 50
                        meme_path, error = self.generate_meme(
                            config, template_id, text0, text1, font, font_size)
                        if meme_path:
                            st.session_state.auto_meme_path = meme_path
                        elif error:
                            st.error(t("meme_error").format(error=error))

                    except json.JSONDecodeError as e:
                        st.error(
                            f"Failed to parse LLM response as JSON: {str(e)}. Raw response: {llm_response}")
                        return
                    except Exception as e:
                        st.error(f"Error processing suggestion: {str(e)}")
                        return

            # Display suggestion and editable fields if available
            if st.session_state.auto_suggestion:
                suggestion = st.session_state.auto_suggestion
                meme_name = suggestion.get("meme_name", self.memes[0]["name"])
                text0 = suggestion.get("text0", "AUTO GENERATED")
                text1 = suggestion.get("text1", "")
                template_id = next(
                    (m["id"] for m in self.memes if m["name"] == meme_name), self.memes[0]["id"])

                selected_meme = st.text_input(
                    t("auto_selected_meme"), meme_name, key="auto_meme")
                edited_text0 = st.text_input(
                    t("meme_text0_label"), text0, key="auto_text0")
                edited_text1 = st.text_input(
                    t("meme_text1_label"), text1, key="auto_text1")

                if st.session_state.auto_meme_path:
                    st.image(st.session_state.auto_meme_path,
                             caption="Generated Meme")
                    st.success(t("meme_success").format(
                        path=st.session_state.auto_meme_path))

                if st.button(t("meme_generate_button"), key="generate_auto"):
                    with st.spinner(t("meme_generating")):
                        meme_path, error = self.generate_meme(
                            config, template_id, edited_text0, edited_text1, "impact", 50)
                        if meme_path:
                            st.session_state.auto_meme_path = meme_path
                            st.image(meme_path, caption="Generated Meme")
                            st.success(
                                t("meme_success").format(path=meme_path))
                        elif error:
                            st.error(t("meme_error").format(error=error))


if __name__ == "__main__":
    st.write("Memegen Plugin standalone test")
