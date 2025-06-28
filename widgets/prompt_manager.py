# widgets/prompt_manager.py

from lib.global_vars import translations, t
from app import Widget
import streamlit as st
import ast
import json

translations["en"].update({
    "prompt_management": "Prompt Management",
    "select_prompt": "Select a prompt",
    "custom_prompt": "Custom prompt (optional)",
    "apply_prompt": "Apply Prompt",
    "new_prompt_name": "New prompt name",
    "new_prompt_content": "New prompt content",
    "add_prompt": "Add Prompt",
    "save_prompt": "Save Prompt",
    "edit_prompt": "Edit Prompt",
    "delete_prompt": "Delete Prompt",
    "prompt_result": "Prompt Result",
    "copy_result": "Copy Result",
    "download_result": "Download Result",
    "result_copied": "Result copied! Use Ctrl+C (or Cmd+C on Mac) to copy it from the code block above.",
    "promt_result_display": "Prompt: {0}",  # Placeholder {0} pour le nom du prompt
})

translations["fr"].update({
    "prompt_management": "Gestion des prompts",
    "select_prompt": "Sélectionner un prompt",
    "custom_prompt": "Prompt personnalisé (optionnel)",
    "apply_prompt": "Appliquer le Prompt",
    "new_prompt_name": "Nom du nouveau prompt",
    "new_prompt_content": "Contenu du nouveau prompt",
    "add_prompt": "Ajouter un Prompt",
    "save_prompt": "Sauver le Prompt",
    "edit_prompt": "Modifier le Prompt",
    "delete_prompt": "Supprimer le Prompt",
    "prompt_result": "Résultat du Prompt",
    "copy_result": "Copier le Résultat",
    "download_result": "Télécharger le Résultat",
    "result_copied": "Résultat copié ! Utilisez Ctrl+C (ou Cmd+C on Mac) pour le copier depuis le bloc de code ci-dessus.",
    "promt_result_display": "Prompt : {0}",  # Placeholder {0} pour le nom du prompt
})

class PromptsManagerWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)
        if 'prompts' not in st.session_state:
            st.session_state.prompts = {}

    def display(self, prompt_varname):
        st.subheader(t("prompt_management"))
        config = self.plugin_manager.config

        try:
            if isinstance(config[self.name][prompt_varname], str):
                st.session_state.prompts = ast.literal_eval(config[self.name][prompt_varname])
            else:
                st.session_state.prompts = config[self.name][prompt_varname]
        except (SyntaxError, ValueError):
            st.error("Erreur lors du décodage des prompts de la configuration. Réinitialisation à un dictionnaire vide.")
            st.session_state.prompts = {}

        prompt_options = list(st.session_state.prompts.keys()) + ['Custom']
        selected_prompt = st.selectbox(t("select_prompt"), options=prompt_options, key=f"{self.prefix}_prompt_select")

        if selected_prompt == 'Custom':
            prompt_content = st.text_area(t("custom_prompt"), "", key=f"{self.prefix}_custom_prompt")
        else:
            prompt_content = st.text_area(t("edit_prompt"), st.session_state.prompts.get(selected_prompt, ""), key=f"{self.prefix}_edit_prompt")

        col1, col2, col3 = st.columns([1, 1, 1])
        new_prompt_name = col1.text_input(t("new_prompt_name"), key=f"{self.prefix}_new_prompt_name")
        if col1.button(t("add_prompt"), key=f"{self.prefix}_add_prompt"):
            if new_prompt_name:
                st.session_state.prompts[new_prompt_name] = prompt_content
                config[self.name][prompt_varname] = str(st.session_state.prompts)
                self.plugin_manager.save_config(config)
                st.success(f"Prompt '{new_prompt_name}' ajouté/mis à jour.")
                st.rerun()

        if selected_prompt != 'Custom' and col2.button(t("delete_prompt"), key=f"{self.prefix}_delete_prompt"):
            del st.session_state.prompts[selected_prompt]
            config[self.name][prompt_varname] = str(st.session_state.prompts)
            self.plugin_manager.save_config(config)
            st.success(f"Prompt '{selected_prompt}' supprimé.")
            st.rerun()

        if selected_prompt != 'Custom' and col3.button(t("save_prompt"), key=f"{self.prefix}_save_prompt"):
            st.session_state.prompts[selected_prompt] = prompt_content
            config[self.name][prompt_varname] = str(st.session_state.prompts)
            self.plugin_manager.save_config(config)
            st.success(f"Prompt '{selected_prompt}' sauvegardé.")
            st.rerun()

    def display_prompts(self, plugin_name, field_name, prompt_type="json"):
        """Affiche les prompts d'un plugin donné à partir du champ de configuration spécifié."""
        config = self.plugin_manager.config.get(plugin_name, {})
        prompts_data = config.get(field_name, "{}" if prompt_type == "json" else "")

        if prompt_type == "json":
            try:
                if isinstance(prompts_data, str):
                    prompts = ast.literal_eval(prompts_data)
                else:
                    prompts = prompts_data
            except (SyntaxError, ValueError):
                st.error(f"Erreur lors du décodage des prompts pour {plugin_name} ({field_name}).")
                return
            for prompt_name, prompt_content in prompts.items():
                st.text_area(
                    t("promt_result_display").format(prompt_name),
                    prompt_content,
                    key=f"{self.prefix}_{plugin_name}_{prompt_name}_display",
                    disabled=True
                )
        else:
            st.text_area(
                t("promt_result_display").format(field_name),
                prompts_data,
                key=f"{self.prefix}_{plugin_name}_{field_name}_display",
                disabled=True
            )
