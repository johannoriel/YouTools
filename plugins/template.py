from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
# Exemple d'utilisation d'une fonction de common.py
from plugins.common import remove_quotes
import os

# Ajout des traductions spécifiques à ce plugin (exemple avec données dummy)
translations["en"].update({
    "template_tab": "Template Plugin",
    "template_header": "Template Plugin Header",
    "template_input_label": "Enter your input here",
    "template_process_button": "Process Input",
    "template_processing": "Processing your request...",
    "template_success": "Processing completed successfully! Result: {result}",
    "template_error": "An error occurred: {error}",
    "template_llm_prompt": "Generate a dummy response based on this input",
    "template_config_field": "Sample Config Field",
    "template_config_default": "default_value",
})

translations["fr"].update({
    "template_tab": "Plugin Modèle",
    "template_header": "En-tête du Plugin Modèle",
    "template_input_label": "Entrez votre entrée ici",
    "template_process_button": "Traiter l'entrée",
    "template_processing": "Traitement de votre demande...",
    "template_success": "Traitement terminé avec succès ! Résultat : {result}",
    "template_error": "Une erreur s'est produite : {error}",
    "template_llm_prompt": "Générer une réponse factice basée sur cette entrée",
    "template_config_field": "Champ de configuration exemple",
    "template_config_default": "valeur_par_défaut",
})

# Le nom du plugin est toujours XxxxxxPlugin : première lettre majuscule, et le reste en minuscule, plus Plugin à la fin.


class TemplatePlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)

    def get_config_fields(self):
        """Définit les champs de configuration du plugin (exemple dummy). Les champs doivent être préfixés par le nom du plugin"""
        return {
            "template_sample_field": {
                "type": "text",
                "label": t("template_config_field"),
                "default": t("template_config_default")
            },
            "teplate_sample_textarea": {
                "type": "textarea",
                "label": "Sample Textarea Field",
                "default": "Default textarea content"
            },
            "template_sample_select": {
                "type": "select",
                "label": "Sample Select Field",
                "options": [("option1", "Option 1"), ("option2", "Option 2")],
                "default": "option1"
            }
        }

    def get_tabs(self):
        """Définit les onglets du plugin dans l'interface."""
        return [{"name": t("template_tab"), "plugin": "templateplugin"}]

    def run(self, config):
        """Logique principale du plugin."""
        st.header(t("template_header"))

        # Exemple d'entrée utilisateur
        user_input = st.text_input(
            t("template_input_label"), value="Default input")

        # Bouton pour lancer le traitement
        if st.button(t("template_process_button")):
            with st.spinner(t("template_processing")):
                try:
                    # Étape 1 : Utilisation d'une fonction de common.py (exemple)
                    cleaned_input = remove_quotes(user_input)
                    st.write(f"Cleaned input: {cleaned_input}")

                    # Prompt dummy pour le LLM
                    llm_prompt = t("template_llm_prompt")
                    # Utilisation de la configuration système par défaut de llm
                    llm_sys_prompt = config['llm']['llm_sys_prompt']
                    # Appel au LLM avec l'entrée utilisateur comme contexte
                    llm_response = self.process_with_llm(
                        llm_prompt,
                        llm_sys_prompt,
                        cleaned_input
                    )
                    st.write(f"LLM Response: {llm_response}")

                    # Exemple de résultat final
                    result = f"Processed '{cleaned_input}' with LLM: {llm_response}"
                    st.success(t("template_success").format(result=result))

                except Exception as e:
                    st.error(t("template_error").format(error=str(e)))

        # Exemple d'accès à la configuration
        sample_config_value = config.get(
            self.name, {}).get("sample_field", "N/A")
        st.write(f"Sample config value: {sample_config_value}")


# Fonction utilitaire (exemple pour montrer comment ajouter des helpers)
def dummy_helper_function(input_string: str) -> str:
    """Exemple de fonction utilitaire."""
    return f"Dummy helper processed: {input_string}"


if __name__ == "__main__":
    # Pour tester le plugin indépendamment (optionnel)
    st.write("Template Plugin standalone test")
