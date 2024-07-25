from global_vars import translations, t
from app import Plugin
import streamlit as st
import yaml
from litellm import completion, model_list
from typing import List, Dict

DEFAULT_MODEL = "ollama-qwen2"

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "llm_prompt": "LLM Prompt",
    "llm_sys_prompt": "LLM System Prompt",
    "llm_model": "LLM Model",
    "llm_loaded": "LLM Plugin loaded",
    "llm_error_fetching_models": "Error fetching models: ",
    "llm_error_calling_llm": "Error calling LLM: ",
    "llm_default_prompt": "Summarize the key points of the transcript in bullet points, without commenting",
    "llm_default_sys_prompt": "You are a YouTube assistant writing viral content. You faithfully execute instructions. You respond in French unless explicitly asked to translate into English."
})
translations["fr"].update({
    "llm_prompt": "Prompt pour le LLM",
    "llm_sys_prompt": "Prompt système pour le LLM",
    "llm_model": "Modèle LLM",
    "llm_loaded": "Plugin LLM chargé",
    "llm_error_fetching_models": "Erreur lors de la récupération des modèles : ",
    "llm_error_calling_llm": "Erreur lors de l'appel au LLM : ",
    "llm_default_prompt": "Résume les grandes lignes du transcript, sous forme de liste à puce, sans commenter",
    "llm_default_sys_prompt": "Tu es un assistant YouTuber qui écrit du contenu viral. Tu exécutes les instructions fidèlement. Tu réponds en français sauf si on te demande de traduire explicitement en anglais."
})

class LlmPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.config = self.load_llm_config()

    def load_llm_config(self) -> Dict:
        with open('.llm-config.yml', 'r') as file:
            return yaml.safe_load(file)

    def get_config_fields(self):
        global lang
        return {
            "llm_prompt": {
                "type": "textarea",
                "label": t("llm_prompt"),
                "default": t("llm_default_prompt")
            },
            "llm_sys_prompt": {
                "type": "textarea",
                "label": t("llm_sys_prompt"),
                "default": t("llm_default_sys_prompt")
            },
            "llm_model": {
                "type": "select",
                "label": t("llm_model"),
                "options": [("none", DEFAULT_MODEL)],
                "default": DEFAULT_MODEL
            }
        }

    def get_config_ui(self, config):
        updated_config = {}
        for field, params in self.get_config_fields().items():
            if params['type'] == 'textarea':
                updated_config[field] = st.text_area(
                    params['label'],
                    value=config.get(field, params['default'])
                )
            elif params['label'] == t('llm_model'):
                available_models = self.get_available_models()
                updated_config[field] = st.selectbox(
                    params['label'],
                    options=available_models,
                    index=available_models.index(config.get('llm_model', DEFAULT_MODEL))
                )
        return updated_config

    def get_available_models(self) -> List[str]:
        try:
            return [model['model_name'] for model in self.config['model_list']]
        except Exception as e:
            st.error(f"{t('llm_error_fetching_models')}{str(e)}")
            return [DEFAULT_MODEL]

    def process_with_llm(self, prompt: str, sysprompt: str, transcript: str, llm_model: str) -> str:
        try:
            model_config = next((model for model in self.config['model_list'] if model['model_name'] == llm_model), None)

            if not model_config:
                raise ValueError(f"Configuration non trouvée pour le modèle {llm_model}")

            litellm_params = model_config['litellm_params']

            response = completion(
                model=litellm_params['model'],
                messages=[
                    {"role": "system", "content": sysprompt},
                    {"role": "user", "content": f"{prompt} : \n {transcript}"}
                ],
                api_base=litellm_params['api_base'],
                **litellm_params.get('additional_params', {})
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"{t('llm_error_calling_llm')}{str(e)}"

    def run(self, config):
        st.write(t("llm_loaded"))
