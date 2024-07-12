from app import Plugin
import streamlit as st
import yaml
from litellm import completion, model_list
from typing import List, Dict

DEFAULT_MODEL = "ollama-qwen2"

class LlmPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.config = self.load_llm_config()

    def load_llm_config(self) -> Dict:
        with open('.llm-config.yml', 'r') as file:
            return yaml.safe_load(file)

    def get_config_fields(self):
        return {
            "llm_prompt": {
                "type": "textarea",
                "label": "Prompt pour le LLM",
                "default": "Résume les grandes lignes du transcript, sous forme de liste à puce, sans commenter"
            },
            "llm_sys_prompt": {
                "type": "textarea",
                "label": "Prompt système pour le LLM",
                "default": "Tu es un assistant YouTuber qui écrit du contenu viral. Tu exécute les instructions fidèlement. Tu réponds en français sauf si on te demande de traduire explicitement en anglais."
            },
            "llm_model": {
                "type": "select",
                "label": "Modèle LLM",
                "options": [("none", DEFAULT_MODEL)],
                "default": "ollama-qwen2"
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
            elif params['label'] == 'Modèle LLM':
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
            st.error(f"Erreur lors de la récupération des modèles : {str(e)}")
            return [DEFAULT_MODEL]

    def process_with_llm(self, prompt: str, sysprompt: str, transcript: str, llm_model: str) -> str:
        try:
            # Chercher les paramètres correspondant au modèle sélectionné
            model_config = next((model for model in self.config['model_list'] if model['model_name'] == llm_model), None)
            
            if not model_config:
                raise ValueError(f"Configuration non trouvée pour le modèle {llm_model}")
            
            # Utiliser les paramètres spécifiés dans litellm_params
            litellm_params = model_config['litellm_params']
            
            response = completion(
                model=litellm_params['model'],
                messages=[
                    {"role": "system", "content": sysprompt},
                    {"role": "user", "content": f"{prompt} : \n {transcript}"}
                ],
                api_base=litellm_params['api_base'],
                **litellm_params.get('additional_params', {})  # Pour tout autre paramètre supplémentaire
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Erreur lors de l'appel au LLM : {str(e)}"

    def run(self, config):
        st.write("Plugin LLM chargé")
