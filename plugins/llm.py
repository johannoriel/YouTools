from app import Plugin
import streamlit as st
import requests

DEFAULT_MODEL = "ollama-qwen2"

class LlmPlugin(Plugin):
        
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
            "llm_url": {
                "type": "text",
                "label": "URL du LLM",
                "default": "http://localhost:4000"
            },
            "llm_model": {
                "type": "select",
                "label": "Modèle LLM",
                "options": [("none", DEFAULT_MODEL)],
                "default": "ollama-dolphin"
            },
            "llm_key": {
                "type": "text",
                "label": "Clé d'API du LLM",
                "default": ""
            }
        }

    def get_config_ui(self, config):
        updated_config = {}
        #updated_config['separator_llm'] = st.header('LLM')
        for field, params in self.get_config_fields().items():
            if params['type'] == 'textarea':
                updated_config[field] = st.text_area(
                    params['label'],
                    value=config.get(field, params['default'])
                )
            elif params['label'] == 'Modèle LLM':
                if 'llm_url' in config and 'llm_key' in config :
                    available_models = self.get_available_models(config['llm_url'], config['llm_key'])
                    updated_config[field] = st.selectbox(
                        params['label'],
                        options=available_models,
                        index=available_models.index(config.get('llm_model', DEFAULT_MODEL))
                    )
                else: 
                    updated_config[field] = st.text_input(params['label'],value= DEFAULT_MODEL)
            else:
                updated_config[field] = st.text_input(
                    params['label'],
                    value=config.get(field, params['default']),
                    type="password" if field == "llm_key" else "default"
                )
        return updated_config

    def get_available_models(self, llm_url, llm_key):
        try:
            headers = {'Authorization': f'Bearer {llm_key}'}
            response = requests.get(f"{llm_url}/models", headers=headers)
            response.raise_for_status()
            models = response.json()['data']
            return [model['id'] for model in models]
        except requests.RequestException as e:
            st.error(f"Erreur lors de la récupération des modèles : {str(e)}")
            return []

    def process_with_llm(self, prompt, sysprompt, transcript, llm_url, llm_model, llm_key):
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

    def run(self, config):
        st.write("Plugin LLM chargé")
