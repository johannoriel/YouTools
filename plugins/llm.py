# llm.py
from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import requests
import json
import time
import random
from typing import List, Dict, Any, Tuple
import ast
from streamlit_lexical import streamlit_lexical
import subprocess
import time
import requests
import re

# Translations
translations["en"].update({
    "llm_llm_tab": "LLM",
    "llm_keys_tab": "API Keys",
    "llm_apis_tab": "APIs",
    "llm_models_tab": "Models",
    "llm_chat_tab": "Chat",
    "llm_keys_header": "Manage API Keys",
    "llm_apis_header": "Manage API Endpoints",
    "llm_models_header": "Manage Models",
    "llm_chat_header": "Test Chat Completion",
    "llm_key_name_label": "Key Name",
    "llm_key_value_label": "Key Value",
    "llm_api_name_label": "API Name",
    "llm_url_label": "API URL",
    "llm_api_key_label": "API Key",
    "llm_model_name_label": "Model Name",
    "llm_model_label": "Model",
    "llm_temp_label": "Temperature",
    "llm_max_tokens_label": "Max Tokens",
    "llm_delay_label": "Delay (seconds)",
    "llm_max_retries_label": "Max Retries",
    "llm_add_key": "Add Key",
    "llm_add_api": "Add API",
    "llm_add_model": "Add Model",
    "llm_get_models": "Get via ",
    "llm_select_api": "Select API",
    "llm_select_model": "Select Model",
    "llm_prompt_label": "Prompt",
    "llm_send_prompt": "Send",
    "llm_response_label": "Response",
    "llm_llm_calling_error": "Error calling LLM: ",
    "llm_no_v1_label": "No /v1 in endpoint",
    "llm_timeout_label": "Timeout (seconds)",
    "llm_sys_prompt": "System prompt for LLM",
    "llm_default_sys_prompt": "You are an faithful AI assistant that execute instructions faithfully without adding comments or explanations.",
    "llm_personas_tab": "Personas",
    "llm_personas_header": "Manage Personas",
    "llm_persona_name_label": "Persona Name",
    "llm_persona_prompt_label": "Persona System Prompt",
    "llm_add_persona": "Add Persona",
    "llm_select_persona": "Select Persona",
    "llm_ollama_restart": "Restart Ollama",
    "llm_prompt_sequence_tab": "Prompt Sequence",
})

translations["fr"].update({
    "llm_llm_tab": "LLM",
    "llm_keys_tab": "Clés API",
    "llm_apis_tab": "APIs",
    "llm_models_tab": "Modèles",
    "llm_chat_tab": "Chat",
    "llm_keys_header": "Gérer les clés API",
    "llm_apis_header": "Gérer les points de terminaison API",
    "llm_models_header": "Gérer les modèles",
    "llm_chat_header": "Tester la complétion de chat",
    "llm_key_name_label": "Nom de la clé",
    "llm_key_value_label": "Valeur de la clé",
    "llm_api_name_label": "Nom de l'API",
    "llm_url_label": "URL API",
    "llm_api_key_label": "Clé API",
    "llm_model_name_label": "Nom du modèle",
    "llm_model_label": "Modèle",
    "llm_temp_label": "Température",
    "llm_max_tokens_label": "Tokens max",
    "llm_delay_label": "Délai (secondes)",
    "llm_max_retries_label": "Rétries max",
    "llm_add_key": "Ajouter une clé",
    "llm_add_api": "Ajouter une API",
    "llm_add_model": "Ajouter un modèle",
    "llm_get_models": "Récupérer via ",
    "llm_select_api": "Sélectionner une API",
    "llm_select_model": "Sélectionner un modèle (Nouveau)",
    "llm_prompt_label": "Prompt",
    "llm_send_prompt": "Envoyer",
    "llm_response_label": "Réponse",
    "llm_llm_calling_error": "Erreur lors de l'appel au LLM : ",
    "llm_no_v1_label": "Pas de /v1 dans l'endpoint",
    "llm_timeout_label": "Timeout (secondes)",
    "llm_sys_prompt": "Prompt système pour le LLM",
    "llm_default_sys_prompt": "Tu es un assistant IA qui exécute fidèlement les tâches demandées sans rajouter de commentaires ou explications.",
    "llm_personas_tab": "Personas",
    "llm_personas_header": "Gérer les Personas",
    "llm_persona_name_label": "Nom du Persona",
    "llm_persona_prompt_label": "Prompt Système du Persona",
    "llm_add_persona": "Ajouter un Persona",
    "llm_select_persona": "Sélectionner un Persona",
    "llm_ollama_restart": "Redémarer Ollama",
    "llm_prompt_sequence_tab": "Séquence de Prompts",
})


class LlmPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        if not plugin_manager.config.get(name):
            plugin_manager.config[name] = self.get_config_fields()

    def get_config_fields(self):
        models = self.get_config("models")
        if isinstance(models, str):
            models = ast.literal_eval(models)
        model_list = [(m["name"], m["name"]) for m in models]
        personas = self.get_config("personas") or []
        if isinstance(personas, str):
            personas = ast.literal_eval(personas)
        persona_options = [("None", "None")] + [(p["name"], p["name"]) for p in personas]
        return {
            "api_keys": {
                "type": "json",
                "label": t("llm_keys_header"),
                "default": [
                    {"name": "groq_key", "value": ""},
                    {"name": "xai_key", "value": ""}
                ]
            },
            "apis": {
                "type": "json",
                "label": t("llm_apis_header"),
                "default": [
                    {"name": "groq", "url": "https://api.groq.com",
                        "api_key": "groq_key"},
                    {"name": "xai", "url": "https://api.x.ai", "api_key": "xai_key"},
                    {"name": "together", "url": "https://api.together.ai", "api_key": ""},
                    {"name": "deepseek", "url": "https://api.deepseek.com", "api_key": ""},
                    {"name": "ollama", "url": "http://localhost:11434", "api_key": ""},
                    {"name": "lmstudio", "url": "http://192.168.1.5:1234", "api_key": ""}
                ]
            },
            "models": {
                "type": "json",
                "label": t("llm_models_header"),
                "default": [{'name': 'ollama-qwen2.5:7b-instruct-q4_K_S', 'url': 'http://localhost:11434', 'model': 'qwen2.5:7b-instruct-q4_K_S', 'api_key': '', 'temperature': 0.7, 'max_tokens': 4096, 'delay': 0.0, 'max_retries': 3}]
            },
            "current_llm_model": {
                "type": "select",
                "label": t("llm_select_model"),
                "options": model_list,
                "default": "ollama-qwen2.5:7b-instruct-q4_K_S"
            },
            "llm_sys_prompt": {
                "type": "textarea",
                "label": t("llm_sys_prompt"),
                "default": t("llm_default_sys_prompt")
            },
            "personas": {
                "type": "json",
                "label": t("llm_personas_header"),
                "default": [
                    {"name": "Default", "prompt": t("llm_default_sys_prompt")},
                    {"name": "Youtuber", "prompt": "You are a charismatic YouTuber with a channel focused on tech reviews. Your opinions are bold, you love engaging your audience with humor, and your channel is called 'TechBit'."},
                    {"name": "Professional", "prompt": "You are a professional consultant providing clear, concise, and formal advice to corporate clients."}
                ]
            },
            "current_persona": {
                "type": "select",
                "label": t("llm_select_persona"),
                "options": persona_options,
                "default": "Default"
            }
        }

    def get_tabs(self):
        return [
            {"name": "LLM", "plugin": "llmplugin", "tab": "LLM"},
        ]

    def get_api_keys(self):
        if 'api_keys' not in st.session_state:
            api_keys = self.get_config("api_keys")
            if isinstance(api_keys, str):
                api_keys = ast.literal_eval(api_keys)
            st.session_state.api_keys = api_keys
        return st.session_state.api_keys

    def get_apis(self):
        if 'apis' not in st.session_state:
            apis = self.get_config("apis")
            if isinstance(apis, str):
                apis = ast.literal_eval(apis)
            st.session_state.apis = apis
        return st.session_state.apis

    def get_models(self):
        if 'models' not in st.session_state:
            models = self.get_config("models")
            if isinstance(models, str):
                models = ast.literal_eval(models)
            st.session_state.models = models
        return st.session_state.models

    def keys_tab(self, config):
        st.header(t("llm_keys_header"))
        self.get_api_keys()
        for i, key in enumerate(st.session_state.api_keys):
            with st.expander(key.get("name", f"Key {i}")):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input(t("llm_key_name_label"), value=key.get(
                        "name", ""), key=f"key_name_{i}")
                with col2:
                    value = st.text_input(t("llm_key_value_label"), value=key.get(
                        "value", ""), key=f"key_value_{i}")
                if st.button("Remove", key=f"remove_key_{i}"):
                    del st.session_state.api_keys[i]
                    st.rerun()
                st.session_state.api_keys[i] = {"name": name, "value": value}

        if st.button(t("llm_add_key")):
            st.session_state.api_keys.append({"name": "", "value": ""})
            st.rerun()

        if st.button("Save Keys"):
            config[self.name]["api_keys"] = st.session_state.api_keys
            self.plugin_manager.save_config(config)
            st.success("API Keys saved successfully!")

    def apis_tab(self, config):
        st.header(t("llm_apis_header"))
        self.get_apis()
        self.get_api_keys()

        key_options = [k["name"] for k in st.session_state.api_keys] + [""]

        for i, api in enumerate(st.session_state.apis):
            with st.expander(api.get("name", f"API {i}")):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input(t("llm_api_name_label"), value=api.get(
                        "name", ""), key=f"api_name_{i}")
                    url = st.text_input(t("llm_url_label"), value=api.get(
                        "url", ""), key=f"api_url_{i}")
                with col2:
                    api_key = st.selectbox(t("llm_api_key_label"), options=key_options,
                                           index=key_options.index(api.get("api_key", "")) if api.get(
                                               "api_key", "") in key_options else len(key_options)-1,
                                           key=f"api_key_{i}")
                    if api_key == "":
                        api_key = st.text_input(
                            "Manual API Key", value="", key=f"manual_key_{i}")
                    # Ajout du checkbox pour no_v1
                    no_v1 = st.checkbox(t("llm_no_v1_label"),
                                        value=api.get("no_v1", False),
                                        key=f"no_v1_{i}")
                if st.button("Remove", key=f"remove_api_{i}"):
                    del st.session_state.apis[i]
                    st.rerun()
                st.session_state.apis[i] = {
                    "name": name, "url": url, "api_key": api_key, "no_v1": no_v1}

        if st.button(t("llm_add_api")):
            st.session_state.apis.append(
                {"name": "", "url": "", "api_key": "", "no_v1": False})
            st.rerun()

        if st.button("Save APIs"):
            config[self.name]["apis"] = st.session_state.apis
            self.plugin_manager.save_config(config)
            st.success("APIs saved successfully!")

    def get_available_models(self, url, api_key, endpoint="/v1/models"):
        """Récupère les modèles disponibles via l'API spécifiée."""
        models = [""]
        try:
            if not url:
                return models
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            response = requests.get(
                f"{url}{endpoint}", headers=headers, timeout=5)
            response.raise_for_status()
            if response.status_code == 200:
                data = response.json()
                if endpoint == "/api/tags" and "models" in data and data["models"]:
                    return [model["name"] for model in data["models"]]
                if endpoint == "/tags" and "models" in data and data["models"]:
                    return [model["name"] for model in data["models"]]
                elif endpoint == "/v1/models" and "data" in data and data["data"]:
                    return [model["id"] for model in data["data"] if model.get("object") == "model"]
                elif endpoint == "/v1/models":
                    return [model["id"] for model in data if model.get("object") == "model"]
            return models
        except Exception as e:
            st.warning(
                f"Could not fetch models from {url}{endpoint}: {str(e)}")
            return models

    def get_sidebar_config_ui(self, expander, config: Dict[str, Any]) -> Dict[str, Any]:
        self.get_models()
        self.get_personas()
        available_models = [m["name"] for m in st.session_state.models]
        available_personas = ["None"] + [p["name"] for p in st.session_state.personas]
        default_model = config[self.name].get("current_llm_model", available_models[0] if available_models else "Unfound")
        default_persona = config[self.name].get("current_persona", "None")

        selected_model = expander.selectbox(
            t("llm_select_model"),
            options=available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            key="llm_api_model"
        )
        selected_persona = expander.selectbox(
            t("llm_select_persona"),
            options=available_personas,
            index=available_personas.index(default_persona) if default_persona in available_personas else 0,
            key="llm_persona"
        )

        config[self.name]["current_llm_model"] = selected_model
        config[self.name]["current_persona"] = selected_persona
        self.plugin_manager.save_config(config)
        return {"current_llm_model": selected_model, "current_persona": selected_persona}

    def models_tab(self, config):
        st.header(t("llm_models_header"))
        if 'models' not in st.session_state:
            st.session_state.models = config.get(self.name, {}).get(
                "models", self.get_config_fields()["models"]["default"])
        if 'apis' not in st.session_state:
            st.session_state.apis = config.get(self.name, {}).get(
                "apis", self.get_config_fields()["apis"]["default"])
        if 'available_models_cache' not in st.session_state:
            st.session_state.available_models_cache = {}

        api_options = [
            f"{api['name']} ({api['url']})" for api in st.session_state.apis]
        selected_api = st.selectbox(t("llm_select_api"), api_options)
        api = next(
            a for a in st.session_state.apis if f"{a['name']} ({a['url']})" == selected_api)
        url, api_key = api["url"], api["api_key"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button(t("llm_get_models") + "/api/tags"):
                st.session_state.available_models_cache[url] = self.get_available_models(
                    url, api_key, "/api/tags")
        with col2:
            if st.button(t("llm_get_models") + "/v1/models"):
                st.session_state.available_models_cache[url] = self.get_available_models(url, api_key, "/v1/models")
        with col3:
            if st.button(t("llm_get_models") + "/tags"):
                st.session_state.available_models_cache[url] = self.get_available_models(url, api_key, "/tags")
        with col4:
            if st.button(t("llm_get_models") + "/models"):
                st.session_state.available_models_cache[url] = self.get_available_models(url, api_key, "/models")

        if 'llm_api_model' not in st.session_state:
            st.session_state.llm_api_model = None
        available_models = st.session_state.available_models_cache.get(url, [""])
        selected_model = st.selectbox(t("llm_model_label"), available_models,
                                      index=available_models.index(st.session_state.get("llm_api_model", available_models[0] if available_models else None)) if st.session_state.get("llm_api_model") in available_models else 0)
        default_name = f"{api['name']}-{selected_model}" if selected_model else ""
        model_name = st.text_input(t("llm_model_name_label"), value=default_name)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            temp = st.number_input(t("llm_temp_label"), min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        with col2:
            max_tokens = st.number_input(t("llm_max_tokens_label"), min_value=1, value=4096, step=100)
        with col3:
            delay = st.number_input(t("llm_delay_label"), min_value=0.0, value=0.0, step=0.1)
        with col4:
            max_retries = st.number_input(t("llm_max_retries_label"), min_value=1, value=1, step=1)

        # Ajout du champ timeout
        timeout = st.number_input("Timeout (seconds)", min_value=1, value=3, step=1)

        if st.button(t("llm_add_model")) and model_name:
            st.session_state.models.append({
                "name": model_name or default_name,
                "url": url,
                "model": selected_model if selected_model else model_name,
                "api_key": api_key,
                "temperature": temp,
                "max_tokens": max_tokens,
                "delay": delay,
                "max_retries": max_retries,
                "timeout": timeout  # Ajout du timeout
            })
            st.rerun()

        for i, model in enumerate(st.session_state.models):
            with st.expander(model.get("name", f"Model {i}")):
                name = st.text_input(t("llm_model_name_label"), value=model.get(
                    "name", ""), key=f"model_name_{i}")
                url = st.text_input(t("llm_url_label"), value=model.get(
                    "url", ""), key=f"model_url_{i}")
                model_name = st.text_input(
                    t("llm_model_label"), value=model.get("model", ""), key=f"model_{i}")
                api_key = st.text_input(t("llm_api_key_label"), value=model.get(
                    "api_key", ""), key=f"model_key_{i}")
                temp = st.number_input(t("llm_temp_label"), min_value=0.0, max_value=2.0, value=model.get(
                    "temperature", 0.7), step=0.1, key=f"temp_{i}")
                max_tokens = st.number_input(t("llm_max_tokens_label"), min_value=1, value=model.get(
                    "max_tokens", 4096), step=100, key=f"max_tokens_{i}")
                delay = st.number_input(t("llm_delay_label"), min_value=0.0, value=model.get(
                    "delay", 0.0), step=0.1, key=f"delay_{i}")
                max_retries = st.number_input(t("llm_max_retries_label"), min_value=1, value=model.get(
                    "max_retries", 3), step=1, key=f"max_retries_{i}")
                # Ajout du champ timeout pour l'édition
                timeout = st.number_input("Timeout (seconds)", min_value=1, value=model.get(
                    "timeout", 3), step=1, key=f"timeout_{i}")
                if st.button("Remove", key=f"remove_model_{i}"):
                    del st.session_state.models[i]
                    st.rerun()
                st.session_state.models[i] = {
                    "name": name, "url": url, "model": model_name, "api_key": api_key,
                    "temperature": temp, "max_tokens": max_tokens, "delay": delay,
                    "max_retries": max_retries, "timeout": timeout  # Ajout du timeout
                }

        if st.button("Save Models"):
            config[self.name]["models"] = st.session_state.models
            self.plugin_manager.save_config(config)
            st.success("Models saved successfully!")

    def call_llm(self, url, api_key, model, prompts, sysprompt=None, temperature=0.7, max_tokens=4096, delay=0, max_retries=1, no_v1=False, timeout=3):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        headers["Content-Type"] = "application/json"

        # Récupérer le sysprompt par défaut si aucun n'est fourni
        sysprompt = sysprompt or self.plugin_manager.config['llm']['llm_sys_prompt']

        # Créer la liste des messages
        messages = [{"role": "system", "content": sysprompt}]

        # Ajouter un message système pour le persona si sélectionné
        current_persona_name = self.plugin_manager.config.get(self.name, {}).get("current_persona", "None")
        if current_persona_name != "None":
            personas = self.get_personas()
            persona = next((p for p in personas if p["name"] == current_persona_name), None)
            if persona:
                messages.append({"role": "system", "content": persona["prompt"]})

        # Gérer prompts comme chaîne ou liste
        if isinstance(prompts, str):
            messages.append({"role": "user", "content": prompts})
        elif isinstance(prompts, list):
            for prompt in prompts:
                messages.append({"role": "user", "content": prompt})
        else:
            raise ValueError("Prompts must be a string or a list of strings")

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        endpoint = "/chat/completions" if no_v1 else "/v1/chat/completions"
        full_url = f"{url}{endpoint}"

        attempts = 0
        while attempts < max_retries:
            try:
                response = requests.post(full_url, headers=headers, data=json.dumps(payload), timeout=timeout)
                response.raise_for_status()
                data = response.json()
                time.sleep(delay)
                result = data["choices"][0]["message"]["content"] if "choices" in data else "Error: Unexpected response format"
                no_think_result = re.sub(r'<think>[\s\S]*?</think>', '', result).strip()
                return no_think_result
            except Exception as e:
                st.warning(f"Failed to call {model} at {full_url} with {api_key} wait {delay}s timeout {timeout}s : {str(e)}")
                attempts += 1
                if attempts == max_retries:
                    return f"Error: Failed after {max_retries} attempts - {str(e)}"
        return "Error: No response"

    def call_all_llms(self, prompt: str) -> List[Dict[str, str]]:
        """Envoie le prompt à tous les modèles configurés et retourne les résultats."""
        self.get_api_keys()
        self.get_models()
        self.get_apis()

        results = []

        for model in st.session_state.models:
            try:
                # Récupérer le paramètre no_v1 de l'API associée
                api = next((a for a in st.session_state.apis if a["url"] == model["url"]), None)
                no_v1 = api.get("no_v1", False) if api else False
                api_key = next((k["value"] for k in st.session_state.api_keys if k["name"] == model["api_key"]), "")

                response = self.call_llm(
                    url=model["url"],
                    api_key=api_key,
                    model=model["model"],
                    prompt=prompt,
                    temperature=model["temperature"],
                    max_tokens=model["max_tokens"],
                    delay=model["delay"],
                    max_retries=model["max_retries"],
                    no_v1=no_v1,
                    timeout=model.get("timeout", 3)
                )

                results.append({
                    "model": model["name"],
                    "response": response,
                    "error": None
                })
            except Exception as e:
                results.append({
                    "model": model["name"],
                    "response": None,
                    "error": str(e)
                })

            time.sleep(0.1)  # Petit délai entre les appels

        return results

    def process_with_llm(self, prompt, sysprompt: str = None, context = None, repeat_on_failure: bool = True, number_repeat: int = 1) -> str:
        self.get_api_keys()
        self.get_models()
        self.get_apis()

        current_model_name = self.plugin_manager.config.get(self.name, {}).get(
            "current_llm_model", st.session_state.models[0]["name"] if st.session_state.models else None)

        if not current_model_name or not st.session_state.models:
            return f"{t('llm_llm_calling_error')}No model configured or selected"

        model = next((m for m in st.session_state.models if m["name"] == current_model_name), None)
        if not model:
            return f"{t('llm_llm_calling_error')}Selected model not found"

        # Récupérer le paramètre no_v1 de l'API associée
        api = next((a for a in st.session_state.apis if a["url"] == model["url"]), None)
        no_v1 = api.get("no_v1", False) if api else False

        attempt = 0
        max_delay = 60
        api_key = next((k["value"] for k in st.session_state.api_keys if k["name"] == model["api_key"]), "")
        while attempt <= number_repeat:
            try:
                # Envoyer contexte et prompt comme une liste
                if isinstance(prompt, str):
                    prompts = [prompt]
                else:
                    prompts = prompt

                prompts.append(context)
                return self.call_llm(
                    url=model["url"],
                    api_key=api_key,
                    model=model["model"],
                    prompts=prompts,
                    sysprompt=sysprompt,
                    temperature=model["temperature"],
                    max_tokens=model["max_tokens"],
                    delay=int(model["delay"]),
                    max_retries=1,
                    no_v1=no_v1,
                    timeout=model.get("timeout", 3)
                )
            except Exception as e:
                if not repeat_on_failure or attempt == number_repeat:
                    return f"{t('llm_llm_calling_error')}{str(e)}"
                delay = min(2 ** attempt * model["delay"] if model["delay"] > 0 else 2 ** attempt, max_delay)
                st.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
                attempt += 1
        return f"{t('llm_llm_calling_error')}Max retries exceeded"

    # Modifier chat_tab pour passer no_v1
    def chat_tab(self, config):
        st.header(t("llm_chat_header"))
        if 'models' not in st.session_state:
            st.session_state.models = config.get(self.name, {}).get(
                "models", self.get_config_fields()["models"]["default"])

        current_model_name = config.get(self.name, {}).get(
            "current_llm_model", st.session_state.models[0]["name"] if st.session_state.models else None)

        if not current_model_name or not st.session_state.models:
            st.write("No models configured yet or no current model selected.")
            return

        model = next(
            (m for m in st.session_state.models if m["name"] == current_model_name), None)
        if not model:
            st.write("Selected model not found in the list.")
            return

        # Récupérer le paramètre no_v1 de l'API associée
        api = next(
            (a for a in st.session_state.apis if a["url"] == model["url"]), None)
        api_key = next(
            (k["value"] for k in st.session_state.api_keys if k["name"] == model["api_key"]), "")
        st.write(f"Api key : {api_key}")
        no_v1 = api.get("no_v1", False) if api else False

        st.write(f"Current Model: {model['name']}")
        prompt = st.text_area(t("llm_prompt_label"), height=100)

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button(t("llm_send_prompt")) and prompt:
                with st.spinner("Generating response..."):
                    response = self.call_llm(
                        url=model["url"],
                        api_key=api_key,
                        model=model["model"],
                        prompts=prompt,
                        temperature=model["temperature"],
                        max_tokens=model["max_tokens"],
                        delay=model["delay"],
                        max_retries=model["max_retries"],
                        no_v1=no_v1,
                        timeout=model.get("timeout", 3)
                    )
                    st.subheader(t("llm_response_label"))
                    st.write(response)

        with col2:
            if st.button("Envoyer à tous") and prompt:
                with st.spinner("Envoi à tous les modèles en cours..."):
                    results = self.call_all_llms(prompt)

                    # Affichage des résultats sous forme de tableau
                    st.subheader("Résultats de tous les modèles")

                    # Création du tableau
                    for result in results:
                        with st.container():
                            cols = st.columns([1, 3])
                            with cols[0]:
                                st.markdown(f"**{result['model']}**")
                            with cols[1]:
                                if result["error"]:
                                    st.error(f"Erreur: {result['error']}")
                                else:
                                    st.markdown(result["response"])

    def free_ollama(self):
        try:
            ollama_model = "qwen2:1.5b"  # smallest
            st.info("Freeing ollama memory "+ollama_model)
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": "bye",
                    "keep_alive": 0
                }
            )
        except Exception as e:
            raise e

    def run_command(self, command: str, sudo: bool = False) -> Tuple[bool, str]:
        """Exécute une commande shell avec ou sans sudo."""
        try:
            if sudo:
                command = f"sudo -S {command}"  # -S permet de lire le mot de passe depuis stdin
                result = subprocess.run(
                    command.split(),
                    input=st.session_state.get("sudo_pwd", "") + "\n",
                    capture_output=True,
                    text=True,
                    check=True
                )
            else:
                result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
            return (True, result.stdout)
        except subprocess.CalledProcessError as e:
            return (False, e.stderr)

    def is_ollama_running(self) -> bool:
        """Vérifie si Ollama tourne."""
        success, _ = self.run_command("pgrep -f ollama")
        return success

    def stop_ollama(self) -> bool:
        """Arrête Ollama avec escalade de privilèges si nécessaire."""
        tab1, tab2 = st.tabs(["Méthode standard", "Avec sudo"])

        with tab1:
            if st.button("Arrêt normal"):
                success, output = self.run_command("pkill -f ollama")
                if success:
                    st.success("Arrêt réussi sans sudo")
                    return True
                else:
                    st.warning("Échec de l'arrêt normal")

        with tab2:
            if "sudo_pwd" not in st.session_state:
                st.session_state.sudo_pwd = st.text_input("Mot de passe sudo", type="password")

            if st.button("Forcer l'arrêt (sudo)"):
                success, output = self.run_command("pkill -9 -f ollama", sudo=True)
                if success:
                    st.success("Arrêt forcé réussi")
                    return True
                else:
                    st.error(f"Échec sudo : {output}")

        return False

    def start_ollama(self) -> bool:
        """Démarre Ollama avec gestion des droits."""
        choice = st.radio(
            "Mode de démarrage",
            ["Utilisateur normal", "Privilèges élevés (systemd)"],
            horizontal=True
        )

        if st.button("Démarrer"):
            if choice == "Utilisateur normal":
                self.ollama_process = subprocess.Popen(["ollama", "serve"])
                st.session_state.start_mode = "user"
            else:
                success, output = self.run_command("systemctl start ollama", sudo=True)
                if not success:
                    st.error(f"Erreur systemd : {output}")
                    return False
                st.session_state.start_mode = "systemd"

            if self.wait_for_ollama():
                st.success("Ollama est opérationnel !")
                return True
        return False

    def wait_for_ollama(self, timeout: int = 30) -> bool:
        """Attend que le serveur soit prêt."""
        with st.spinner("Attente du démarrage..."):
            for _ in range(timeout):
                try:
                    requests.get("http://localhost:11434", timeout=1)
                    return True
                except:
                    time.sleep(1)
        st.error("Timeout : serveur non répondant")
        return False

    def restart_ollama(self):
        """Interface complète de redémarrage."""
        st.title("🔌 Gestion Ollama - Nécessite sudo")

        if not self.is_ollama_running():
            if st.button("Démarrer simple (sans sudo)"):
                self.start_ollama()
            return

        with st.expander("Journal système (sudo requis)"):
            if st.button("Afficher les logs"):
                _, logs = self.run_command("journalctl -u ollama -n 20", sudo=True)
                st.code(logs)

        if self.stop_ollama():
            time.sleep(2)  # Pause entre arrêt/démarrage
            self.start_ollama()

    def get_personas(self):
        if 'personas' not in st.session_state:
            if not 'personas' in self.plugin_manager.config[self.name] :
                st.session_state.personas = []
                return []
            personas = self.get_config("personas")
            if isinstance(personas, str):
                personas = ast.literal_eval(personas)
            st.session_state.personas = personas
        return st.session_state.personas

    def personas_tab(self, config):
        st.header(t("llm_personas_header"))
        if 'personas' not in st.session_state:
            personas = self.get_config("personas")
            if isinstance(personas, str):
                personas = ast.literal_eval(personas)
            st.session_state.personas = personas

        for i, persona in enumerate(st.session_state.personas):
            with st.expander(persona.get("name", f"Persona {i}")):
                col1, col2 = st.columns([1, 4])
                with col1:
                    name = st.text_input(t("llm_persona_name_label"), value=persona.get(
                        "name", ""), key=f"persona_name_{i}")
                with col2:
                    prompt = st.text_area(t("llm_persona_prompt_label"), value=persona.get(
                        "prompt", ""), key=f"persona_prompt_{i}")
                if st.button("Remove", key=f"remove_persona_{i}"):
                    del st.session_state.personas[i]
                    st.rerun()
                st.session_state.personas[i] = {"name": name, "prompt": prompt}

        if st.button(t("llm_add_persona")):
            st.session_state.personas.append({"name": "", "prompt": ""})
            st.rerun()

        if st.button("Save Personas"):
            config[self.name]["personas"] = st.session_state.personas
            self.plugin_manager.save_config(config)
            st.success("Personas saved successfully!")

    def ollama_restart(self, config):
        self.restart_ollama()

    def run(self, config):
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [t("llm_keys_tab"), t("llm_apis_tab"), t("llm_models_tab"),
             t("llm_personas_tab"), t("llm_chat_tab"), t("llm_prompt_sequence_tab"),
             t("llm_ollama_restart")])
        with tab1:
            self.keys_tab(config)
        with tab2:
            self.apis_tab(config)
        with tab3:
            self.models_tab(config)
        with tab4:
            self.personas_tab(config)
        with tab5:
            self.chat_tab(config)
        with tab6:
            from widgets.prompt_sequence import PromptSequenceWidget
            PromptSequenceWidget("prompt_sequence", f"{self.name}_prompt_sequence", self.plugin_manager).display()
        with tab7:
            self.ollama_restart(config)
if __name__ == "__main__":
    st.write("LLM Plugin standalone test")
