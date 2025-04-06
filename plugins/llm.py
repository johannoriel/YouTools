# llm.py
from global_vars import translations, t
from app import Plugin
import streamlit as st
import requests
import json
import time
import random
from typing import List, Dict, Any
import ast

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
                    {"name": "groq", "url": "https://api.groq.com", "api_key": "groq_key"},
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
            }
        }

    def get_tabs(self):
        return [
            {"name": t("llm_llm_tab"), "plugin": "llmplugin"},]
        return [
            {"name": t("llm_keys_tab"), "plugin": "llmplugin", "tab": "keys"},
            {"name": t("llm_apis_tab"), "plugin": "llmplugin", "tab": "apis"},
            {"name": t("llm_models_tab"), "plugin": "llmplugin", "tab": "models"},
            {"name": t("llm_chat_tab"), "plugin": "llmplugin", "tab": "chat"}
        ]

    def run(self, config):
        tab1, tab2, tab3, tab4 = st.tabs(
            [t("llm_keys_tab"), t("llm_apis_tab"), t("llm_models_tab"), t("llm_chat_tab")])
        with tab1:
            self.keys_tab(config)
        with tab2:
            self.apis_tab(config)
        with tab3:
            self.models_tab(config)
        with tab4:
            self.chat_tab(config)

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
                    name = st.text_input(t("llm_key_name_label"), value=key.get("name", ""), key=f"key_name_{i}")
                with col2:
                    value = st.text_input(t("llm_key_value_label"), value=key.get("value", ""), key=f"key_value_{i}")
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
                    name = st.text_input(t("llm_api_name_label"), value=api.get("name", ""), key=f"api_name_{i}")
                    url = st.text_input(t("llm_url_label"), value=api.get("url", ""), key=f"api_url_{i}")
                with col2:
                    api_key = st.selectbox(t("llm_api_key_label"), options=key_options,
                                           index=key_options.index(api.get("api_key", "")) if api.get("api_key", "") in key_options else len(key_options)-1,
                                           key=f"api_key_{i}")
                    if api_key == "":
                        api_key = st.text_input("Manual API Key", value="", key=f"manual_key_{i}")
                if st.button("Remove", key=f"remove_api_{i}"):
                    del st.session_state.apis[i]
                    st.rerun()
                st.session_state.apis[i] = {"name": name, "url": url, "api_key": api_key}

        if st.button(t("llm_add_api")):
            st.session_state.apis.append({"name": "", "url": "", "api_key": ""})
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
            response = requests.get(f"{url}{endpoint}", headers=headers, timeout=5)
            response.raise_for_status()
            if response.status_code == 200:
                data = response.json()
                if endpoint == "/api/tags" and "models" in data and data["models"]:
                    return [model["name"] for model in data["models"]]
                if endpoint == "/tags" and "models" in data and data["models"]:
                    return [model["name"] for model in data["models"]]
                elif endpoint == "/v1/models" and "data" in data and data["data"]:
                    return [model["id"] for model in data["data"] if model.get("object") == "model"]
                elif endpoint == "/v1/models" :
                    return [model["id"] for model in data if model.get("object") == "model"]
            return models
        except Exception as e:
            st.warning(f"Could not fetch models from {url}{endpoint}: {str(e)}")
            return models

    def get_sidebar_config_ui(self, expander, config: Dict[str, Any]) -> Dict[str, Any]:
        self.get_models()
        available_models = [m["name"] for m in st.session_state.models]
        default_model = config.get(self.name, {}).get(
            "current_llm_model", available_models[0] if available_models else None)

        selected_model = expander.selectbox(
            t("llm_select_model"),
            options=available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            key="llm_api_model"
        )
        return {"current_llm_model": selected_model}

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

        api_options = [f"{api['name']} ({api['url']})" for api in st.session_state.apis]
        selected_api = st.selectbox(t("llm_select_api"), api_options)
        api = next(a for a in st.session_state.apis if f"{a['name']} ({a['url']})" == selected_api)
        url, api_key = api["url"], next((k["value"] for k in st.session_state.api_keys if k["name"] == api["api_key"]), "")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(t("llm_get_models") + "/api/tags"):
                st.session_state.available_models_cache[url] = self.get_available_models(url, api_key, "/api/tags")
        with col2:
            if st.button(t("llm_get_models") + "/v1/models"):
                st.session_state.available_models_cache[url] = self.get_available_models(url, api_key, "/v1/models")
        with col3:
            if st.button(t("llm_get_models") + "/tags"):
                st.session_state.available_models_cache[url] = self.get_available_models(url, api_key, "/tags")

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

        if st.button(t("llm_add_model")) and model_name:
            st.session_state.models.append({
                "name": model_name or default_name,
                "url": url,
                "model": selected_model if selected_model else model_name,
                "api_key": api_key,
                "temperature": temp,
                "max_tokens": max_tokens,
                "delay": delay,
                "max_retries": max_retries
            })
            st.rerun()

        for i, model in enumerate(st.session_state.models):
            with st.expander(model.get("name", f"Model {i}")):
                name = st.text_input(t("llm_model_name_label"), value=model.get("name", ""), key=f"model_name_{i}")
                url = st.text_input(t("llm_url_label"), value=model.get("url", ""), key=f"model_url_{i}")
                model_name = st.text_input(t("llm_model_label"), value=model.get("model", ""), key=f"model_{i}")
                api_key = st.text_input(t("llm_api_key_label"), value=model.get("api_key", ""), key=f"model_key_{i}")
                temp = st.number_input(t("llm_temp_label"), min_value=0.0, max_value=2.0, value=model.get("temperature", 0.7), step=0.1, key=f"temp_{i}")
                max_tokens = st.number_input(t("llm_max_tokens_label"), min_value=1, value=model.get("max_tokens", 4096), step=100, key=f"max_tokens_{i}")
                delay = st.number_input(t("llm_delay_label"), min_value=0.0, value=model.get("delay", 0.0), step=0.1, key=f"delay_{i}")
                max_retries = st.number_input(t("llm_max_retries_label"), min_value=1, value=model.get("max_retries", 3), step=1, key=f"max_retries_{i}")
                if st.button("Remove", key=f"remove_model_{i}"):
                    del st.session_state.models[i]
                    st.rerun()
                st.session_state.models[i] = {
                    "name": name, "url": url, "model": model_name, "api_key": api_key,
                    "temperature": temp, "max_tokens": max_tokens, "delay": delay, "max_retries": max_retries
                }

        if st.button("Save Models"):
            config[self.name]["models"] = st.session_state.models
            self.plugin_manager.save_config(config)
            st.success("Models saved successfully!")

    def call_llm(self, url, api_key, model, prompt, temperature=0.7, max_tokens=4096, delay=0, max_retries=1):
        """Appelle l'API LLM avec gestion des retries et du délai."""
        print(f"Generating with model {model} at {url}")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        headers["Content-Type"] = "application/json"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        attempts = 0
        while attempts < max_retries:
            try:
                print("Calling LLM...")
                response = requests.post(f"{url}/v1/chat/completions", headers=headers, data=json.dumps(payload), timeout=10)
                print(response)
                response.raise_for_status()
                data = response.json()
                time.sleep(delay)
                return data["choices"][0]["message"]["content"] if "choices" in data else "Error: Unexpected response format"
            except Exception as e:
                st.warning(f"Failed to call {response}")
                attempts += 1
                if attempts == max_retries:
                    return f"Error: Failed after {max_retries} attempts - {str(e)}"
        return "Error: No response"

    def process_with_llm(self, prompt: str, sysprompt: str, context: str, repeat_on_failure: bool = True, number_repeat: int = 1) -> str:
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

        attempt = 0
        max_delay = 60  # Maximum 1 minute entre appels
        api_key = next((k["value"] for k in st.session_state.api_keys if k["name"] == model["api_key"]), "")
        while attempt <= number_repeat:
            try:
                return self.call_llm(
                    url=model["url"],
                    api_key=api_key,
                    model=model["model"],
                    prompt=f"{context}\n\n{prompt}",
                    temperature=model["temperature"],
                    max_tokens=model["max_tokens"],
                    delay=0,
                    max_retries=1
                )
            except Exception as e:
                if not repeat_on_failure or attempt == number_repeat:
                    return f"{t('llm_llm_calling_error')}{str(e)}"
                # Calcul du délai exponentiel basé sur le delay par défaut du modèle
                delay = min(2 ** attempt * model["delay"] if model["delay"] > 0 else 2 ** attempt, max_delay)
                st.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
                attempt += 1
        raise Exception()
        return f"{t('llm_llm_calling_error')}Max retries exceeded"

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

        model = next((m for m in st.session_state.models if m["name"] == current_model_name), None)
        if not model:
            st.write("Selected model not found in the list.")
            return

        st.write(f"Current Model: {model['name']}")
        prompt = st.text_area(t("llm_prompt_label"), height=100)

        if st.button(t("llm_send_prompt")) and prompt:
            with st.spinner("Generating response..."):
                response = self.call_llm(
                    url=model["url"],
                    api_key=next((k["value"] for k in st.session_state.api_keys if k["name"] == model["api_key"]), ""),
                    model=model["model"],
                    prompt=prompt,
                    temperature=model["temperature"],
                    max_tokens=model["max_tokens"],
                    delay=model["delay"],
                    max_retries=model["max_retries"]
                )
                st.subheader(t("llm_response_label"))
                st.write(response)


if __name__ == "__main__":
    st.write("LLM Plugin standalone test")
