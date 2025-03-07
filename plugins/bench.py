# bench.py
from global_vars import translations, t
from app import Plugin
import streamlit as st
from plugins.common import remove_quotes
from plugins.ragllm import RagllmPlugin
import os
import requests
import torch
import gc
import json

# Translations
translations["en"].update({
    "bench_tab": "LLM Benchmark",
    "config_tab": "Configuration",
    "compare_tab": "Compare Results",
    "bench_header": "LLM Benchmark Tool",
    "servers_list": "LLM Servers",
    "prompts_list": "Test Prompts",
    "add_server": "Add Server",
    "add_prompt": "Add Prompt",
    "url_label": "API URL",
    "api_key_label": "API Key (optional)",
    "model_label": "Model",
    "prompt_label": "Prompt",
    "run_bench": "Run Benchmark",
    "running_bench": "Running benchmark...",
    "select_models": "Select models to benchmark",
    "compare_models": "Compare Models",
    "model1_label": "Model 1",
    "model2_label": "Model 2",
})

translations["fr"].update({
    "bench_tab": "Benchmark LLM",
    "config_tab": "Configuration",
    "compare_tab": "Comparer Résultats",
    "bench_header": "Outil de Benchmark LLM",
    "servers_list": "Serveurs LLM",
    "prompts_list": "Prompts de Test",
    "add_server": "Ajouter Serveur",
    "add_prompt": "Ajouter Prompt",
    "url_label": "URL API",
    "api_key_label": "Clé API (optionnel)",
    "model_label": "Modèle",
    "prompt_label": "Prompt",
    "run_bench": "Lancer Benchmark",
    "running_bench": "Exécution du benchmark...",
    "select_models": "Sélectionner modèles à tester",
    "compare_models": "Comparer Modèles",
    "model1_label": "Modèle 1",
    "model2_label": "Modèle 2",
})


class BenchPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.ragllm_plugin = self.plugin_manager.get_plugin('ragllm')
        if 'bench_results' not in st.session_state:
            st.session_state.bench_results = {}
        # Ensure config has our structure
        if not plugin_manager.config.get(name):
            plugin_manager.config[name] = self.get_config_fields()

    def get_config_fields(self):
        return {
            "bench_servers": {
                "type": "json",
                "label": t("servers_list"),
                "default": [
                    {"url": "http://localhost:11434", "api_key": "", "model": ""},
                    {"url": "http://192.168.1.5:1234", "api_key": "", "model": ""}
                ]
            },
            "bench_prompts": {
                "type": "json",
                "label": t("prompts_list"),
                "default": ["Hello, how are you?", "Explain quantum physics simply"]
            }
        }

    def get_tabs(self):
        return [
            {"name": t("config_tab"), "plugin": "benchplugin"},
            {"name": t("bench_tab"), "plugin": "benchplugin"},
            {"name": t("compare_tab"), "plugin": "benchplugin"}
        ]

    def run(self, config):
        # Define the tabs
        tab1, tab2, tab3 = st.tabs(
            [t("config_tab"), t("bench_tab"), t("compare_tab")])

        # Reset button available in all tabs
        if st.button("Reset Session State"):
            st.session_state.clear()
            plugin_config = config.get(self.name, {})
            st.session_state.servers = plugin_config.get(
                "bench_servers", self.get_config_fields()["bench_servers"]["default"])
            st.session_state.prompts = plugin_config.get(
                "bench_prompts", self.get_config_fields()["bench_prompts"]["default"])
            st.rerun()

        # Config tab
        with tab1:
            self.config_tab(config)

        # Bench tab
        with tab2:
            self.bench_tab(config)

        # Compare tab
        with tab3:
            self.compare_tab(config)

    def config_tab(self, config):
        st.header(t("servers_list"))

        plugin_config = config.get(self.name, {})
        if 'servers' not in st.session_state:
            st.session_state.servers = plugin_config.get(
                "bench_servers", self.get_config_fields()["bench_servers"]["default"])
        if 'prompts' not in st.session_state or len(st.session_state.prompts) == 0:
            # Ensure we get prompts from config, falling back to default if not present
            prompts = plugin_config.get("bench_prompts")
            if not prompts or not isinstance(prompts, list):
                prompts = self.get_config_fields()["bench_prompts"]["default"]
            st.session_state.prompts = prompts

        for i, server in enumerate(st.session_state.servers):
            url = server.get("url", "")
            model = server.get("model", "")
            server_name = self.get_server_display_name(url, model)

            with st.expander(server_name):
                if not isinstance(server, dict):
                    server = {"url": server, "api_key": "", "model": ""} if isinstance(
                        server, str) else {"url": "", "api_key": "", "model": ""}

                col1, col2 = st.columns([3, 1])
                with col1:
                    url = st.text_input(t("url_label"), value=server.get(
                        "url", ""), key=f"url_{i}")
                with col2:
                    if st.button("Refresh Models", key=f"refresh_{i}"):
                        st.session_state.servers[i]["url"] = url

                api_key = st.text_input(t("api_key_label"), value=server.get(
                    "api_key", ""), key=f"key_{i}")
                models = self.get_available_models(url, api_key)

                if len(models) == 1 and models[0] == "":
                    model = st.text_input(t("model_label"), value=server.get("model", ""),
                                          placeholder="Enter model name manually", key=f"model_{i}")
                else:
                    model = st.selectbox(t("model_label"), options=models,
                                         index=models.index(server.get("model", "")) if server.get(
                                             "model", "") in models else 0,
                                         key=f"model_{i}")

                if st.button("Remove", key=f"remove_{i}"):
                    del st.session_state.servers[i]
                    st.rerun()
                    continue

                st.session_state.servers[i] = {
                    "url": url, "api_key": api_key, "model": model}

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Add Ollama"):
                st.session_state.servers.append(
                    {"url": "http://localhost:11434", "api_key": "", "model": ""})
                st.rerun()
        with col2:
            if st.button("Add LM Studio"):
                st.session_state.servers.append(
                    {"url": "http://192.168.1.5:1234", "api_key": "", "model": ""})
                st.rerun()
        with col3:
            if st.button("Add API"):
                st.session_state.servers.append(
                    {"url": "", "api_key": "", "model": ""})
                st.rerun()

        st.header(t("prompts_list"))
        if not st.session_state.prompts:
            st.write("No prompts defined yet.")
        else:
            for i, prompt in enumerate(st.session_state.prompts):
                updated_prompt = st.text_area(
                    t("prompt_label"), value=prompt, key=f"prompt_{i}")
                if st.button("Remove", key=f"remove_prompt_{i}"):
                    del st.session_state.prompts[i]
                    st.rerun()
                    continue
                st.session_state.prompts[i] = updated_prompt

        if st.button(t("add_prompt")):
            st.session_state.prompts.append("")
            st.rerun()

        if st.button("Save Configuration"):
            config[self.name] = {
                "bench_servers": st.session_state.servers,
                "bench_prompts": st.session_state.prompts
            }
            self.plugin_manager.save_config(config)
            st.success("Configuration saved successfully!")

    def get_available_models(self, url, api_key):
        """Try to get models from both Ollama and LM Studio endpoints with better empty response handling"""
        models = [""]
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

            # Try Ollama endpoint
            response = requests.get(f"{url}/api/tags", headers=headers)
            if response.status_code == 200:
                data = response.json()
                # Check if response has actual models
                if data and "models" in data and data["models"]:
                    return [model["name"] for model in data["models"]]

            # Try LM Studio endpoint
            response = requests.get(f"{url}/v1/models", headers=headers)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "data" in data and data["data"]:
                    return [model["id"] for model in data["data"] if model.get("object") == "model"]

            return models
        except Exception as e:
            st.warning(f"Could not fetch models from {url}: {str(e)}")
            return models

    def get_server_display_name(self, url: str, model: str = "") -> str:
        """Decode server URL into a friendly display name with optional model"""
        if "localhost:11434" in url:
            name = "Ollama"
        elif "192.168.1.5:1234" in url:
            name = "LM Studio"
        else:
            name = url  # Keep full URL for custom APIs
        return f"{model} ({name})" if model else name

    def reset_cuda_context(self):
        """Clean up CUDA context"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
            torch.cuda.synchronize()

    def call_llm(self, url: str, api_key: str, model: str, prompt: str, sysprompt: str = "You are a helpful AI assistant") -> str:
        """Custom LLM call for benchmarking different endpoints"""
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            headers["Content-Type"] = "application/json"

            # Standard OpenAI-compatible payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sysprompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }

            # Ollama uses /api/chat, LM Studio uses /v1/chat/completions
            endpoint = "/v1/chat/completions"

            response = requests.post(
                f"{url}{endpoint}", headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            data = response.json()

            # Handle different response formats
            if "choices" in data:  # OpenAI-compatible (LM Studio)
                return data["choices"][0]["message"]["content"]
            elif "response" in data:  # Ollama
                return data["response"]
            else:
                return "Error: Unexpected response format"

        except Exception as e:
            raise e
            return f"Error calling LLM at {url}: {str(e)}"

    def bench_tab(self, config):
        st.header(t("bench_header"))

        servers = st.session_state.get("servers", config.get(
            self.name, {}).get("bench_servers", []))
        prompts = st.session_state.get("prompts", config.get(
            self.name, {}).get("bench_prompts", []))

        model_options = [self.get_server_display_name(
            s["url"], s["model"]) for s in servers if s.get("model")]

        selected_models = st.multiselect(
            t("select_models"),
            model_options
        )

        if st.button(t("run_bench")) and selected_models:
            with st.spinner(t("running_bench")):
                self.reset_cuda_context()

                # Calculate total tasks (models × prompts)
                total_tasks = len(selected_models) * len(prompts)
                progress_bar = st.progress(0.0)
                tasks_completed = 0

                for model_id in selected_models:
                    server = next(s for s in servers if self.get_server_display_name(
                        s["url"], s["model"]) == model_id)
                    with st.expander(f"Results for {model_id}"):
                        results = []
                        for prompt in prompts:
                            try:
                                response = self.call_llm(
                                    url=server["url"],
                                    api_key=server["api_key"],
                                    model=server["model"],
                                    prompt=prompt
                                )
                                st.write(f"Prompt: {prompt}")
                                st.write(f"Response: {response}")
                                results.append(
                                    {"prompt": prompt, "response": response})
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                            # Update progress
                            tasks_completed += 1
                            progress_bar.progress(
                                tasks_completed / total_tasks)

                        st.session_state.bench_results[model_id] = results

                # Ensure progress reaches 100% at the end
                progress_bar.progress(1.0)

    def compare_tab(self, config):
        st.header(t("compare_models"))

        available_models = list(st.session_state.bench_results.keys())
        if not available_models:
            st.write("No benchmark results available yet")
            return

        col1, col2 = st.columns(2)
        with col1:
            model1 = st.selectbox(t("model1_label"), available_models)
        with col2:
            model2 = st.selectbox(t("model2_label"), available_models)

        if model1 and model2:
            results1 = st.session_state.bench_results.get(model1, [])
            results2 = st.session_state.bench_results.get(model2, [])

            for i, (r1, r2) in enumerate(zip(results1, results2)):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Prompt: {r1['prompt']}")
                    st.write(f"Response: {r1['response']}")
                with col2:
                    st.write(f"Prompt: {r2['prompt']}")
                    st.write(f"Response: {r2['response']}")

                if i < len(results1) - 1:
                    st.divider()


if __name__ == "__main__":
    st.write("Bench Plugin standalone test")
