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
import ast
import time

try:
    from pylatexenc.latex2text import LatexNodes2Text
    LATEX_AVAILABLE = True
except ImportError:
    LATEX_AVAILABLE = False

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
    "expected_response_label": "Expected Response",
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
    "expected_response_label": "Réponse Attendue",
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
                "default": [
                    {"prompt": "Hello, how are you?",
                        "expected": "I'm doing great, thanks!"},
                    {"prompt": "Explain quantum physics simply",
                        "expected": "Quantum physics is about tiny particles behaving strangely."}
                ]
            },
            "nstart": {
                "type": "int",
                "label": "Characters to keep at start",
                "default": 200
            },
            "nend": {
                "type": "int",
                "label": "Characters to keep at end",
                "default": 200
            },
        }

    def get_tabs(self):
        return [
            {"name": t("config_tab"), "plugin": "benchplugin"},
            {"name": t("bench_tab"), "plugin": "benchplugin"},
            {"name": t("compare_tab"), "plugin": "benchplugin"}
        ]

    def shorten_response(self, response: str, nstart: int = 100, nend: int = 50) -> str:
        """Raccourcit une réponse en conservant nstart premiers et nend derniers caractères"""
        if len(response) <= nstart + nend:
            return response

        return f"{response[:nstart]}\n\n[...]\n\n{response[-nend:]}"

    def convert_latex_to_text(self, text: str) -> str:
        """Convert LaTeX markup in text to readable plain text."""
        if not text:
            return text

        # If pylatexenc is available, use it
        if LATEX_AVAILABLE:
            try:
                converter = LatexNodes2Text()
                return converter.latex_to_text(text)
            except Exception as e:
                st.warning(
                    f"Failed to convert LaTeX with pylatexenc: {str(e)}. Falling back to basic cleanup.")

        # Fallback: Basic regex cleanup for common LaTeX commands
        # Remove \[ \] or \( \) delimiters
        text = re.sub(r'\\\[(.*?)\\\]', r'\1', text)
        text = re.sub(r'\\\((.*?)\\\)', r'\1', text)
        # Replace common commands
        replacements = {
            r'\\boxed{(.*?)}': r'\1',           # Remove \boxed, keep content
            r'\\times': '×',                   # Multiplication symbol
            r'\\div': '÷',                     # Division symbol
            r'\\frac{(.*?)}{(.*?)}': r'\1/\2',  # Fraction to slash
            # Remove \, (thousands separator)
            r'(\d+)\\,!(\d+)': r'\1\2',
            r'(\d+)\^{(\d+)}': r'\1^\2',       # Keep exponent as-is
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        # Clean up extra spaces or braces
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('{', '').replace('}', '').strip()

        return text

    def run(self, config):
        # Define the tabs
        tab1, tab2, tab3 = st.tabs(
            [t("config_tab"), t("bench_tab"), t("compare_tab")])

        # Config tab
        with tab1:
            self.config_tab(config)

        # Bench tab
        with tab2:
            self.bench_tab(config)

        # Compare tab
        with tab3:
            self.compare_tab(config)

    def get_available_models(self, url, api_key):
        """Try to get models from both Ollama and LM Studio endpoints with better empty response handling"""
        models = [""]
        try:
            if not url:  # Early return if URL is empty
                return models

            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

            # Try Ollama endpoint
            response = requests.get(f"{url}/api/tags", headers=headers)
            if response.status_code == 200:
                data = response.json()
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
            return models  # Always return a list

    def get_server_display_name(self, url: str, model: str = "") -> str:
        """Decode server URL into a friendly display name with optional model"""
        if "localhost:11434" in url:
            name = "Ollama"
        elif "192.168.1.5:1234" in url:
            name = "LM Studio"
        else:
            name = url  # Keep full URL for custom APIs
        return f"{model} ({name})" if model else name

    def call_llm(self, url: str, api_key: str, model: str, prompt: str, sysprompt: str = "You are a helpful AI assistant") -> str:
        """Custom LLM call for benchmarking different endpoints"""
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            headers["Content-Type"] = "application/json"

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sysprompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }

            endpoint = "/v1/chat/completions"
            response = requests.post(
                f"{url}{endpoint}", headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            data = response.json()

            # Handle different response formats
            if "choices" in data:  # OpenAI-compatible (LM Studio)
                raw_response = data["choices"][0]["message"]["content"]
            elif "response" in data:  # Ollama
                raw_response = data["response"]
            else:
                raw_response = "Error: Unexpected response format"

            # Log raw response length
            st.write(
                f"API raw response length for {model} at {url}: {len(raw_response)} characters")

            # Convert LaTeX to readable text
            converted_response = self.convert_latex_to_text(raw_response)
            st.write(
                f"Length after LaTeX conversion: {len(converted_response)} characters")

            # Get nstart and nend from config
            config = self.plugin_manager.config.get(self.name, {})
            nstart = int(config.get("nstart", 100))
            nend = int(config.get("nend", 150))

            # Shorten the response
            shortened_response = self.shorten_response(
                converted_response, nstart, nend)
            st.write(
                f"Shortened response length: {len(shortened_response)} characters")

            return shortened_response

        except Exception as e:
            st.error(f"Error calling LLM at {url}: {str(e)}")
            return f"Error: {str(e)}"

    def get_cuda_memory_stats(self, device_index=0):
        """Retourne les stats de mémoire CUDA en Mo"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "max_allocated": 0, "reserved": 0}

        torch.cuda.set_device(device_index)
        allocated = torch.cuda.memory_allocated(device_index) / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated(
            device_index) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device_index) / 1024 / 1024

        return {
            "allocated": allocated,
            "max_allocated": max_allocated,
            "reserved": reserved
        }

    def config_tab(self, config):
        st.header(t("servers_list"))

        plugin_config = config.get(self.name, {})
        if 'servers' not in st.session_state:
            bench_servers = plugin_config.get("bench_servers")
            if isinstance(bench_servers, str):
                bench_servers = ast.literal_eval(bench_servers)
            st.session_state.servers = bench_servers
        if 'prompts' not in st.session_state or len(st.session_state.prompts) == 0:
            bench_prompts = plugin_config.get("bench_prompts")
            if isinstance(bench_prompts, str):
                bench_prompts = ast.literal_eval(bench_prompts)
            st.session_state.prompts = bench_prompts

        # Initialize a cache for available models if not already present
        if 'available_models_cache' not in st.session_state:
            st.session_state.available_models_cache = {}

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
                    new_url = st.text_input(
                        t("url_label"), value=server.get("url", ""), key=f"url_{i}")
                with col2:
                    refresh_key = f"refresh_{i}"
                    if st.button("Refresh Models", key=refresh_key):
                        # Fetch models only on refresh
                        st.session_state.available_models_cache[new_url] = self.get_available_models(
                            new_url, server.get("api_key", ""))
                        st.session_state.servers[i]["url"] = new_url

                api_key = st.text_input(t("api_key_label"), value=server.get(
                    "api_key", ""), key=f"key_{i}")

                # Check if we need to fetch models (new server, URL/API key changed, or refresh triggered)
                cache_key = f"{new_url}_{api_key}"
                if (cache_key not in st.session_state.available_models_cache or
                    new_url != url or
                        api_key != server.get("api_key", "")):
                    st.session_state.available_models_cache[cache_key] = self.get_available_models(
                        new_url, api_key)

                # Get cached models, default to [""] if not available
                models = st.session_state.available_models_cache.get(cache_key, [
                                                                     ""])

                # Handle model selection
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
                    # Optionally remove from cache if no other server uses this URL/API key combo
                    st.rerun()
                    continue

                st.session_state.servers[i] = {
                    "url": new_url, "api_key": api_key, "model": model}

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Add Ollama"):
                new_server = {"url": "http://localhost:11434",
                              "api_key": "", "model": ""}
                st.session_state.servers.append(new_server)
                # Pre-fetch models for new server
                st.session_state.available_models_cache[f"{new_server['url']}_{new_server['api_key']}"] = self.get_available_models(
                    new_server["url"], new_server["api_key"])
                st.rerun()
        with col2:
            if st.button("Add LM Studio"):
                new_server = {"url": "http://192.168.1.5:1234",
                              "api_key": "", "model": ""}
                st.session_state.servers.append(new_server)
                st.session_state.available_models_cache[f"{new_server['url']}_{new_server['api_key']}"] = self.get_available_models(
                    new_server["url"], new_server["api_key"])
                st.rerun()
        with col3:
            if st.button("Add API"):
                new_server = {"url": "", "api_key": "", "model": ""}
                st.session_state.servers.append(new_server)
                # No pre-fetch here since URL is empty; user must fill it first
                st.rerun()

        st.header(t("prompts_list"))
        if not st.session_state.prompts:
            st.write("No prompts defined yet.")
        else:
            for i, prompt_data in enumerate(st.session_state.prompts):
                col1, col2 = st.columns([2, 1])
                with col1:
                    prompt = st.text_area(t("prompt_label"), value=prompt_data.get(
                        "prompt", ""), key=f"prompt_{i}")
                with col2:
                    expected = st.text_area(t("expected_response_label"), value=prompt_data.get(
                        "expected", ""), key=f"expected_{i}")

                if st.button("Remove", key=f"remove_prompt_{i}"):
                    del st.session_state.prompts[i]
                    st.rerun()
                    continue

                st.session_state.prompts[i] = {
                    "prompt": prompt, "expected": expected}

        if st.button(t("add_prompt")):
            st.session_state.prompts.append({"prompt": "", "expected": ""})
            st.rerun()

        if st.button("Save Configuration"):
            config[self.name] = {
                "bench_servers": st.session_state.servers,
                "bench_prompts": st.session_state.prompts
            }
            self.plugin_manager.save_config(config)
            st.success("Configuration saved successfully!")

    def bench_tab(self, config):
        st.header(t("bench_header"))

        if torch.cuda.is_available():
            mem_before = self.get_cuda_memory_stats()
            st.write(
                f"Avant exécution - Mémoire allouée: {mem_before['reserved']:.2f} Mo")

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

        debug_mode = st.checkbox(
            "Debug (use only first 3 prompts)", value=False)

        if st.button(t("run_bench")) and selected_models:
            with st.spinner(t("running_bench")):
                st.session_state.bench_results = {}
                active_prompts = prompts[:3] if debug_mode and len(
                    prompts) > 3 else prompts

                total_tasks = len(selected_models) * len(active_prompts)
                progress_bar = st.progress(0.0)
                tasks_completed = 0

                sorted_models = sorted(
                    selected_models,
                    key=lambda model_id: 0 if "Ollama" in model_id else 1
                )

                server = ""
                for i, model_id in enumerate(sorted_models):
                    prev_server = server
                    server = next(s for s in servers if self.get_server_display_name(
                        s["url"], s["model"]) == model_id)
                    current_is_ollama = "localhost:11434" in server["url"]
                    if i == 0:
                        previous_is_ollama = current_is_ollama

                    if not current_is_ollama and previous_is_ollama:
                        st.write(
                            f"Transitioning from Ollama ({prev_server['model']}) to another server type. Resetting CUDA context...")
                        self.ragllm_plugin.free_llm(model=prev_server['model'])
                        previous_is_ollama = False

                    with st.expander(f"Results for {model_id}", expanded=True):
                        results = []
                        # Début de la mesure du temps pour ce modèle
                        start_time = time.time()

                        for prompt_data in active_prompts:
                            prompt = prompt_data["prompt"]
                            expected = prompt_data["expected"]
                            try:
                                response = self.call_llm(
                                    url=server["url"],
                                    api_key=server["api_key"],
                                    model=server["model"],
                                    prompt=prompt
                                )
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.write(f"Prompt: {prompt}")
                                    st.write(f"Response: {response}")
                                with col2:
                                    st.write(f"Expected: {expected}")
                                results.append({
                                    "prompt": prompt,
                                    "expected": expected,
                                    "response": response
                                })
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                            tasks_completed += 1
                            progress_bar.progress(
                                tasks_completed / total_tasks)

                        # Fin de la mesure du temps et calcul
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        # Affichage du temps dans bench_tab
                        st.write(
                            f"Total time for {len(active_prompts)} prompts: {elapsed_time:.2f} seconds")

                        # Stockage des résultats avec le temps
                        st.session_state.bench_results[model_id] = {
                            "results": results,
                            "total_time": elapsed_time
                        }

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
            # Récupération des résultats et temps
            results1 = st.session_state.bench_results.get(
                model1, {}).get("results", [])
            time1 = st.session_state.bench_results.get(
                model1, {}).get("total_time", 0)
            results2 = st.session_state.bench_results.get(
                model2, {}).get("results", [])
            time2 = st.session_state.bench_results.get(
                model2, {}).get("total_time", 0)

            # Affichage des temps totaux
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"{model1} - Total time: {time1:.2f} seconds")
            with col2:
                st.write(f"{model2} - Total time: {time2:.2f} seconds")

            # Affichage des résultats comme avant
            for i, (r1, r2) in enumerate(zip(results1, results2)):
                col0, col1, col2 = st.columns([1, 3, 3])
                with col0:
                    st.write(f"Expected: {r1['expected']}")
                with col1:
                    st.write(f"Prompt: {r1['prompt']}")
                    st.markdown(r1['response'])
                with col2:
                    st.write(f"Prompt: {r2['prompt']}")
                    st.markdown(r2['response'])

                if i < len(results1) - 1:
                    st.divider()


if __name__ == "__main__":
    st.write("Bench Plugin standalone test")
