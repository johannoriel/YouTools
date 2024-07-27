import os
import json
import importlib
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv
from global_vars import translations, t

# Constantes
CONFIG_FILE = "config.json"

def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config: Dict[str, Any]):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

# Fonction pour mettre à jour la langue
def set_lang(language):
    st.session_state.lang = language

# Fonction de traduction
def t(key: str) -> str:
    return translations[st.session_state.lang].get(key, key)

class Plugin:
    def __init__(self, name, plugin_manager):
        self.name = name
        self.plugin_manager = plugin_manager

    def get_config_fields(self) -> Dict[str, Any]:
        return {}

    def get_config_ui(self, config):
        updated_config = {}
        for field, params in self.get_config_fields().items():
            if params['type'] == 'select':
                updated_config[field] = st.selectbox(
                    params['label'],
                    options=[option[0] for option in params['options']],
                    format_func=lambda x: dict(params['options'])[x],
                    index=[option[0] for option in params['options']].index(config.get(field, params['default']))
                )
            elif params['type'] == 'textarea':
                updated_config[field] = st.text_area(
                    params['label'],
                    value=config.get(field, params['default'])
                )
            else:
                updated_config[field] = st.text_input(
                    params['label'],
                    value=config.get(field, params['default']),
                    type="password" if field.startswith("pass") else "default"
                )
        return updated_config

    def get_tabs(self) -> List[Dict[str, Any]]:
        return []

    def run(self, config: Dict[str, Any]):
        pass

    def get_sidebar_config_ui(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {}

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}

    def load_plugins(self):
        plugins_dir = 'plugins'
        for filename in os.listdir(plugins_dir):
            if filename.endswith('.py'):
                module_name = filename[:-3]
                module = importlib.import_module(f'plugins.{module_name}')
                plugin_class = getattr(module, f'{module_name.capitalize()}Plugin')
                self.plugins[module_name] = plugin_class(module_name, self)

    def get_plugin(self, plugin_name: str) -> Plugin:
        return self.plugins.get(plugin_name)

    def get_all_config_ui(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        all_ui = {}
        for plugin_name, plugin in self.plugins.items():
             with st.expander(f"{t('configurations')} {plugin_name}"):
                all_ui[plugin_name] = plugin.get_config_ui(config.get(plugin_name, {}))
        return all_ui

    def get_all_tabs(self) -> List[Dict[str, Any]]:
        all_tabs = []
        for plugin in self.plugins.values():
            all_tabs.extend(plugin.get_tabs())
        return all_tabs

    def run_plugin(self, plugin_name: str, config: Dict[str, Any]):
        if plugin_name in self.plugins:
            self.plugins[plugin_name].run(config)

    def save_config(self, config):
        save_config(config)

def main():
    st.set_page_config(page_title="YoutTools", layout="wide")
    # Initialisation du gestionnaire de plugins
    plugin_manager = PluginManager()
    plugin_manager.load_plugins()

    # Chargement de la configuration
    config = load_config()
    # Initialisation de la langue dans st.session_state
    if 'lang' not in st.session_state:
        st.session_state.lang = config['common']['language']
    st.title(t("page_title"))

    load_dotenv()
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    LLM_KEY = os.getenv("LLM_API_KEY")

    config['api_key'] = API_KEY
    config['llm_key'] = LLM_KEY

    # Création des onglets avec des identifiants uniques
    tabs = [{"id": "configurations", "name": t("configurations")}] + [{"id": tab['plugin'], "name": tab['name']} for tab in plugin_manager.get_all_tabs()]

    # Gestion de la langue
    if 'lang' not in st.session_state:
        st.session_state.lang = "fr"

    new_lang = st.sidebar.selectbox("Choose your language / Choisissez votre langue", options=["en", "fr"], index=["en", "fr"].index(st.session_state.lang), key="lang_selector")

    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    # Ajout des éléments de configuration de la sidebar pour chaque plugin
    for plugin_name, plugin in plugin_manager.plugins.items():
        sidebar_config = plugin.get_sidebar_config_ui(config.get(plugin_name, {}))
        if sidebar_config:
            #st.sidebar.markdown(f"**{plugin_name} Configuration**")
            for key, value in sidebar_config.items():
                config.setdefault(plugin_name, {})[key] = value

    # Gestion de l'onglet sélectionné
    if 'selected_tab_id' not in st.session_state:
        st.session_state.selected_tab_id = "directpublish"

    selected_tab_index = [tab["id"] for tab in tabs].index(st.session_state.selected_tab_id)
    selected_tab = st.sidebar.radio(t("navigation"), [tab["name"] for tab in tabs], index=selected_tab_index, key="tab_selector")

    new_selected_tab_id = next(tab["id"] for tab in tabs if tab["name"] == selected_tab)

    if new_selected_tab_id != st.session_state.selected_tab_id:
        st.session_state.selected_tab_id = new_selected_tab_id
        st.rerun()

    if st.session_state.selected_tab_id == "configurations":
        st.header(t("configurations"))
        all_config_ui = plugin_manager.get_all_config_ui(config)

        for plugin_name, ui_config in all_config_ui.items():
            with st.expander(f"{t('configurations')} {plugin_name}"):
                config[plugin_name] = ui_config

        if st.button(t("save_button")):
            save_config(config)
            st.success(t("success_message"))

    else:
        # Exécution du plugin correspondant à l'onglet sélectionné
        for tab in plugin_manager.get_all_tabs():
            if tab['plugin'] == st.session_state.selected_tab_id:
                plugin_manager.run_plugin(tab['plugin'], config)
                break

if __name__ == "__main__":
    main()
