import os
import json
import importlib
import streamlit as st
from typing import List, Dict, Any, Set
from dotenv import load_dotenv
from global_vars import translations, t

# Constants
CONFIG_FILE = "config.json"
CORE_PLUGINS = {'common', 'ragllm'}  # Add your essential plugins here


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config: Dict[str, Any]):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def set_lang(language):
    st.session_state.lang = language


def t(key: str) -> str:
    return translations[st.session_state.lang].get(key, key)


class Plugin:
    def __init__(self, name, plugin_manager):
        self.name = name
        self.plugin_manager = plugin_manager

    def get_config_fields(self) -> Dict[str, Any]:
        return {}

    def get_config(self, config_value):
        return self.plugin_manager.config[self.name][config_value]

    def get_config_ui(self, config):
        updated_config = {}
        for field, params in self.get_config_fields().items():
            print(params['label'])
            if params['type'] == 'select':
                updated_config[field] = st.selectbox(
                    params['label'],
                    options=[option[0] for option in params['options']],
                    format_func=lambda x: dict(params['options'])[x],
                    index=[option[0] for option in params['options']].index(
                        config.get(field, params['default']))
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
    def __init__(self, config):
        self.plugins: Dict[str, Plugin] = {}
        self.starred_plugins: Set[str] = set()
        self.config = config
        self.available_plugins = self._scan_plugins_directory()

    def _scan_plugins_directory(self) -> Set[str]:
        """Scan the plugins directory and return available plugin names without loading them."""
        plugins_dir = 'plugins'
        return {filename[:-3] for filename in os.listdir(plugins_dir)
                if filename.endswith('.py') and not filename.startswith('__')}

    def load_plugin(self, plugin_name: str) -> None:
        """Load a single plugin by name."""
        if plugin_name not in self.plugins and plugin_name in self.available_plugins:
            module = importlib.import_module(f'plugins.{plugin_name}')
            plugin_class = getattr(module, f'{plugin_name.capitalize()}Plugin')
            self.plugins[plugin_name] = plugin_class(plugin_name, self)

    def load_core_plugins(self):
        """Load only core plugins that are required at startup."""
        for plugin_name in CORE_PLUGINS:
            self.load_plugin(plugin_name)

    def get_plugin(self, plugin_name: str) -> Plugin:
        """Get a plugin, loading it if necessary."""
        if plugin_name not in self.plugins:
            self.load_plugin(plugin_name)
        return self.plugins.get(plugin_name)

    def get_all_config_ui(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Load all plugins when accessing configurations."""
        for plugin_name in self.available_plugins:
            if plugin_name not in self.plugins:
                self.load_plugin(plugin_name)

        all_ui = {}
        for plugin_name, plugin in sorted(self.plugins.items()):
            with st.expander(f"{'⭐ ' if plugin_name in self.starred_plugins else ''}{t('configurations')} {plugin_name}"):
                all_ui[plugin_name] = plugin.get_config_ui(
                    config.get(plugin_name, {}))
                if st.button(f"{'Unstar' if plugin_name in self.starred_plugins else 'Star'} {plugin_name}"):
                    if plugin_name in self.starred_plugins:
                        self.starred_plugins.remove(plugin_name)
                    else:
                        self.starred_plugins.add(plugin_name)
                    self.save_starred_plugins(config)
                    st.rerun()
        return all_ui

    def get_sidebar_config_for_core_plugins(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get sidebar configuration for core plugins only."""
        sidebar_configs = {}
        for plugin_name in CORE_PLUGINS:
            plugin = self.get_plugin(plugin_name)
            if plugin:
                sidebar_config = plugin.get_sidebar_config_ui(
                    config.get(plugin_name, {}))
                if sidebar_config:
                    sidebar_configs[plugin_name] = sidebar_config
        return sidebar_configs

    def get_all_tabs(self) -> List[Dict[str, Any]]:
        """Get all available tabs without loading the plugins."""
        all_tabs = []
        for plugin_name in self.available_plugins:
            # Create a placeholder tab info that will be replaced when the plugin is loaded
            tab = {
                'plugin': plugin_name,
                'name': plugin_name.capitalize(),
                'id': plugin_name,
                'starred': plugin_name in self.starred_plugins
            }
            all_tabs.append(tab)

        # For loaded plugins, get the actual tab information
        for plugin_name, plugin in self.plugins.items():
            tabs = plugin.get_tabs()
            for tab in tabs:
                tab['id'] = plugin_name
                tab['starred'] = plugin_name in self.starred_plugins
            # Replace the placeholder with actual tabs
            all_tabs = [t for t in all_tabs if t['plugin']
                        != plugin_name] + tabs

        return all_tabs

    def load_starred_plugins(self, config: Dict[str, Any]):
        self.starred_plugins = set(config.get('starred_plugins', []))

    def save_starred_plugins(self, config: Dict[str, Any]):
        config['starred_plugins'] = list(self.starred_plugins)
        save_config(config)

    def save_config(self, config: Dict[str, Any]):
        save_config(config)

    def run_plugin(self, plugin_name: str, config: Dict[str, Any]):
        """Run a plugin, loading it if necessary."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            plugin.run(config)


def main():
    st.set_page_config(page_title="YoutTools", layout="wide")

    # Load configuration
    config = load_config()

    # Initialize plugin manager and load core plugins only
    plugin_manager = PluginManager(config)
    plugin_manager.load_core_plugins()
    plugin_manager.load_starred_plugins(config)

    # Initialize language
    if 'lang' not in st.session_state:
        st.session_state.lang = config['common']['language']
    st.title(t("page_title"))

    # Load environment variables
    load_dotenv()
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    LLM_KEY = os.getenv("LLM_API_KEY")
    config['api_key'] = API_KEY
    config['llm_key'] = LLM_KEY

    # Create tabs
    tabs = [{"id": "configurations", "name": t(
        "configurations")}] + plugin_manager.get_all_tabs()

    # Language selection
    new_lang = st.sidebar.selectbox(
        "Choose your language / Choisissez votre langue",
        options=["en", "fr"],
        index=["en", "fr"].index(st.session_state.lang),
        key="lang_selector"
    )

    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    # Handle core plugins sidebar configuration
    core_sidebar_configs = plugin_manager.get_sidebar_config_for_core_plugins(
        config)
    for plugin_name, sidebar_config in core_sidebar_configs.items():
        for key, value in sidebar_config.items():
            config.setdefault(plugin_name, {})[key] = value

    # Ajouter le bouton "Clean session" dans la barre latérale
    if st.sidebar.button(t("Clean session")):
        st.session_state.clear()  # Cela réinitialise st.session_state
        st.rerun()  # Relancer l'application pour refléter les changements

    # Initialize selected tab
    if 'selected_tab_id' not in st.session_state:
        st.session_state.selected_tab_id = "directpublish"

    # Sort and display tabs
    sorted_tabs = sorted(tabs, key=lambda x: (
        not x.get('starred', False), x['name']))
    tab_names = [
        f"{'⭐ ' if tab.get('starred', False) else ''}{tab['name']}" for tab in sorted_tabs]

    selected_tab_index = [tab["id"] for tab in sorted_tabs].index(
        st.session_state.selected_tab_id)
    selected_tab = st.sidebar.radio(
        t("navigation"), tab_names, index=selected_tab_index, key="tab_selector")

    new_selected_tab_id = next(
        tab["id"] for tab in sorted_tabs if f"{'⭐ ' if tab.get('starred', False) else ''}{tab['name']}" == selected_tab)

    if new_selected_tab_id != st.session_state.selected_tab_id:
        st.session_state.selected_tab_id = new_selected_tab_id
        st.rerun()

    # Handle selected tab
    if st.session_state.selected_tab_id == "configurations":
        st.header(t("configurations"))
        all_config_ui = plugin_manager.get_all_config_ui(config)

        for plugin_name, ui_config in all_config_ui.items():
            config[plugin_name] = ui_config

        if st.button(t("save_button")):
            save_config(config)
            st.success(t("success_message"))
    else:
        # Load and run only the selected plugin
        plugin_manager.run_plugin(st.session_state.selected_tab_id, config)


if __name__ == "__main__":
    main()
