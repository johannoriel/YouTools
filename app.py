import os
import json
import importlib
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv


# Constantes
CONFIG_FILE = "config.json"
    
class Plugin:
    def __init__(self, name, plugin_manager):
        self.name = name
        self.plugin_manager = plugin_manager

    def get_config_fields(self) -> Dict[str, Any]:
        """Retourne les champs de configuration du plugin."""
        return {}

    def get_config_ui(self, config):
        """Génère une interface par défaut. Si le champ commance par 'pass' il sera considéré password" """
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
        """Retourne les onglets du plugin."""
        return []

    def run(self, config: Dict[str, Any]):
        """Exécute les fonctionnalités principales du plugin."""
        pass

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}

    def load_plugins(self):
        """Charge tous les plugins du dossier 'plugins'."""
        plugins_dir = 'plugins'
        for filename in os.listdir(plugins_dir):
            if filename.endswith('.py'):
                module_name = filename[:-3]
                module = importlib.import_module(f'plugins.{module_name}')
                plugin_class = getattr(module, f'{module_name.capitalize()}Plugin')
                self.plugins[module_name] = plugin_class(module_name, self)

    def get_plugin(self, plugin_name: str) -> Plugin:
        """Récupère un plugin spécifique."""
        return self.plugins.get(plugin_name)

    def get_all_config_ui(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Récupère toutes les interfaces utilisateur de configuration de tous les plugins."""
        all_ui = {}
        for plugin_name, plugin in self.plugins.items():
             with st.expander(f"Configuration {plugin_name}"):
                all_ui[plugin_name] = plugin.get_config_ui(config.get(plugin_name, {}))
        return all_ui

    def get_all_tabs(self) -> List[Dict[str, Any]]:
        """Récupère tous les onglets de tous les plugins."""
        all_tabs = []
        for plugin in self.plugins.values():
            all_tabs.extend(plugin.get_tabs())
        return all_tabs

    def run_plugin(self, plugin_name: str, config: Dict[str, Any]):
        """Exécute un plugin spécifique."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].run(config)

def load_config() -> Dict[str, Any]:
    """Charge la configuration depuis le fichier JSON."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_config(config: Dict[str, Any]):
    """Sauvegarde la configuration dans le fichier JSON."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
        
def list_video_files(directory):
    video_files = []
    outfile_videos = []
    chroma_videos = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.mkv', '.mp4')):
            full_path = os.path.join(directory, file)
            mod_time = os.path.getmtime(full_path)
            if file.startswith('outfile_'):
                outfile_videos.append((file, full_path, mod_time))
            elif file.startswith('chroma_'):
                chroma_videos.append((file, full_path, mod_time))
            else:
                video_files.append((file, full_path, mod_time))
    
    # Trier par date de modification, la plus récente en premier
    video_files.sort(key=lambda x: x[2], reverse=True)
    outfile_videos.sort(key=lambda x: x[2], reverse=True)
    chroma_videos.sort(key=lambda x: x[2], reverse=True)
    return video_files, outfile_videos, chroma_videos

    
def main():
    st.set_page_config(page_title="YoutTools", layout="wide")
    st.title("YoutTools : YouTube Channel helpers")

    # Initialisation du gestionnaire de plugins
    plugin_manager = PluginManager()
    plugin_manager.load_plugins()

    # Chargement de la configuration
    config = load_config()

    load_dotenv()
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    LLM_KEY = os.getenv("LLM_API_KEY")

    config['api_key'] = API_KEY
    config['llm_key'] = LLM_KEY

    # Création des onglets
    tabs = ["Configuration"] + [tab['name'] for tab in plugin_manager.get_all_tabs()]
    selected_tab = st.sidebar.radio("Navigation", tabs)

    if selected_tab == "Configuration":
        st.header("Configurations")
        all_config_ui = plugin_manager.get_all_config_ui(config)

        if st.button("Sauvegarder la configuration"):
            save_config(config)
            st.success("Configuration sauvegardée!")

    else:
        # Exécution du plugin correspondant à l'onglet sélectionné
        for tab in plugin_manager.get_all_tabs():
            if tab['name'] == selected_tab:
                plugin_manager.run_plugin(tab['plugin'], config)
                break

if __name__ == "__main__":
    main()
