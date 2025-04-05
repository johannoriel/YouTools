from global_vars import translations, t
from app import Plugin
import streamlit as st
from typing import Dict, Any

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "dualplugin_header": "Multi Plugin Viewer",
    "dualplugin_config_plugins": "List of Plugins with Presets (use # for preset titles)",
    "dualplugin_add_tab": "Add Tab",
    "dualplugin_remove_tab": "Remove Last Tab",
    "dualplugin_select_plugin": "Select Plugin for Tab {index}",
    "dualplugin_select_preset": "Select Preset",
    "dualplugin_default_plugins": "# Video Tools\npexels\nmovied\n# Analytics\nanalytics\nstats",
})

translations["fr"].update({
    "dualplugin_header": "Visionneur Multi-Plugin",
    "dualplugin_config_plugins": "Liste des plugins avec préréglages (utilisez # pour les titres des préréglages)",
    "dualplugin_add_tab": "Ajouter un onglet",
    "dualplugin_remove_tab": "Supprimer le dernier onglet",
    "dualplugin_select_plugin": "Sélectionner un plugin pour l'onglet {index}",
    "dualplugin_select_preset": "Sélectionner un préréglage",
    "dualplugin_default_plugins": "# Outils Vidéo\npexels\nmovied\n# Analytiques\nanalytics\nstats",
})


class DualpluginPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.plugin_manager = plugin_manager

    def get_config_fields(self):
        """Définit les champs de configuration du plugin"""
        return {
            "dualplugin_plugins": {
                "type": "textarea",
                "label": t("dualplugin_config_plugins"),
                "default": t("dualplugin_default_plugins")
            }
        }

    def parse_presets(self, plugin_list_str: str) -> Dict[str, list]:
        """Parse la config pour extraire les presets et leurs plugins"""
        presets = {}
        current_preset = None
        lines = plugin_list_str.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Enlever le # et les espaces
                current_preset = line[1:].strip()
                presets[current_preset] = []
            elif current_preset is not None and line:
                presets[current_preset].append(line)

        # Si pas de preset défini, utiliser une clé par défaut
        if not presets and any(line for line in lines if line and not line.startswith("#")):
            presets["Default"] = [line.strip()
                                  for line in lines if line and not line.startswith("#")]

        return presets

    def get_tabs(self):
        """Définit les onglets du plugin dynamiquement avec le nom des plugins"""
        tab_count = st.session_state.get(f"{self.name}_tab_count", 0)
        selected_plugins = st.session_state.get(
            f"{self.name}_selected_plugins", [])
        tab_names = [
            selected_plugins[i].capitalize() if i < len(
                selected_plugins) else f"Tab {i + 1}"
            for i in range(tab_count)
        ]
        return [{"name": name, "plugin": "dualplugin"} for name in tab_names]

    def get_sidebar_config_ui(self, root, config: Dict[str, Any]):
        """Configuration dans la sidebar pour gérer les onglets, presets et sélectionner les plugins"""
        expander = root.expander("Dual Plugin")
        sidebar_config = {}

        # Récupérer la liste des plugins avec presets depuis la config
        plugin_list_str = config.get(self.name, {}).get(
            "dualplugin_plugins",
            t("dualplugin_default_plugins")
        )
        presets = self.parse_presets(plugin_list_str)
        preset_options = list(presets.keys())

        # Initialiser les plugins sélectionnés et le nombre d'onglets si vide
        if f"{self.name}_selected_plugins" not in st.session_state or not st.session_state[f"{self.name}_selected_plugins"]:
            default_preset = preset_options[0] if preset_options else "Default"
            st.session_state[f"{self.name}_selected_plugins"] = presets.get(
                default_preset, [])
            st.session_state[f"{self.name}_tab_count"] = len(
                presets.get(default_preset, []))

        # Sélection du preset
        selected_preset = expander.selectbox(
            t("dualplugin_select_preset"),
            options=preset_options,
            index=preset_options.index(st.session_state.get(f"{self.name}_current_preset", preset_options[0])) if st.session_state.get(
                f"{self.name}_current_preset") in preset_options else 0,
            key=f"{self.name}_preset_selector"
        )
        if selected_preset != st.session_state.get(f"{self.name}_current_preset"):
            st.session_state[f"{self.name}_selected_plugins"] = presets[selected_preset]
            st.session_state[f"{self.name}_tab_count"] = len(
                presets[selected_preset])
            st.session_state[f"{self.name}_current_preset"] = selected_preset
            st.rerun()

        # Liste des plugins disponibles
        available_plugins = self.plugin_manager.available_plugins
        plugin_options = [(plugin, plugin.capitalize())
                        for plugin in available_plugins]

        # Sélection dynamique des plugins pour chaque onglet
        selected_plugins = st.session_state[f"{self.name}_selected_plugins"]
        for i in range(st.session_state[f"{self.name}_tab_count"]):
            current_plugin = selected_plugins[i] if i < len(selected_plugins) else (
                available_plugins[0] if available_plugins else ""
            )
            selected_plugin = expander.selectbox(
                t("dualplugin_select_plugin").format(index=i + 1),
                options=[opt[0] for opt in plugin_options],
                format_func=lambda x: dict(plugin_options)[x],
                index=[opt[0] for opt in plugin_options].index(
                    current_plugin) if current_plugin in available_plugins else 0,
                key=f"{self.name}_plugin_selector_{i}"
            )
            if i < len(selected_plugins):
                selected_plugins[i] = selected_plugin
            else:
                selected_plugins.append(selected_plugin)

        # Bouton pour ajouter un onglet
        if expander.button(t("dualplugin_add_tab")):
            st.session_state[f"{self.name}_tab_count"] += 1
            st.rerun()

        # Bouton pour supprimer le dernier onglet (si plus d'un onglet)
        if st.session_state[f"{self.name}_tab_count"] > 1:
            if expander.button(t("dualplugin_remove_tab")):
                st.session_state[f"{self.name}_tab_count"] -= 1
                if len(selected_plugins) > st.session_state[f"{self.name}_tab_count"]:
                    selected_plugins.pop()
                st.rerun()

        st.session_state[f"{self.name}_selected_plugins"] = selected_plugins
        return sidebar_config

    def run(self, config):
        """Logique principale du plugin"""
        self.get_sidebar_config_ui(st.sidebar, config)

        # Récupérer la liste des plugins sélectionnés
        selected_plugins = st.session_state[f"{self.name}_selected_plugins"]
        tab_count = st.session_state[f"{self.name}_tab_count"]

        # Créer les onglets avec les noms des plugins
        tab_names = [
            selected_plugins[i].capitalize() if i < len(
                selected_plugins) else f"Tab {i + 1}"
            for i in range(tab_count)
        ]
        tabs = st.tabs(tab_names)

        # Exécuter les plugins dans chaque onglet
        for i, tab in enumerate(tabs):
            with tab:
                if i < len(selected_plugins):
                    plugin_name = selected_plugins[i]
                    plugin = self.plugin_manager.get_plugin(plugin_name)
                    if plugin:
                        plugin.run(config)
                    else:
                        st.error(
                            f"Plugin '{plugin_name}' not found or failed to load.")
                else:
                    st.write("No plugin selected for this tab.")


if __name__ == "__main__":
    st.write("DualPlugin standalone test")
