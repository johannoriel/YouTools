from global_vars import translations, t
from app import Plugin
import streamlit as st
from typing import Dict, Any

# Ajout des traductions spécifiques à ce plugin
translations["en"].update({
    "dualplugin_tab1": "First Plugin Tab",
    "dualplugin_tab2": "Second Plugin Tab",
    "dualplugin_header": "Dual Plugin Viewer",
    "dualplugin_config_plugin1": "First Plugin Selection",
    "dualplugin_config_plugin2": "Second Plugin Selection",
    "dualplugin_default_plugin1": "pexels",
    "dualplugin_default_plugin2": "movied",
})

translations["fr"].update({
    "dualplugin_tab1": "Premier Onglet Plugin",
    "dualplugin_tab2": "Deuxième Onglet Plugin",
    "dualplugin_header": "Visionneur Double Plugin",
    "dualplugin_config_plugin1": "Sélection du Premier Plugin",
    "dualplugin_config_plugin2": "Sélection du Deuxième Plugin",
    "dualplugin_default_plugin1": "pexels",
    "dualplugin_default_plugin2": "movied",
})


class DualpluginPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.plugin_manager = plugin_manager

    def get_config_fields(self):
        """Définit les champs de configuration du plugin"""
        # Récupère la liste des plugins disponibles
        available_plugins = self.plugin_manager.available_plugins

        return {
            "dualplugin_plugin1": {
                "type": "select",
                "label": t("dualplugin_config_plugin1"),
                "options": [(plugin, plugin.capitalize()) for plugin in available_plugins],
                "default": t("dualplugin_default_plugin1")
            },
            "dualplugin_plugin2": {
                "type": "select",
                "label": t("dualplugin_config_plugin2"),
                "options": [(plugin, plugin.capitalize()) for plugin in available_plugins],
                "default": t("dualplugin_default_plugin2")
            }
        }

    def get_tabs(self):
        """Définit les onglets du plugin"""
        return [
            {"name": t("dualplugin_tab1"), "plugin": "dualplugin"},
            {"name": t("dualplugin_tab2"), "plugin": "dualplugin"}
        ]

    def get_sidebar_config_ui(self, expander, config: Dict[str, Any]):
        """Configuration dans la sidebar pour changer les plugins dynamiquement"""
        sidebar_config = {}

        # Récupère la liste des plugins disponibles
        available_plugins = self.plugin_manager.available_plugins
        plugin_options = [(plugin, plugin.capitalize())
                          for plugin in available_plugins]

        # Sélection du premier plugin
        current_plugin1 = config.get(self.name, {}).get(
            "dualplugin_plugin1",
            t("dualplugin_default_plugin1")
        )
        sidebar_config["dualplugin_plugin1"] = expander.selectbox(
            t("dualplugin_config_plugin1"),
            options=[opt[0] for opt in plugin_options],
            format_func=lambda x: dict(plugin_options)[x],
            index=[opt[0] for opt in plugin_options].index(current_plugin1),
            key=f"{self.name}_plugin1_selector"
        )

        # Sélection du deuxième plugin
        current_plugin2 = config.get(self.name, {}).get(
            "dualplugin_plugin2",
            t("dualplugin_default_plugin2")
        )
        sidebar_config["dualplugin_plugin2"] = expander.selectbox(
            t("dualplugin_config_plugin2"),
            options=[opt[0] for opt in plugin_options],
            format_func=lambda x: dict(plugin_options)[x],
            index=[opt[0] for opt in plugin_options].index(current_plugin2),
            key=f"{self.name}_plugin2_selector"
        )

        return sidebar_config

    def run(self, config):
        """Logique principale du plugin"""

        self.get_sidebar_config_ui(st.sidebar, config)
        # st.header(t("dualplugin_header"))

        # Récupération des plugins sélectionnés depuis la config
        plugin1_name = config.get(self.name, {}).get(
            "dualplugin_plugin1",
            t("dualplugin_default_plugin1")
        )
        plugin2_name = config.get(self.name, {}).get(
            "dualplugin_plugin2",
            t("dualplugin_default_plugin2")
        )

        # Création des deux onglets
        tab1, tab2 = st.tabs([t("dualplugin_tab1"), t("dualplugin_tab2")])

        # Onglet 1 : Exécution du premier plugin
        with tab1:
            plugin1 = self.plugin_manager.get_plugin(plugin1_name)
            if plugin1:
                st.subheader(f"Running {plugin1_name.capitalize()}")
                plugin1.run(config)
            else:
                st.error(
                    f"Plugin '{plugin1_name}' not found or failed to load.")

        # Onglet 2 : Exécution du deuxième plugin
        with tab2:
            plugin2 = self.plugin_manager.get_plugin(plugin2_name)
            if plugin2:
                st.subheader(f"Running {plugin2_name.capitalize()}")
                plugin2.run(config)
            else:
                st.error(
                    f"Plugin '{plugin2_name}' not found or failed to load.")


if __name__ == "__main__":
    st.write("DualPlugin standalone test")
