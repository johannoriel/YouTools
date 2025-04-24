from lib.global_vars import translations, t
from app import Widget
import streamlit as st
from lib.global_vars import translations, t

translations["en"].update({
    "template_string1": "In english",
})

translations["fr"].update({
    "template_string1": "En français",
})


class TemplateWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def display(self):
        st.title(t("template_string1"))
        work_directory = self.plugin_manager.config["common"]["work_directory"]
        st.input(t("template_string2"), key=f"{self.prefix}_input")
