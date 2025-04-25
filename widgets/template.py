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

"""
A widget is a reusable component that can be used to display information or interact with the user.
It can be used to create a user interface for a specific task or feature.
BUT to be compatible with streamlit, all interactive elements like checkbox, radio button, selectbox, etc. must have a unique key, to avoid conflicts.
so the key must use the widget self.prefix for that.
A widget can use internally st.session_state, but it cannot use st.session_state of other widgets or plugins.
To exchange data it must use files stored in the work directory : self.work_dir()
"""


class TemplateWidget(Widget):
    def __init__(self, name, prefix, plugin_manager):
        super().__init__(name, prefix, plugin_manager)

    def display(self):
        st.title(t("template_string1"))
        work_directory = self.work_dir()
        result = self.process_llm("My prompt : ask something to an llm")
        st.input(t("template_string2"), key=f"{self.prefix}_input")


""" Widget integration in plugin example"""


class ExamplePlugin(Plugin):
    def templatewidget_process(self, config):
        from widgets.template_widget import TemplateWidget
        TemplateWidget("pluginname", "widgetprefix",
                       plugin_manager=self.plugin_manager).display()

    def run(self, config):
        tab1, tab2 = st.tabs(["tab1", "idget tab"])
        with tab1:
            pass
        with tab2:
            self.templatewidget_process(config)
