from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
from widgets.recentvideos import RecentVideosWidget

translations["en"].update({
    "recent_videos_tab": "Recent YouTube Videos",
})

translations["fr"].update({
    "recent_videos_tab": "Vidéos récentes",
})


class RecentvideosPlugin(Plugin):
    def get_tabs(self):
        return [{"name": t("recent_videos_tab"), "plugin": "recentvideos"}]

    def run(self, config):
        widget = RecentVideosWidget("recentvideos", "rvw", self.plugin_manager)
        widget.display(config)
