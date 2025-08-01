from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import requests
import markdown
import re
from datetime import datetime, timedelta
import json
from lib.mailinglist import BrevoProvider, MailerLiteProvider

# Translations for the plugin
translations["en"].update({
    "email_campaign_tab": "Email Campaign",
    "email_campaign_header": "Create Email Campaign",
    "platform_select_label": "Email Platform",
    "mailerlite_api_token_label": "MailerLite API Token",
    "brevo_api_key_label": "Brevo API Key",
    "test_email_label": "Test Email Address",
    "campaign_content_label": "Campaign Content (Markdown)",
    "group_select_label": "Select Recipient Group/List",
    "create_button": "Create Campaign",
    "send_button": "Send Existing Campaign",
    "test_button": "Send Test Campaign",
    "processing": "Processing your request...",
    "success": "Campaign created successfully! Campaign ID: {campaign_id}",
    "send_success": "Campaign sent successfully! Campaign ID: {campaign_id}",
    "test_success": "Test email sent successfully!",
    "error": "An error occurred: {error}",
    "no_title": "No title found in Markdown content",
    "no_subject": "No subject found in Markdown content",
    "from_name_label": "From Name",
    "from_email_label": "From Email",
    "platform_mailerlite": "MailerLite",
    "platform_brevo": "Brevo",
    "debug_mode_label": "Enable Debug Messages",
    "select_campaign_label": "Select Campaign to Send"
})

translations["fr"].update({
    "email_campaign_tab": "Campagne Email",
    "email_campaign_header": "Créer une campagne email",
    "platform_select_label": "Plateforme email",
    "mailerlite_api_token_label": "Jeton API MailerLite",
    "brevo_api_key_label": "Clé API Brevo",
    "test_email_label": "Adresse e-mail de test",
    "campaign_content_label": "Contenu de la campagne (Markdown)",
    "group_select_label": "Sélectionner le groupe/liste de destinataires",
    "create_button": "Créer la campagne",
    "send_button": "Envoyer une campagne existante",
    "test_button": "Envoyer une campagne de test",
    "processing": "Traitement de votre demande...",
    "success": "Campagne créée avec succès ! ID de la campagne : {campaign_id}",
    "send_success": "Campagne envoyée avec succès ! ID de la campagne : {campaign_id}",
    "test_success": "E-mail de test envoyé avec succès !",
    "error": "Une erreur s'est produite : {error}",
    "no_title": "Aucun titre trouvé dans le contenu Markdown",
    "no_subject": "Aucun sujet trouvé dans le contenu Markdown",
    "from_name_label": "Nom de l'expéditeur",
    "from_email_label": "Email de l'expéditeur",
    "platform_mailerlite": "MailerLite",
    "platform_brevo": "Brevo",
    "debug_mode_label": "Activer les messages de débogage",
    "select_campaign_label": "Sélectionner une campagne à envoyer"
})

class CampaignPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)

    def get_config_fields(self):
        """Define plugin configuration fields."""
        return {
            "email_platform": {
                "type": "selectbox",
                "label": t("platform_select_label"),
                "options": ["mailerlite", "brevo"],
                "format_func": lambda x: t(f"platform_{x}"),
                "default": "mailerlite"
            },
            "mailerlite_api_token": {
                "type": "text",
                "label": t("mailerlite_api_token_label"),
                "default": ""
            },
            "brevo_api_key": {
                "type": "text",
                "label": t("brevo_api_key_label"),
                "default": ""
            },
            "test_email": {
                "type": "text",
                "label": t("test_email_label"),
                "default": ""
            },
            "from_name": {
                "type": "text",
                "label": t("from_name_label"),
                "default": "Your Name"
            },
            "from_email": {
                "type": "text",
                "label": t("from_email_label"),
                "default": "your_email@example.com"
            },
            "debug_mode": {
                "type": "checkbox",
                "label": t("debug_mode_label"),
                "default": False
            }
        }

    def get_tabs(self):
        """Define plugin tabs in the interface."""
        return [{"name": t("email_campaign_tab"), "plugin": "emailcampaignplugin"}]

    def run(self, config):
        """Main plugin logic."""
        st.header(t("email_campaign_header"))

        # Get configuration values
        plugin_config = config.get(self.name, {})
        config_platform = plugin_config.get("email_platform", "mailerlite")
        test_email = plugin_config.get("test_email", "")
        config_from_name = plugin_config.get("from_name", "Your Name")
        config_from_email = plugin_config.get("from_email", "your_email@example.com")
        debug_mode = plugin_config.get("debug_mode", False)

        # Debug mode checkbox
        st.subheader("⚙️ Configuration")
        debug_mode = st.checkbox(t("debug_mode_label"), value=debug_mode)

        # Platform selection in UI (overrides config)
        st.subheader("🔧 Platform Selection")
        platform = st.selectbox(
            t("platform_select_label"),
            options=["mailerlite", "brevo"],
            format_func=lambda x: t(f"platform_{x}"),
            index=0 if config_platform == "mailerlite" else 1
        )

        # Sender configuration with override capability
        st.subheader("👤 Sender Configuration")
        col1, col2 = st.columns(2)
        with col1:
            from_name = st.text_input(
                t("from_name_label"),
                value=config_from_name,
                help="Current config value: " + config_from_name
            )
        with col2:
            from_email = st.text_input(
                t("from_email_label"),
                value=config_from_email,
                help="Current config value: " + config_from_email
            )

        # Show current sender info
        if debug_mode:
            st.info(f"📧 **Current Sender:** {from_name} <{from_email}>")

        # Initialize the appropriate provider
        provider = None
        st.subheader("🔗 API Connection")

        if platform == "mailerlite":
            api_token = plugin_config.get("mailerlite_api_token", "")
            if api_token:
                if debug_mode:
                    st.write(f"✅ **MailerLite API Token:** `{api_token[:10]}...{api_token[-4:]}`")
                provider = MailerLiteProvider(api_token, debug=debug_mode)
            else:
                st.error("❌ **MailerLite API Token not configured**")
        elif platform == "brevo":
            api_key = plugin_config.get("brevo_api_key", "")
            if api_key:
                if debug_mode:
                    st.write(f"✅ **Brevo API Key:** `{api_key[:10]}...{api_key[-4:]}`")
                provider = BrevoProvider(api_key, debug=debug_mode)
            else:
                st.error("❌ **Brevo API Key not configured**")

        if not provider:
            st.error(t("error").format(error=f"Please configure API credentials for {platform}"))
            return

        if debug_mode:
            st.success(f"🚀 **Active Platform:** {t(f'platform_{platform}')}")

        # Input for campaign content (Markdown)
        st.subheader("📝 Campaign Content")
        campaign_content = st.text_area(
            t("campaign_content_label"),
            height=300,
            value="# Campaign Title\n\n## Subject: My Campaign Subject\n\nYour campaign content here..."
        )

        # Extract title and subject from Markdown
        title = self.extract_title(campaign_content)
        subject = self.extract_subject(campaign_content)

        # Display extracted values
        st.subheader("📋 Extracted Information")
        col1, col2 = st.columns(2)
        with col1:
            if title:
                st.success(f"**📌 Campaign Title:** {title}")
            else:
                st.warning("⚠️ " + t("no_title"))
        with col2:
            if subject:
                st.success(f"**📧 Subject:** {subject}")
            else:
                st.warning("⚠️ " + t("no_subject"))

        # Fetch groups/lists from the selected provider
        st.subheader("👥 Recipient Groups/Lists")
        with st.spinner("🔄 Fetching groups/lists..."):
            if debug_mode:
                st.write(f"🔍 **Fetching groups from {platform}...**")
            groups = provider.get_groups()
            if debug_mode:
                st.write(f"📊 **Found {len(groups)} groups/lists**")

        if groups:
            for group in groups:
                if debug_mode:
                    st.write(f"  • **{group['name']}** (ID: {group['id']})")

        group_options = [(group["name"], group["id"]) for group in groups] if groups else []
        selected_group = st.selectbox(
            t("group_select_label"),
            options=group_options,
            format_func=lambda x: x[0]
        ) if group_options else None

        if selected_group and debug_mode:
            st.info(f"🎯 **Selected Group:** {selected_group[0]} (ID: {selected_group[1]})")

        # Fetch existing campaigns
        st.subheader("📚 Existing Campaigns")
        with st.spinner("🔄 Fetching existing campaigns..."):
            if debug_mode:
                st.write(f"🔍 **Fetching campaigns from {platform}...**")
            campaigns = provider.get_campaigns()
            if debug_mode:
                st.write(f"📊 **Found {len(campaigns)} campaigns**")

        campaign_options = [(campaign["name"], campaign["id"]) for campaign in campaigns] if campaigns else []
        selected_campaign = st.selectbox(
            t("select_campaign_label"),
            options=campaign_options,
            format_func=lambda x: x[0]
        ) if campaign_options else None

        if selected_campaign and debug_mode:
            st.info(f"🎯 **Selected Campaign:** {selected_campaign[0]} (ID: {selected_campaign[1]})")

        # Test email configuration
        st.subheader("🧪 Test Email Configuration")
        test_email_input = st.text_input(
            t("test_email_label"),
            value=test_email,
            help="Current config value: " + test_email
        )

        # Create three columns for buttons
        st.subheader("🚀 Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Button to create campaign
            if st.button(t("create_button"), type="primary"):
                if debug_mode:
                    st.write("🚀 **Starting campaign creation process...**")

                if not title or not subject or not selected_group:
                    st.error("❌ **Validation Error:** Missing title, subject, or group selection")
                else:
                    with st.spinner(t("processing")):
                        try:
                            if debug_mode:
                                st.write(f"📝 **Converting Markdown to HTML...**")
                            html_content = markdown.markdown(campaign_content)
                            if debug_mode:
                                st.write(f"✅ **HTML Content Length:** {len(html_content)} characters")
                                st.write(f"📧 **Creating campaign with:**")
                                st.write(f"  • **Platform:** {platform}")
                                st.write(f"  • **Title:** {title}")
                                st.write(f"  • **Subject:** {subject}")
                                st.write(f"  • **From:** {from_name} <{from_email}>")
                                st.write(f"  • **Group ID:** {selected_group[1]}")

                            # Create campaign
                            campaign_id = provider.create_campaign(
                                title, subject, html_content,
                                selected_group[1], from_name, from_email
                            )
                            if debug_mode:
                                st.write(f"✅ **Campaign Created Successfully!**")
                            st.success(t("success").format(campaign_id=campaign_id))
                        except Exception as e:
                            if debug_mode:
                                st.write(f"❌ **Campaign Creation Failed:**")
                                st.write(f"**Error Details:** {str(e)}")
                            st.error(t("error").format(error=str(e)))

        with col2:
            # Button to send existing campaign
            if st.button(t("send_button")):
                if debug_mode:
                    st.write("🚀 **Starting campaign sending process...**")

                if not selected_campaign:
                    st.error("❌ **Validation Error:** No campaign selected")
                else:
                    with st.spinner(t("processing")):
                        try:
                            if debug_mode:
                                st.write(f"📧 **Sending campaign with:**")
                                st.write(f"  • **Platform:** {platform}")
                                st.write(f"  • **Campaign ID:** {selected_campaign[1]}")

                            # Send existing campaign
                            provider.send_campaign(selected_campaign[1])
                            if debug_mode:
                                st.write(f"✅ **Campaign Sent Successfully!**")
                            st.success(t("send_success").format(campaign_id=selected_campaign[1]))
                        except Exception as e:
                            if debug_mode:
                                st.write(f"❌ **Campaign Sending Failed:**")
                                st.write(f"**Error Details:** {str(e)}")
                            st.error(t("error").format(error=str(e)))

        with col3:
            # Button to send test campaign
            if st.button(t("test_button")):
                if debug_mode:
                    st.write("🧪 **Starting test email process...**")

                if not test_email_input or not title or not subject:
                    st.error("❌ **Validation Error:** Missing test email, title, or subject")
                else:
                    with st.spinner(t("processing")):
                        try:
                            if debug_mode:
                                st.write(f"📝 **Converting Markdown to HTML...**")
                            html_content = markdown.markdown(campaign_content)
                            if debug_mode:
                                st.write(f"✅ **HTML Content Length:** {len(html_content)} characters")
                                st.write(f"📧 **Sending test email with:**")
                                st.write(f"  • **Platform:** {platform}")
                                st.write(f"  • **Title:** {title}")
                                st.write(f"  • **Subject:** {subject}")
                                st.write(f"  • **From:** {from_name} <{from_email}>")
                                st.write(f"  • **Test Email:** {test_email_input}")

                            # Send test email
                            test_id = provider.send_test_campaign(
                                title, subject, html_content,
                                test_email_input, from_name, from_email
                            )
                            if debug_mode:
                                st.write(f"✅ **Test Email Process Completed!**")
                                st.write(f"**Test ID/Message ID:** {test_id}")
                            st.success(t("test_success"))
                        except Exception as e:
                            if debug_mode:
                                st.write(f"❌ **Test Email Failed:**")
                                st.write(f"**Error Details:** {str(e)}")
                            st.error(t("error").format(error=str(e)))

    def extract_title(self, markdown_content: str) -> str:
        """Extract the first h1 heading as the campaign title."""
        match = re.search(r'^# (.+)$', markdown_content, re.MULTILINE)
        return match.group(1) if match else ""

    def extract_subject(self, markdown_content: str) -> str:
        """Extract the subject line from Markdown (e.g., ## Subject: ...)."""
        match = re.search(r'^## Subject: (.+)$', markdown_content, re.MULTILINE)
        return match.group(1) if match else ""

if __name__ == "__main__":
    st.write("Email Campaign Plugin standalone test")
