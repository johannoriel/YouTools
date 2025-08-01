from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import requests
import markdown
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import json

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
    "create_button": "Create and Send Campaign",
    "test_button": "Send Test Campaign",
    "processing": "Processing your request...",
    "success": "Campaign created successfully! Campaign ID: {campaign_id}",
    "test_success": "Test email sent successfully!",
    "error": "An error occurred: {error}",
    "no_title": "No title found in Markdown content",
    "no_subject": "No subject found in Markdown content",
    "from_name_label": "From Name",
    "from_email_label": "From Email",
    "platform_mailerlite": "MailerLite",
    "platform_brevo": "Brevo",
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
    "create_button": "Créer et envoyer la campagne",
    "test_button": "Envoyer une campagne de test",
    "processing": "Traitement de votre demande...",
    "success": "Campagne créée avec succès ! ID de la campagne : {campaign_id}",
    "test_success": "E-mail de test envoyé avec succès !",
    "error": "Une erreur s'est produite : {error}",
    "no_title": "Aucun titre trouvé dans le contenu Markdown",
    "no_subject": "Aucun sujet trouvé dans le contenu Markdown",
    "from_name_label": "Nom de l'expéditeur",
    "from_email_label": "Email de l'expéditeur",
    "platform_mailerlite": "MailerLite",
    "platform_brevo": "Brevo",
})

class EmailProvider(ABC):
    """Abstract base class for email campaign providers."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def get_groups(self) -> list:
        """Fetch recipient groups/lists."""
        pass

    @abstractmethod
    def create_and_send_campaign(self, title: str, subject: str, html_content: str,
                                group_id: str, from_name: str, from_email: str) -> str:
        """Create and send a campaign."""
        pass

    @abstractmethod
    def send_test_campaign(self, title: str, subject: str, html_content: str,
                          test_email: str, from_name: str, from_email: str) -> str:
        """Send a test campaign."""
        pass

class MailerLiteProvider(EmailProvider):
    """MailerLite email campaign provider."""

    def __init__(self, api_token: str):
        super().__init__(api_token)
        self.base_url = "https://connect.mailerlite.com/api"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def get_groups(self) -> list:
        """Fetch subscriber groups from MailerLite API."""
        try:
            response = requests.get(f"{self.base_url}/groups", headers=self.headers)
            response.raise_for_status()
            groups = response.json().get("data", [])
            return [{"id": group["id"], "name": group["name"]} for group in groups]
        except Exception as e:
            st.error(t("error").format(error=f"Failed to fetch groups: {str(e)}"))
            return []

    def create_and_send_campaign(self, title: str, subject: str, html_content: str,
                                group_id: str, from_name: str, from_email: str) -> str:
        """Create and send a MailerLite campaign."""
        try:
            # Create campaign
            campaign_payload = {
                "name": title,
                "type": "regular",
                "emails": [{
                    "subject": subject,
                    "from_name": from_name,
                    "from": from_email,
                    "content": html_content
                }],
                "filter": [
                    [
                        {
                            "operator": "in_any",
                            "args": [
                                "groups",
                                [group_id]
                            ]
                        }
                    ]
                ]
            }

            response = requests.post(f"{self.base_url}/campaigns",
                                   json=campaign_payload, headers=self.headers)
            response.raise_for_status()
            campaign_id = response.json()["data"]["id"]

            # Send campaign
            send_payload = {"delivery": "instant"}
            send_response = requests.post(f"{self.base_url}/campaigns/{campaign_id}/schedule",
                                        json=send_payload, headers=self.headers)
            send_response.raise_for_status()

            return campaign_id

        except requests.exceptions.HTTPError as e:
            error_detail = f"MailerLite API Error - HTTP {response.status_code}: {response.text}"
            raise Exception(error_detail)

    def send_test_campaign(self, title: str, subject: str, html_content: str,
                          test_email: str, from_name: str, from_email: str) -> str:
        """Send a test MailerLite campaign."""
        try:
            # Create test campaign without group filter
            campaign_payload = {
                "name": f"TEST - {title}",
                "type": "regular",
                "emails": [{
                    "subject": f"[TEST] {subject}",
                    "from_name": from_name,
                    "from": from_email,
                    "content": html_content
                }]
                # Note: Pour un vrai test, il faudrait que l'email soit dans un groupe
            }

            response = requests.post(f"{self.base_url}/campaigns",
                                   json=campaign_payload, headers=self.headers)
            response.raise_for_status()
            campaign_id = response.json()["data"]["id"]

            return campaign_id

        except requests.exceptions.HTTPError as e:
            error_detail = f"MailerLite Test Error - HTTP {response.status_code}: {response.text}"
            raise Exception(error_detail)

class BrevoProvider(EmailProvider):
    """Brevo (Sendinblue) email campaign provider."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.brevo.com/v3"
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def get_groups(self) -> list:
        """Fetch contact lists from Brevo API."""
        try:
            response = requests.get(f"{self.base_url}/contacts/lists", headers=self.headers)
            response.raise_for_status()
            lists = response.json().get("lists", [])
            return [{"id": str(lst["id"]), "name": lst["name"]} for lst in lists]
        except Exception as e:
            st.error(t("error").format(error=f"Failed to fetch lists: {str(e)}"))
            return []

    def create_and_send_campaign(self, title: str, subject: str, html_content: str,
                                group_id: str, from_name: str, from_email: str) -> str:
        """Create and send a Brevo campaign."""
        try:
            # Schedule sending for immediate delivery
            scheduled_at = (datetime.now() + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")

            campaign_payload = {
                "name": title,
                "subject": subject,
                "sender": {
                    "name": from_name,
                    "email": from_email
                },
                "type": "classic",
                "htmlContent": html_content,
                "recipients": {
                    "listIds": [int(group_id)]
                },
                "scheduledAt": scheduled_at
            }

            response = requests.post(f"{self.base_url}/emailCampaigns",
                                   json=campaign_payload, headers=self.headers)
            response.raise_for_status()
            campaign_id = response.json()["id"]

            return str(campaign_id)

        except requests.exceptions.HTTPError as e:
            error_detail = f"Brevo API Error - HTTP {response.status_code}: {response.text}"
            raise Exception(error_detail)

    def send_test_campaign(self, title: str, subject: str, html_content: str,
                          test_email: str, from_name: str, from_email: str) -> str:
        """Send a test email via Brevo."""
        try:
            # Use Brevo's transactional email API for test
            test_payload = {
                "sender": {
                    "name": from_name,
                    "email": from_email
                },
                "to": [
                    {
                        "email": test_email
                    }
                ],
                "subject": f"[TEST] {subject}",
                "htmlContent": html_content
            }

            response = requests.post(f"{self.base_url}/smtp/email",
                                   json=test_payload, headers=self.headers)
            response.raise_for_status()
            message_id = response.json().get("messageId", "test-sent")

            return str(message_id)

        except requests.exceptions.HTTPError as e:
            error_detail = f"Brevo Test Error - HTTP {response.status_code}: {response.text}"
            raise Exception(error_detail)

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
        platform = plugin_config.get("email_platform", "mailerlite")
        test_email = plugin_config.get("test_email", "")
        from_name = plugin_config.get("from_name", "Your Name")
        from_email = plugin_config.get("from_email", "your_email@example.com")

        # Initialize the appropriate provider
        provider = None
        if platform == "mailerlite":
            api_token = plugin_config.get("mailerlite_api_token", "")
            if api_token:
                provider = MailerLiteProvider(api_token)
        elif platform == "brevo":
            api_key = plugin_config.get("brevo_api_key", "")
            if api_key:
                provider = BrevoProvider(api_key)

        if not provider:
            st.error(t("error").format(error=f"Please configure API credentials for {platform}"))
            return

        # Display current platform
        st.info(f"Current platform: **{t(f'platform_{platform}')}**")

        # Input for campaign content (Markdown)
        campaign_content = st.text_area(
            t("campaign_content_label"),
            height=300,
            value="# Campaign Title\n\n## Subject: My Campaign Subject\n\nYour campaign content here..."
        )

        # Extract title and subject from Markdown
        title = self.extract_title(campaign_content)
        subject = self.extract_subject(campaign_content)

        # Display extracted values
        if title:
            st.write(f"**Extracted Campaign Title:** {title}")
        else:
            st.warning(t("no_title"))
        if subject:
            st.write(f"**Extracted Subject:** {subject}")
        else:
            st.warning(t("no_subject"))

        # Fetch groups/lists from the selected provider
        groups = provider.get_groups()
        group_options = [(group["name"], group["id"]) for group in groups] if groups else []
        selected_group = st.selectbox(
            t("group_select_label"),
            options=group_options,
            format_func=lambda x: x[0]
        ) if group_options else None

        # Create two columns for buttons
        col1, col2 = st.columns(2)

        with col1:
            # Button to create and send campaign
            if st.button(t("create_button")):
                if not title or not subject or not selected_group:
                    st.error(t("error").format(error="Missing title, subject, or group selection"))
                else:
                    with st.spinner(t("processing")):
                        try:
                            # Convert Markdown to HTML
                            html_content = markdown.markdown(campaign_content)
                            # Create and send campaign
                            campaign_id = provider.create_and_send_campaign(
                                title, subject, html_content,
                                selected_group[1], from_name, from_email
                            )
                            st.success(t("success").format(campaign_id=campaign_id))
                        except Exception as e:
                            st.error(t("error").format(error=str(e)))

        with col2:
            # Button to send test campaign
            if st.button(t("test_button")):
                if not test_email or not title or not subject:
                    st.error(t("error").format(error="Missing test email, title, or subject"))
                else:
                    with st.spinner(t("processing")):
                        try:
                            # Convert Markdown to HTML
                            html_content = markdown.markdown(campaign_content)
                            # Send test email
                            test_id = provider.send_test_campaign(
                                title, subject, html_content,
                                test_email, from_name, from_email
                            )
                            st.success(t("test_success"))
                        except Exception as e:
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
