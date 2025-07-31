from lib.global_vars import translations, t
from app import Plugin
import streamlit as st
import requests
import markdown
import re

# Translations for the plugin
translations["en"].update({
    "mailerlite_tab": "MailerLite Campaign",
    "mailerlite_header": "Create MailerLite Campaign",
    "mailerlite_api_token_label": "MailerLite API Token",
    "mailerlite_test_email_label": "Test Email Address",
    "mailerlite_campaign_content_label": "Campaign Content (Markdown)",
    "mailerlite_group_select_label": "Select Subscriber Group",
    "mailerlite_create_button": "Create and Send Campaign",
    "mailerlite_test_button": "Send Test Campaign",
    "mailerlite_processing": "Processing your request...",
    "mailerlite_success": "Campaign created successfully! Campaign ID: {campaign_id}",
    "mailerlite_test_success": "Test email sent successfully!",
    "mailerlite_error": "An error occurred: {error}",
    "mailerlite_no_title": "No title found in Markdown content",
    "mailerlite_no_subject": "No subject found in Markdown content",
})

translations["fr"].update({
    "mailerlite_tab": "Campagne MailerLite",
    "mailerlite_header": "Créer une campagne MailerLite",
    "mailerlite_api_token_label": "Jeton API MailerLite",
    "mailerlite_test_email_label": "Adresse e-mail de test",
    "mailerlite_campaign_content_label": "Contenu de la campagne (Markdown)",
    "mailerlite_group_select_label": "Sélectionner le groupe d'abonnés",
    "mailerlite_create_button": "Créer et envoyer la campagne",
    "mailerlite_test_button": "Envoyer une campagne de test",
    "mailerlite_processing": "Traitement de votre demande...",
    "mailerlite_success": "Campagne créée avec succès ! ID de la campagne : {campaign_id}",
    "mailerlite_test_success": "E-mail de test envoyé avec succès !",
    "mailerlite_error": "Une erreur s'est produite : {error}",
    "mailerlite_no_title": "Aucun titre trouvé dans le contenu Markdown",
    "mailerlite_no_subject": "Aucun sujet trouvé dans le contenu Markdown",
})

class MailerlitePlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)

    def get_config_fields(self):
        """Define plugin configuration fields."""
        return {
            "mailerlite_api_token": {
                "type": "text",
                "label": t("mailerlite_api_token_label"),
                "default": ""
            },
            "mailerlite_test_email": {
                "type": "text",
                "label": t("mailerlite_test_email_label"),
                "default": ""
            }
        }

    def get_tabs(self):
        """Define plugin tabs in the interface."""
        return [{"name": t("mailerlite_tab"), "plugin": "mailerlitecampaignplugin"}]

    def run(self, config):
        """Main plugin logic."""
        st.header(t("mailerlite_header"))

        # Get API token from config
        api_token = config.get(self.name, {}).get("mailerlite_api_token", "")
        test_email = config.get(self.name, {}).get("mailerlite_test_email", "")

        # Input for campaign content (Markdown)
        campaign_content = st.text_area(
            t("mailerlite_campaign_content_label"),
            height=300,
            value="# Campaign Title\n\n## Subject: My Campaign Subject\n\nYour campaign content here..."
        )

        # Extract title and subject from Markdown
        title = self.extract_title(campaign_content)
        subject = self.extract_subject(campaign_content)

        # Display extracted values
        if title:
            st.write(f"Extracted Campaign Title: {title}")
        else:
            st.warning(t("mailerlite_no_title"))
        if subject:
            st.write(f"Extracted Subject: {subject}")
        else:
            st.warning(t("mailerlite_no_subject"))

        # Fetch groups from MailerLite API
        groups = self.get_groups(api_token)
        group_options = [(group["name"], group["id"]) for group in groups] if groups else []
        selected_group = st.selectbox(
            t("mailerlite_group_select_label"),
            options=group_options,
            format_func=lambda x: x[0]
        )

        # Create two columns for buttons
        col1, col2 = st.columns(2)

        with col1:
            # Button to create and send campaign
            if st.button(t("mailerlite_create_button")):
                if not api_token or not title or not subject or not selected_group:
                    st.error(t("mailerlite_error").format(error="Missing API token, title, subject, or group selection"))
                else:
                    with st.spinner(t("mailerlite_processing")):
                        try:
                            # Convert Markdown to HTML
                            html_content = markdown.markdown(campaign_content)
                            # Create campaign
                            campaign_id = self.create_campaign(api_token, title, subject, selected_group[1], t("language"))
                            # Upload content
                            self.upload_campaign_content(api_token, campaign_id, html_content)
                            # Send campaign
                            self.send_campaign(api_token, campaign_id)
                            st.success(t("mailerlite_success").format(campaign_id=campaign_id))
                        except Exception as e:
                            st.error(t("mailerlite_error").format(error=str(e)))

        with col2:
            # Button to send test campaign
            if st.button(t("mailerlite_test_button")):
                if not api_token or not test_email or not title or not subject:
                    st.error(t("mailerlite_error").format(error="Missing API token, test email, title, or subject"))
                else:
                    with st.spinner(t("mailerlite_processing")):
                        try:
                            # Convert Markdown to HTML
                            html_content = markdown.markdown(campaign_content)
                            # Create test campaign
                            campaign_id = self.create_campaign(api_token, title, subject, None, t("language"))
                            # Upload content
                            self.upload_campaign_content(api_token, campaign_id, html_content)
                            # Send test email
                            self.send_test_email(api_token, campaign_id, test_email)
                            st.success(t("mailerlite_test_success"))
                        except Exception as e:
                            st.error(t("mailerlite_error").format(error=str(e)))

    def extract_title(self, markdown_content: str) -> str:
        """Extract the first h1 heading as the campaign title."""
        match = re.search(r'^# (.+)$', markdown_content, re.MULTILINE)
        return match.group(1) if match else ""

    def extract_subject(self, markdown_content: str) -> str:
        """Extract the subject line from Markdown (e.g., ## Subject: ...)."""
        match = re.search(r'^## Subject: (.+)$', markdown_content, re.MULTILINE)
        return match.group(1) if match else ""

    def get_groups(self, api_token: str) -> list:
        """Fetch subscriber groups from MailerLite API."""
        try:
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            response = requests.get("https://connect.mailerlite.com/api/groups", headers=headers)
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            st.error(t("mailerlite_error").format(error=f"Failed to fetch groups: {str(e)}"))
            return []

    def create_campaign(self, api_token: str, title: str, subject: str, group_id: str, language: str) -> str:
        """Create a campaign in MailerLite."""
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "name": title,
            "type": "regular",
            "emails": [{
                "subject": subject,
                "from_name": "Your Name",  # Default value
                "from": "your_email@example.com"  # Default value
            }],
            "filter": [[{"operator": "in_any", "args": ["groups", [group_id]]}]] if group_id else [],
            "language_id": language
        }
        response = requests.post("https://connect.mailerlite.com/api/campaigns", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["data"]["id"]

    def upload_campaign_content(self, api_token: str, campaign_id: str, html_content: str):
        """Upload HTML content to the campaign."""
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "html": html_content,
            "plain_text": "This is a plain text version of the campaign.\n\nTo unsubscribe, click here: {$unsubscribe}"
        }
        response = requests.put(f"https://connect.mailerlite.com/api/campaigns/{campaign_id}/content", json=payload, headers=headers)
        response.raise_for_status()

    def send_campaign(self, api_token: str, campaign_id: str):
        """Send the campaign."""
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {"schedule": {"delivery": "instant"}}
        response = requests.post(f"https://connect.mailerlite.com/api/campaigns/{campaign_id}/actions/schedule", json=payload, headers=headers)
        response.raise_for_status()

    def send_test_email(self, api_token: str, campaign_id: str, test_email: str):
        """Send a test email for the campaign."""
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {"emails": [test_email]}
        response = requests.post(f"https://connect.mailerlite.com/api/campaigns/{campaign_id}/actions/send-test", json=payload, headers=headers)
        response.raise_for_status()

if __name__ == "__main__":
    st.write("MailerLite Campaign Plugin standalone test")
