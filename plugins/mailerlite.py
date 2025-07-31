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
    "mailerlite_from_name_label": "From Name",
    "mailerlite_from_email_label": "From Email",
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
    "mailerlite_from_name_label": "Nom de l'expéditeur",
    "mailerlite_from_email_label": "Email de l'expéditeur",
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
            },
            "mailerlite_from_name": {
                "type": "text",
                "label": t("mailerlite_from_name_label"),
                "default": "Your Name"
            },
            "mailerlite_from_email": {
                "type": "text",
                "label": t("mailerlite_from_email_label"),
                "default": "your_email@example.com"
            }
        }

    def get_tabs(self):
        """Define plugin tabs in the interface."""
        return [{"name": t("mailerlite_tab"), "plugin": "mailerlitecampaignplugin"}]

    def run(self, config):
        """Main plugin logic."""
        st.header(t("mailerlite_header"))

        # Get configuration values
        plugin_config = config.get(self.name, {})
        api_token = plugin_config.get("mailerlite_api_token", "")
        test_email = plugin_config.get("mailerlite_test_email", "")
        from_name = plugin_config.get("mailerlite_from_name", "Your Name")
        from_email = plugin_config.get("mailerlite_from_email", "your_email@example.com")

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
        ) if group_options else None

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
                            campaign_id = self.create_campaign(api_token, title, subject, selected_group[1], from_name, from_email)
                            # Upload content
                            self.upload_campaign_content(api_token, campaign_id, html_content)
                            # Send campaign
                            self.send_campaign(api_token, campaign_id)
                            st.success(t("mailerlite_success").format(campaign_id=campaign_id))
                        except Exception as e:
                            st.error(t("mailerlite_error").format(error=str(e)))
                            # Debug information
                            st.error(f"Debug: API response details might be in the logs")

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
                            # Create test campaign (without group filter)
                            campaign_id = self.create_campaign(api_token, title, subject, None, from_name, from_email)
                            # Upload content
                            self.upload_campaign_content(api_token, campaign_id, html_content)
                            # Send test email
                            self.send_test_email(api_token, campaign_id, test_email)
                            st.success(t("mailerlite_test_success"))
                        except Exception as e:
                            st.error(t("mailerlite_error").format(error=str(e)))
                            # Debug information
                            st.error(f"Debug: API response details might be in the logs")

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

    def create_campaign(self, api_token: str, title: str, subject: str, group_id: str, from_name: str, from_email: str) -> str:
        """Create a campaign in MailerLite."""
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Base payload structure
        payload = {
            "name": title,
            "type": "regular",
            "emails": [{
                "subject": subject,
                "from_name": from_name,
                "from": from_email
            }]
        }

        # Add group filter only if group_id is provided
        if group_id:
            payload["filter"] = {
                "groups": [group_id]
            }

        try:
            print(f"Debug: Creating campaign with payload: {payload}")  # Debug log
            response = requests.post("https://connect.mailerlite.com/api/campaigns", json=payload, headers=headers)

            # Log response details for debugging
            print(f"Debug: Response status: {response.status_code}")
            print(f"Debug: Response body: {response.text}")

            response.raise_for_status()
            return response.json()["data"]["id"]
        except requests.exceptions.HTTPError as e:
            # More detailed error information
            error_detail = f"HTTP {response.status_code}: {response.text}"
            raise Exception(error_detail)

    def upload_campaign_content(self, api_token: str, campaign_id: str, html_content: str):
        """Upload HTML content to the campaign."""
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Generate plain text from HTML (basic conversion)
        plain_text = re.sub('<[^<]+?>', '', html_content)
        plain_text = plain_text.strip()

        payload = {
            "html": html_content,
            "plain_text": plain_text + "\n\nTo unsubscribe, click here: {$unsubscribe}"
        }

        try:
            response = requests.put(f"https://connect.mailerlite.com/api/campaigns/{campaign_id}/content", json=payload, headers=headers)
            print(f"Debug: Content upload response: {response.status_code} - {response.text}")
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_detail = f"Content upload failed - HTTP {response.status_code}: {response.text}"
            raise Exception(error_detail)

    def send_campaign(self, api_token: str, campaign_id: str):
        """Send the campaign."""
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "type": "instant"
        }

        try:
            response = requests.post(f"https://connect.mailerlite.com/api/campaigns/{campaign_id}/actions/schedule", json=payload, headers=headers)
            print(f"Debug: Send campaign response: {response.status_code} - {response.text}")
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_detail = f"Campaign send failed - HTTP {response.status_code}: {response.text}"
            raise Exception(error_detail)

    def send_test_email(self, api_token: str, campaign_id: str, test_email: str):
        """Send a test email for the campaign."""
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {"emails": [test_email]}

        try:
            response = requests.post(f"https://connect.mailerlite.com/api/campaigns/{campaign_id}/actions/send-test", json=payload, headers=headers)
            print(f"Debug: Test email response: {response.status_code} - {response.text}")
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_detail = f"Test email failed - HTTP {response.status_code}: {response.text}"
            raise Exception(error_detail)

if __name__ == "__main__":
    st.write("MailerLite Campaign Plugin standalone test")
