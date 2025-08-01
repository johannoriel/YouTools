from abc import ABC, abstractmethod
import streamlit as st
import requests
import markdown
import re
import json
from datetime import datetime, timedelta

class EmailProvider(ABC):
    """Abstract base class for email campaign providers."""

    def __init__(self, api_key: str, debug: bool = False):
        self.api_key = api_key
        self.debug = debug

    def _debug_write(self, message: str):
        """Write debug message if debug is enabled."""
        if self.debug:
            st.write(message)

    @abstractmethod
    def get_groups(self) -> list:
        """Fetch recipient groups/lists."""
        pass

    @abstractmethod
    def create_campaign(self, title: str, subject: str, html_content: str,
                       group_id: str, from_name: str, from_email: str) -> str:
        """Create a campaign."""
        pass

    @abstractmethod
    def send_campaign(self, campaign_id: str) -> None:
        """Send a previously created campaign."""
        pass

    @abstractmethod
    def create_and_send_campaign(self, title: str, subject: str, html_content: str,
                                group_id: str, from_name: str, from_email: str) -> str:
        """Create and send a campaign (for backward compatibility)."""
        pass

    @abstractmethod
    def get_campaigns(self) -> list:
        """List all campaigns."""
        pass

    @abstractmethod
    def send_test_campaign(self, title: str, subject: str, html_content: str,
                          test_email: str, from_name: str, from_email: str) -> str:
        """Send a test campaign."""
        pass

class MailerLiteProvider(EmailProvider):
    """MailerLite email campaign provider."""

    def __init__(self, api_token: str, debug: bool = False):
        super().__init__(api_token, debug)
        self.base_url = "https://connect.mailerlite.com/api"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def get_groups(self) -> list:
        """Fetch subscriber groups from MailerLite API."""
        try:
            self._debug_write("🔍 **Making API call to MailerLite groups endpoint...**")
            response = requests.get(f"{self.base_url}/groups", headers=self.headers)
            self._debug_write(f"📡 **API Response Status:** {response.status_code}")

            if response.status_code != 200:
                self._debug_write(f"❌ **API Error Response:** {response.text}")

            response.raise_for_status()
            groups = response.json().get("data", [])
            self._debug_write(f"📋 **Raw API Response:** {groups[:2]}...")
            return [{"id": group["id"], "name": group["name"]} for group in groups]
        except Exception as e:
            self._debug_write(f"❌ **Exception in get_groups:** {str(e)}")
            st.error(f"Failed to fetch groups: {str(e)}")
            return []

    def create_campaign(self, title: str, subject: str, html_content: str,
                       group_id: str, from_name: str, from_email: str) -> str:
        """Create a MailerLite campaign."""
        try:
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

            self._debug_write(f"📤 **Campaign Payload:**")
            self._debug_write(f"```json\n{json.dumps(campaign_payload, indent=2)}\n```")

            self._debug_write("🚀 **Making API call to create campaign...**")
            response = requests.post(f"{self.base_url}/campaigns",
                                  json=campaign_payload, headers=self.headers)

            self._debug_write(f"📡 **Create Campaign Response Status:** {response.status_code}")
            self._debug_write(f"📄 **Response Body:** {response.text}")

            response.raise_for_status()
            campaign_id = response.json()["data"]["id"]
            self._debug_write(f"✅ **Campaign created with ID:** {campaign_id}")

            return campaign_id

        except requests.exceptions.HTTPError as e:
            self._debug_write(f"❌ **HTTP Error in create_campaign:**")
            self._debug_write(f"**Status Code:** {response.status_code}")
            self._debug_write(f"**Response Body:** {response.text}")
            raise Exception(f"MailerLite API Error - HTTP {response.status_code}: {response.text}")
        except Exception as e:
            self._debug_write(f"❌ **General Exception in create_campaign:** {str(e)}")
            raise Exception(f"MailerLite Campaign Error: {str(e)}")

    def send_campaign(self, campaign_id: str) -> None:
        """Send a previously created MailerLite campaign."""
        try:
            self._debug_write("📨 **Scheduling campaign for immediate sending...**")
            send_payload = {"delivery": "instant"}
            send_response = requests.post(f"{self.base_url}/campaigns/{campaign_id}/schedule",
                                       json=send_payload, headers=self.headers)

            self._debug_write(f"📡 **Send Campaign Response Status:** {send_response.status_code}")
            self._debug_write(f"📄 **Send Response Body:** {send_response.text}")

            send_response.raise_for_status()
            self._debug_write(f"✅ **Campaign scheduled successfully!**")

        except requests.exceptions.HTTPError as e:
            self._debug_write(f"❌ **HTTP Error in send_campaign:**")
            self._debug_write(f"**Status Code:** {send_response.status_code}")
            self._debug_write(f"**Response Body:** {send_response.text}")
            raise Exception(f"MailerLite API Error - HTTP {send_response.status_code}: {send_response.text}")
        except Exception as e:
            self._debug_write(f"❌ **General Exception in send_campaign:** {str(e)}")
            raise Exception(f"MailerLite Campaign Error: {str(e)}")

    def create_and_send_campaign(self, title: str, subject: str, html_content: str,
                                group_id: str, from_name: str, from_email: str) -> str:
        """Create and send a MailerLite campaign (for backward compatibility)."""
        campaign_id = self.create_campaign(title, subject, html_content, group_id, from_name, from_email)
        self.send_campaign(campaign_id)
        return campaign_id

    def get_campaigns(self) -> list:
        """List all MailerLite campaigns."""
        try:
            self._debug_write("🔍 **Making API call to MailerLite campaigns endpoint...**")
            response = requests.get(f"{self.base_url}/campaigns", headers=self.headers)
            self._debug_write(f"📡 **API Response Status:** {response.status_code}")

            if response.status_code != 200:
                self._debug_write(f"❌ **API Error Response:** {response.text}")

            response.raise_for_status()
            campaigns = response.json().get("data", [])
            self._debug_write(f"📋 **Raw API Response:** {campaigns[:2]}...")
            return [{"id": campaign["id"], "name": campaign["name"], "status": campaign["status"]}
                    for campaign in campaigns]
        except Exception as e:
            self._debug_write(f"❌ **Exception in get_campaigns:** {str(e)}")
            st.error(f"Failed to fetch campaigns: {str(e)}")
            return []

    def send_test_campaign(self, title: str, subject: str, html_content: str,
                         test_email: str, from_name: str, from_email: str) -> str:
        """Send a test MailerLite campaign."""
        try:
            campaign_payload = {
                "name": f"TEST - {title}",
                "type": "regular",
                "emails": [{
                    "subject": f"[TEST] {subject}",
                    "from_name": from_name,
                    "from": from_email,
                    "content": html_content
                }]
            }

            self._debug_write(f"📤 **Test Campaign Payload:**")
            self._debug_write(f"```json\n{json.dumps(campaign_payload, indent=2)}\n```")

            self._debug_write("🚀 **Making API call to create test campaign...**")
            response = requests.post(f"{self.base_url}/campaigns",
                                  json=campaign_payload, headers=self.headers)

            self._debug_write(f"📡 **Test Campaign Response Status:** {response.status_code}")
            self._debug_write(f"📄 **Response Body:** {response.text}")

            response.raise_for_status()
            campaign_id = response.json()["data"]["id"]
            self._debug_write(f"✅ **Test campaign created with ID:** {campaign_id}")
            self._debug_write("⚠️ **Note:** MailerLite test requires the email to be in a subscriber group")

            return campaign_id

        except requests.exceptions.HTTPError as e:
            self._debug_write(f"❌ **HTTP Error in send_test_campaign:**")
            self._debug_write(f"**Status Code:** {response.status_code}")
            self._debug_write(f"**Response Body:** {response.text}")
            raise Exception(f"MailerLite Test Error - HTTP {response.status_code}: {response.text}")
        except Exception as e:
            self._debug_write(f"❌ **General Exception in send_test_campaign:** {str(e)}")
            raise Exception(f"MailerLite Test Error: {str(e)}")

class BrevoProvider(EmailProvider):
    """Brevo (Sendinblue) email campaign provider."""

    def __init__(self, api_key: str, debug: bool = False):
        super().__init__(api_key, debug)
        self.base_url = "https://api.brevo.com/v3"
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def get_groups(self) -> list:
        """Fetch contact lists from Brevo API."""
        try:
            self._debug_write("🔍 **Making API call to Brevo lists endpoint...**")
            response = requests.get(f"{self.base_url}/contacts/lists", headers=self.headers)
            self._debug_write(f"📡 **API Response Status:** {response.status_code}")

            if response.status_code != 200:
                self._debug_write(f"❌ **API Error Response:** {response.text}")

            response.raise_for_status()
            lists = response.json().get("lists", [])
            self._debug_write(f"📋 **Raw API Response:** {lists[:2]}...")
            return [{"id": str(lst["id"]), "name": lst["name"]} for lst in lists]
        except Exception as e:
            self._debug_write(f"❌ **Exception in get_groups:** {str(e)}")
            st.error(f"Failed to fetch lists: {str(e)}")
            return []

    def create_campaign(self, title: str, subject: str, html_content: str,
                       group_id: str, from_name: str, from_email: str) -> str:
        """Create a Brevo campaign."""
        try:
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
                }
            }

            self._debug_write(f"📤 **Campaign Payload:**")
            self._debug_write(f"```json\n{json.dumps(campaign_payload, indent=2)}\n```")

            self._debug_write("🚀 **Making API call to create campaign...**")
            response = requests.post(f"{self.base_url}/emailCampaigns",
                                  json=campaign_payload, headers=self.headers)

            self._debug_write(f"📡 **Create Campaign Response Status:** {response.status_code}")
            self._debug_write(f"📋 **Response Headers:** {dict(response.headers)}")
            self._debug_write(f"📄 **Response Body:** {response.text}")

            response.raise_for_status()
            campaign_id = response.json()["id"]
            self._debug_write(f"✅ **Campaign created with ID:** {campaign_id}")

            return str(campaign_id)

        except requests.exceptions.HTTPError as e:
            self._debug_write(f"❌ **HTTP Error in create_campaign:**")
            self._debug_write(f"**Status Code:** {response.status_code}")
            self._debug_write(f"**Response Body:** {response.text}")
            raise Exception(f"Brevo API Error - HTTP {response.status_code}: {response.text}")
        except Exception as e:
            self._debug_write(f"❌ **General Exception in create_campaign:** {str(e)}")
            raise Exception(f"Brevo Campaign Error: {str(e)}")

    def send_campaign(self, campaign_id: str) -> None:
        """Send a previously created Brevo campaign."""
        try:
            scheduled_at = (datetime.now() + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
            self._debug_write(f"⏰ **Scheduled sending time:** {scheduled_at}")

            send_payload = {"scheduledAt": scheduled_at}
            self._debug_write("📨 **Scheduling campaign for sending...**")
            response = requests.put(f"{self.base_url}/emailCampaigns/{campaign_id}",
                                  json=send_payload, headers=self.headers)

            self._debug_write(f"📡 **Send Campaign Response Status:** {response.status_code}")
            self._debug_write(f"📄 **Response Body:** {response.text}")

            response.raise_for_status()
            self._debug_write(f"✅ **Campaign scheduled successfully!**")

        except requests.exceptions.HTTPError as e:
            self._debug_write(f"❌ **HTTP Error in send_campaign:**")
            self._debug_write(f"**Status Code:** {response.status_code}")
            self._debug_write(f"**Response Body:** {response.text}")
            raise Exception(f"Brevo API Error - HTTP {response.status_code}: {response.text}")
        except Exception as e:
            self._debug_write(f"❌ **General Exception in send_campaign:** {str(e)}")
            raise Exception(f"Brevo Campaign Error: {str(e)}")

    def create_and_send_campaign(self, title: str, subject: str, html_content: str,
                                group_id: str, from_name: str, from_email: str) -> str:
        """Create and send a Brevo campaign (for backward compatibility)."""
        campaign_id = self.create_campaign(title, subject, html_content, group_id, from_name, from_email)
        self.send_campaign(campaign_id)
        return str(campaign_id)

    def get_campaigns(self) -> list:
        """List all Brevo campaigns."""
        try:
            self._debug_write("🔍 **Making API call to Brevo campaigns endpoint...**")
            response = requests.get(f"{self.base_url}/emailCampaigns", headers=self.headers)
            self._debug_write(f"📡 **API Response Status:** {response.status_code}")

            if response.status_code != 200:
                self._debug_write(f"❌ **API Error Response:** {response.text}")

            response.raise_for_status()
            campaigns = response.json().get("campaigns", [])
            self._debug_write(f"📋 **Raw API Response:** {campaigns[:2]}...")
            return [{"id": str(campaign["id"]), "name": campaign["name"], "status": campaign["status"]}
                    for campaign in campaigns]
        except Exception as e:
            self._debug_write(f"❌ **Exception in get_campaigns:** {str(e)}")
            st.error(f"Failed to fetch campaigns: {str(e)}")
            return []

    def send_test_campaign(self, title: str, subject: str, html_content: str,
                         test_email: str, from_name: str, from_email: str) -> str:
        """Send a test email via Brevo."""
        try:
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

            self._debug_write(f"📤 **Test Email Payload:**")
            self._debug_write(f"```json\n{json.dumps(test_payload, indent=2)}\n```")

            self._debug_write("🚀 **Making API call to send test email...**")
            response = requests.post(f"{self.base_url}/smtp/email",
                                  json=test_payload, headers=self.headers)

            self._debug_write(f"📡 **Test Email Response Status:** {response.status_code}")
            self._debug_write(f"📋 **Response Headers:** {dict(response.headers)}")
            self._debug_write(f"📄 **Response Body:** {response.text}")

            response.raise_for_status()
            response_data = response.json() if response.text else {}
            message_id = response_data.get("messageId", "test-sent-no-id")
            self._debug_write(f"✅ **Test email sent with Message ID:** {message_id}")

            return str(message_id)

        except requests.exceptions.HTTPError as e:
            self._debug_write(f"❌ **HTTP Error in send_test_campaign:**")
            self._debug_write(f"**Status Code:** {response.status_code}")
            self._debug_write(f"**Response Body:** {response.text}")
            raise Exception(f"Brevo Test Error - HTTP {response.status_code}: {response.text}")
        except Exception as e:
            self._debug_write(f"❌ **General Exception in send_test_campaign:** {str(e)}")
            raise Exception(f"Brevo Test Error: {str(e)}")
