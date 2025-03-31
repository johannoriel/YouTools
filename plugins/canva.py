from global_vars import translations, t
from app import Plugin
import streamlit as st
import requests
import os
import base64
import hashlib
import secrets
from urllib.parse import urlencode
import json
from plugins.common import remove_quotes
import time

# Ajout des traductions pour le plugin Canva
translations["en"].update({
    "canva_tab": "Canva",
    "canva_header": "Canva Design Manager",
    "canva_designs_tab": "Designs",
    "canva_jobs_tab": "Jobs",
    "canva_login_tab": "Login",
    "canva_warning_login": "Please login in the Login tab.",
    "canva_your_designs": "Your designs",
    "canva_export": "Export",
    "canva_export_format": "Export format",
    "canva_jpg_quality": "JPG quality",
    "canva_export_pages": "Pages to export (e.g. 1 or 1,3)",
    "canva_confirm_export": "Confirm export",
    "canva_export_jobs": "Export jobs",
    "canva_add_job_id": "Add Job ID manually",
    "canva_add_job": "Add Job ID",
    "canva_refresh_jobs": "Refresh job status",
    "canva_no_jobs": "No jobs to monitor.",
    "canva_current_jobs": "Current jobs:",
    "canva_no_designs": "No designs found.",
    "canva_login_success": "Successfully logged in!",
    "canva_logout": "Logout (delete token)",
    "canva_login_prompt": "Connect to Canva",
    "canva_login_instruction": "Click the link to authenticate",
    "canva_refresh_token": "Refresh token",
    "canva_token_refreshed": "Token refreshed successfully!",
    "canva_export_created": "Export job created: {job_id}",
    "canva_follow_job": "Track its progress in the Jobs tab.",
    "canva_export_downloaded": "Export downloaded: {filepath}",
    "canva_job_failed": "Job {job_id} failed.",
    "canva_token_loaded": "Token loaded successfully!",
    "canva_token_deleted": "Token deleted. Please login again.",
    "canva_export_in_progress": "Creating export...",
    "canva_export_error": "Export error: {error}",
    "canva_download_error": "Download error",
    "canva_redirect_uri": "Redirect URI",
    "canva_client_id": "Canva Client ID",
    "canva_client_secret": "Canva Client Secret",
    "canva_download_dir": "Download directory",
    "canva_default_redirect": "http://127.0.0.1:8501/",
})

translations["fr"].update({
    "canva_tab": "Canva",
    "canva_header": "Gestionnaire de designs Canva",
    "canva_designs_tab": "Designs",
    "canva_jobs_tab": "Jobs",
    "canva_login_tab": "Connexion",
    "canva_warning_login": "Veuillez vous connecter dans l'onglet Connexion.",
    "canva_your_designs": "Vos designs",
    "canva_export": "Exporter",
    "canva_export_format": "Format d'export",
    "canva_jpg_quality": "Qualité JPG",
    "canva_export_pages": "Pages à exporter (ex: 1 ou 1,3)",
    "canva_confirm_export": "Confirmer l'export",
    "canva_export_jobs": "Jobs d'export",
    "canva_add_job_id": "Ajouter un Job ID manuellement",
    "canva_add_job": "Ajouter Job ID",
    "canva_refresh_jobs": "Rafraîchir l'état des jobs",
    "canva_no_jobs": "Aucun job à surveiller.",
    "canva_current_jobs": "Jobs en cours :",
    "canva_no_designs": "Aucun design trouvé.",
    "canva_login_success": "Authentification réussie !",
    "canva_logout": "Déconnexion (supprimer le token)",
    "canva_login_prompt": "Connectez-vous à Canva",
    "canva_login_instruction": "Cliquez sur le lien pour vous authentifier",
    "canva_refresh_token": "Renouveler le token",
    "canva_token_refreshed": "Token renouvelé avec succès !",
    "canva_export_created": "Job d'export créé : {job_id}",
    "canva_follow_job": "Suivez son état dans l'onglet Jobs.",
    "canva_export_downloaded": "Export téléchargé : {filepath}",
    "canva_job_failed": "Le job {job_id} a échoué.",
    "canva_token_loaded": "Token chargé avec succès !",
    "canva_token_deleted": "Token supprimé. Veuillez vous reconnecter.",
    "canva_export_in_progress": "Création de l'export...",
    "canva_export_error": "Erreur d'export : {error}",
    "canva_download_error": "Erreur de téléchargement",
    "canva_redirect_uri": "URI de redirection",
    "canva_client_id": "Canva Client ID",
    "canva_client_secret": "Canva Client Secret",
    "canva_download_dir": "Répertoire de téléchargement",
    "canva_default_redirect": "http://127.0.0.1:8501/",
})

class CanvaPlugin(Plugin):
    def __init__(self, name: str, plugin_manager):
        super().__init__(name, plugin_manager)
        self.CODE_VERIFIER_FILE = "code_verifier.txt"
        self.TOKEN_FILE = "canva_token.json"

    def get_config_fields(self):
        """Définit les champs de configuration du plugin"""
        return {
            "canva_client_id": {
                "type": "text",
                "label": t("canva_client_id"),
                "default": ""
            },
            "canva_client_secret": {
                "type": "text",
                "label": t("canva_client_secret"),
                "default": ""
            },
            "canva_redirect_uri": {
                "type": "text",
                "label": t("canva_redirect_uri"),
                "default": t("canva_default_redirect")
            },
            "canva_download_dir": {
                "type": "text",
                "label": t("canva_download_dir"),
                "default": os.path.expanduser("~/Videos")
            }
        }

    def get_tabs(self):
        """Définit les onglets du plugin dans l'interface."""
        return [{"name": t("canva_tab"), "plugin": "canva"}]

    def generate_pkce_pair(self):
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(code_challenge).decode().rstrip("=")
        return code_verifier, code_challenge

    def save_code_verifier(self, code_verifier):
        with open(self.CODE_VERIFIER_FILE, "w") as f:
            f.write(code_verifier)

    def load_code_verifier(self):
        if os.path.exists(self.CODE_VERIFIER_FILE):
            with open(self.CODE_VERIFIER_FILE, "r") as f:
                return f.read().strip()
        return None

    def clear_code_verifier(self):
        if os.path.exists(self.CODE_VERIFIER_FILE):
            os.remove(self.CODE_VERIFIER_FILE)

    def save_token(self, token_data):
        with open(self.TOKEN_FILE, "w") as f:
            json.dump(token_data, f)

    def load_token(self, config):
        if not os.path.exists(self.TOKEN_FILE):
            return None

        try:
            with open(self.TOKEN_FILE, "r") as f:
                token_data = json.load(f)

            # Vérifier si le token est expiré
            if token_data.get("expires_at", 0) < time.time():
                st.warning("Token expired, attempting refresh...")
                if "refresh_token" in token_data:
                    new_token = self.refresh_token(token_data["refresh_token"], config)
                    if new_token:
                        return new_token
                return None

            return token_data
        except Exception as e:
            st.error(f"Error loading token: {str(e)}")
            return None

    def get_auth_url(self, code_challenge, config):
        auth_url = "https://www.canva.com/api/oauth/authorize"
        params = {
            "client_id": self.getconfig("canva_client_id"),
            "response_type": "code",
            "scope": "design:content:read folder:read asset:read design:meta:read",
            "redirect_uri": self.get_config("canva_redirect_uri"),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256"
        }
        return f"{auth_url}?{urlencode(params)}"

    def exchange_code_for_token(self, code, code_verifier, config):
        token_url = "https://api.canva.com/rest/v1/oauth/token"
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.get_config("canva_client_id"),
            "client_secret": self.get_config("canva_client_secret"),
            "redirect_uri": self.get_config("canva_redirect_uri"),
            "code": code,
            "code_verifier": code_verifier
        }
        response = requests.post(token_url, data=payload)
        if response.status_code == 200:
            token_data = response.json()
            # Ajouter le timestamp d'expiration
            token_data["expires_at"] = time.time() + token_data.get("expires_in", 3600) - 300  # 5 minutes de marge
            return token_data
        else:
            st.error(t("canva_export_error").format(error=response.text))
            return None

    def refresh_token(self, refresh_token, config):
        if not refresh_token:
            st.error("No refresh token available")
            return None

        token_url = "https://api.canva.com/rest/v1/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.get_config("canva_client_id"),
            "client_secret": self.get_config("canva_client_secret"),
            "refresh_token": refresh_token
        }

        try:
            response = requests.post(token_url, data=payload)
            if response.status_code == 200:
                token_data = response.json()
                token_data["expires_at"] = time.time() + token_data.get("expires_in", 3600) - 300
                token_data["refresh_token"] = refresh_token  # Canva ne renvoie pas toujours un nouveau refresh_token
                self.save_token(token_data)
                st.success("Token refreshed successfully!")
                return token_data
            else:
                error_msg = f"Refresh failed ({response.status_code}): {response.text}"
                st.error(error_msg)
                # Si le refresh token est invalide, supprimer le token
                if response.status_code in [400, 401]:
                    if os.path.exists(self.TOKEN_FILE):
                        os.remove(self.TOKEN_FILE)
                return None
        except Exception as e:
            st.error(f"Refresh token error: {str(e)}")
            return None

    def list_designs(self, token):
        if not token:
            st.error("No access token provided")
            return None

        url = "https://api.canva.com/rest/v1/designs"
        headers = {"Authorization": f"Bearer {token}"}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                st.error("Unauthorized - Token may be invalid or expired")
                return None
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                return None
        except Exception as e:
            st.error(f"Network error: {str(e)}")
            return None

    def create_export_job(self, token, design_id, format_type="jpg", quality=80, pages="1"):
        url = "https://api.canva.com/rest/v1/exports"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "design_id": design_id,
            "format": {
                "type": format_type,
                "quality": quality if format_type == "jpg" else None,
                "pages": [int(page.strip()) for page in pages.split(",")]
            }
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["job"]["id"]
        else:
            st.error(t("canva_export_error").format(error=response.text))
            raise Exception("Canva create export job failed")
            return None

    def check_export_status(self, token, job_id):
        url = f"https://api.canva.com/rest/v1/exports/{job_id}"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            job_data = response.json()["job"]
            return job_data["status"], job_data.get("urls")
        else:
            st.error(t("canva_export_error").format(error=response.text))
            raise Exception("Canva check export status failed")
            return None, None

    def download_export(self, url, design_name, format_type, config):
        response = requests.get(url)
        if response.status_code == 200:
            download_dir = os.path.expanduser(self.get_config("canva_download_dir"))
            os.makedirs(download_dir, exist_ok=True)
            filename = f"{design_name.replace(' ', '_')}.{format_type}"
            filepath = os.path.join(download_dir, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            return filepath
        else:
            st.error(t("canva_download_error"))
            raise Exception("Canva download export failed")
            return None

    def run(self, config):
        st.header(t("canva_header"))

        # Initialisation des états dans session_state
        if "job_ids" not in st.session_state:
            st.session_state.job_ids = []
        if "export_in_progress" not in st.session_state:
            st.session_state.export_in_progress = None

        tab1, tab2, tab3 = st.tabs([
            t("canva_designs_tab"),
            t("canva_jobs_tab"),
            t("canva_login_tab")
        ])

        # Onglet Designs
        with tab1:
            token_data = self.load_token(config)
            if not token_data or "access_token" not in token_data:
                st.warning(t("canva_warning_login"))
            else:
                try:
                    designs = self.list_designs(token_data["access_token"])
                    if designs and "items" in designs:
                        st.write(f"### {t('canva_your_designs')}")
                        for item in designs["items"]:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if "thumbnail" in item and "url" in item["thumbnail"]:
                                    st.image(item["thumbnail"]["url"], width=100)
                            with col2:
                                st.write(f"**Nom**: {item.get('title', 'Sans nom')}")
                                st.write(f"**ID**: {item['id']}")
                                if st.button(t("canva_export"), key=item["id"]):
                                    st.session_state.export_in_progress = item["id"]

                                if st.session_state.export_in_progress == item["id"]:
                                    format_type = st.selectbox(
                                        t("canva_export_format"),
                                        ["jpg", "png", "pdf"],
                                        index=0,
                                        key=f"format_{item['id']}"
                                    )
                                    quality = 80
                                    if format_type == "jpg":
                                        quality = st.slider(
                                            t("canva_jpg_quality"),
                                            0, 100, 80,
                                            key=f"quality_{item['id']}"
                                        )
                                    pages = st.text_input(
                                        t("canva_export_pages"),
                                        value="1",
                                        key=f"pages_{item['id']}"
                                    )
                                    if st.button(t("canva_confirm_export"), key=f"confirm_{item['id']}"):
                                        with st.spinner(t("canva_export_in_progress")):
                                            job_id = self.create_export_job(
                                                token_data["access_token"],
                                                item["id"],
                                                format_type,
                                                quality,
                                                pages
                                            )
                                            if job_id:
                                                st.session_state.job_ids.append(job_id)
                                                st.success(t("canva_export_created").format(job_id=job_id))
                                                st.write(t("canva_follow_job"))
                                                st.session_state.export_in_progress = None
                    else:
                        st.warning(t("canva_no_designs"))
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
                    if "refresh_token" in token_data:
                        st.info("Tentative de rafraîchissement du token...")
                        new_token = self.refresh_token(token_data["refresh_token"], config)
                        if new_token:
                            st.rerun()  # Recharger la page avec le nouveau token

        # Onglet Jobs
        with tab2:
            token_data = self.load_token(config)
            if not token_data or "access_token" not in token_data:
                st.warning(t("canva_warning_login"))
            else:
                st.write(f"### {t('canva_export_jobs')}")
                manual_job_id = st.text_input(t("canva_add_job_id"))
                if st.button(t("canva_add_job")):
                    if manual_job_id and manual_job_id not in st.session_state.job_ids:
                        st.session_state.job_ids.append(manual_job_id)
                        st.success(f"Job ID {manual_job_id} added.")

                if st.button(t("canva_refresh_jobs")):
                    if not st.session_state.job_ids:
                        st.warning(t("canva_no_jobs"))
                    else:
                        for job_id in st.session_state.job_ids[:]:
                            status, urls = self.check_export_status(token_data["access_token"], job_id)
                            if status:
                                st.write(f"**Job ID**: {job_id} - **État**: {status}")
                                if status == "success" and urls:
                                    url = urls[0]
                                    design_name = f"export_{job_id}"
                                    filepath = self.download_export(url, design_name, "jpg", config)
                                    if filepath:
                                        st.success(t("canva_export_downloaded").format(filepath=filepath))
                                        st.session_state.job_ids.remove(job_id)
                                elif status == "failed":
                                    st.error(t("canva_job_failed").format(job_id=job_id))
                                    st.session_state.job_ids.remove(job_id)
                            else:
                                st.error(f"Impossible de vérifier le job {job_id}")

                if st.session_state.job_ids:
                    st.write(t("canva_current_jobs"))
                    for job_id in st.session_state.job_ids:
                        st.write(f"- {job_id}")
                else:
                    st.write(t("canva_no_jobs"))

        # Onglet Connexion
        with tab3:
            token_data = self.load_token(config)

            if token_data and "access_token" in token_data:
                expires_at = token_data.get("expires_at", 0)
                expires_in = max(0, int(expires_at - time.time()))

                st.success(f"Authentifié (expire dans {expires_in//60} minutes)")
                st.write(f"Client ID: {self.get_config('canva_client_id')}")
                st.write(f"Redirect URI: {self.get_config('canva_redirect_uri')}")

                if st.button(t("canva_logout")):
                    if os.path.exists(self.TOKEN_FILE):
                        os.remove(self.TOKEN_FILE)
                    st.session_state.clear()
                    st.rerun()

                if "refresh_token" in token_data:
                    if st.button(t("canva_refresh_token")):
                        with st.spinner("Refreshing token..."):
                            new_token = self.refresh_token(token_data["refresh_token"], config)
                            if new_token:
                                st.rerun()
            else:
                query_params = st.query_params
                auth_code = query_params.get("code")

                if not auth_code:
                    code_verifier, code_challenge = self.generate_pkce_pair()
                    self.save_code_verifier(code_verifier)
                    auth_url = self.get_auth_url(code_challenge, config)

                    st.write("### Configuration OAuth")
                    st.write(f"Client ID: {self.get_config('canva_client_id')}")
                    st.write(f"Redirect URI: {self.get_config('canva_redirect_uri')}")

                    st.markdown(f"[{t('canva_login_prompt')}]({auth_url})")
                    st.write(t("canva_login_instruction"))
                else:
                    code_verifier = self.load_code_verifier()
                    if code_verifier:
                        with st.spinner("Authenticating..."):
                            token_data = self.exchange_code_for_token(auth_code, code_verifier, config)
                            if token_data:
                                self.save_token(token_data)
                                st.query_params.clear()
                                st.rerun()
