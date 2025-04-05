import requests
import os
import shutil
from PIL import Image
import cv2
from io import BytesIO
from typing import Dict, List
import uuid
from bs4 import BeautifulSoup

def asset_download(media_info: Dict, dest_dir: str) -> str:
    """Télécharge un média depuis Pexels"""
    os.makedirs(dest_dir, exist_ok=True)
    url = media_info["original_url"]

    # Déterminer l'extension à partir de l'URL ou utiliser .jpg par défaut
    ext = os.path.splitext(url.split('?')[0])[1] or ".jpg"
    filename = f"{media_info['name']}{ext}"
    filepath = os.path.join(dest_dir, filename)

    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        return filepath
    raise Exception(f"Download failed: {response.status_code}")

def asset_memory_download(media_info: Dict) -> tuple:
    """Télécharge un média en mémoire sans écrire sur le disque"""
    url = media_info["original_url"]
    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        file_data = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            file_data.write(chunk)
        file_data.seek(0)  # Rewind to start of file
        return file_data, media_info["type"]
    raise Exception(f"Download failed: {response.status_code}")

def extract_title_from_url(url):
    """Extrait le titre descriptif depuis une URL Pexels."""
    try:
        if "video" in url:
            prefix = "https://www.pexels.com/video/"
        else:
            prefix = "https://www.pexels.com/photo/"

        title = url.replace(prefix, "").rstrip("/").rsplit("-", 1)[0]
        # Convertit en format lisible
        return title.replace("-", " ").title()
    except Exception:
        return ""

class PexelsAPI:


    # Dans assets_api.py, méthode search de PexelsAPI:
    def search(self, keywords, api_key, media_type="photos"):
        headers = {"Authorization": api_key}
        params = {
            "query": keywords,
            "per_page": 80  # Maximum allowed by Pexels API
        }

        endpoint = "https://api.pexels.com/v1/search"
        if media_type == "videos":
            endpoint = "https://api.pexels.com/videos/search"

        response = requests.get(endpoint, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Pexels API error: {response.text}")

        data = response.json()
        results = []

        if media_type == "photos":
            for photo in data.get("photos", []):
                title = extract_title_from_url(
                    photo["url"]) or f"Photo {photo['id']}"
                results.append({
                    "id": photo["id"],
                    "url": photo["src"]["medium"],
                    "name": title,
                    "date": photo.get("created_at", ""),
                    "original_url": photo["src"]["original"],
                    "photographer": photo["photographer"],
                    "type": "photo"
                })
        else:  # videos
            for video in data.get("videos", []):
                # Prendre la première vidéo de qualité moyenne disponible
                title = extract_title_from_url(
                    video["url"]) or f"Video {video['id']}"
                video_file = next(
                    (v for v in video["video_files"] if v["quality"] == "sd"), video["video_files"][0])
                results.append({
                    "id": video["id"],
                    "url": video["image"],  # Image de preview
                    "name": title,
                    "date": video.get("created_at", ""),
                    "original_url": video_file["link"],
                    "photographer": video['user']['name'],
                    "type": "video"
                })

        return results



class CanvaAPI:
    def search(self, keywords: str, api_key: str) -> List[Dict]:
        """Recherche d'images sur Canva"""
        # Implémentation simplifiée - à adapter avec la vraie API Canva
        return []

    def download(self, media_info: Dict, dest_dir: str) -> str:
        """Télécharge un média depuis Canva"""
        raise NotImplementedError("Canva API not fully implemented yet")


class GoogleImageAPI:
    def __init__(self):
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, keywords: str, api_key: str, cx: str, num_results: int = 10) -> List[Dict]:
        """
        Recherche d'images via Google Custom Search API
        :param keywords: Mots-clés de recherche
        :param api_key: Clé API Google
        :param cx: ID du moteur de recherche personnalisé (Custom Search Engine ID)
        :param num_results: Nombre de résultats (max 10 par requête)
        """
        params = {
            "q": keywords,
            "searchType": "image",
            "key": api_key,
            "cx": cx,
            "num": min(num_results, 10)  # Google limite à 10 par requête
        }

        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Google API error: {response.text}")

        data = response.json()
        results = []

        for item in data.get("items", []):
            title = item.get("title", f"Image {item.get('link', '').split('/')[-1]}")
            results.append({
                "id": item.get("link", "").split("/")[-1] or str(uuid.uuid4()),  # ID basé sur URL ou UUID
                "url": item.get("image", {}).get("thumbnailLink", ""),  # Aperçu
                "name": title,
                "date": item.get("snippet", ""),
                "original_url": item["link"],  # URL originale de l'image
                "type": "photo"
            })

        return results


class DuckDuckGoImageAPI:
    def __init__(self):
        self.base_url = "https://duckduckgo.com/"

    def search(self, keywords: str, max_results: int = 10) -> List[Dict]:
        """
        Recherche d'images via DuckDuckGo (scraping basique car pas d'API officielle)
        :param keywords: Mots-clés de recherche
        :param max_results: Nombre maximum de résultats
        """
        params = {"q": keywords, "t": "h_", "iar": "images", "iax": "images"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        headers = {
                'authority': 'duckduckgo.com',
                'accept': 'application/json, text/javascript, */*; q=0.01',
                'sec-fetch-dest': 'empty',
                'x-requested-with': 'XMLHttpRequest',
                'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36',
                'sec-fetch-site': 'same-origin',
                'sec-fetch-mode': 'cors',
                'referer': 'https://duckduckgo.com/',
                'accept-language': 'en-US,en;q=0.9',
            }

        # Première requête pour obtenir le token vqd
        response = requests.get(self.base_url, params=params, headers=headers)
        if response.status_code != 200:
            raise Exception(f"DuckDuckGo error: {response.status_code}")

        soup = BeautifulSoup(response.text, "html.parser")
        vqd = None
        for script in soup.find_all("script"):
            if "vqd=" in str(script):
                vqd = str(script).split('vqd="')[1].split('"')[0]
                break

        if not vqd:
            raise Exception("Could not extract vqd token")

        # Requête pour les images
        image_url = "https://duckduckgo.com/i.js"
        params = {"q": keywords, "vqd": vqd, "l": "us-en", "o": "json", "p": "1"}
        params = (
                ('l', 'us-en'),
                ('o', 'json'),
                ('q', keywords),
                ('vqd', vqd),
                ('f', ',,,'),
                ('p', '1'),
                ('v7exp', 'a'),
            )
        response = requests.get(image_url, params=params, headers=headers)
        import streamlit as st
        st.write(response)
        if response.status_code != 200:
            raise Exception(f"DuckDuckGo image fetch error: {response.status_code}")

        data = response.json()
        results = []

        for i, item in enumerate(data.get("results", [])[:max_results]):
            title = item.get("title", f"Image {i}")
            results.append({
                "id": str(uuid.uuid4()),  # Pas d'ID natif, on génère un UUID
                "url": item.get("thumbnail", ""),  # Aperçu
                "name": title,
                "date": "",  # Pas de date disponible facilement
                "original_url": item["image"],  # URL originale
                "type": "photo"
            })

        return results

class VlipsyAPI:
    def __init__(self, api_key: str = "vl_hFxn07bG43d0n9t"):
        """
        Initialise l'API Vlipsy avec une clé par défaut ou une clé fournie par l'utilisateur.
        :param api_key: Clé API Vlipsy (par défaut : vl_hFxn07bG43d0n9t)
        """
        self.base_url = "https://apiv2.vlipsy.com/v1/vlips"
        self.api_key = api_key

    def search(self, keywords: str, max_results: int = 10) -> List[Dict]:
        """
        Recherche des vidéos MP4 sur Vlipsy via l'API.
        :param keywords: Mots-clés de recherche
        :param max_results: Nombre maximum de résultats à retourner
        :return: Liste de dictionnaires avec les détails des vidéos
        """
        # Étape 1 : Requête de recherche
        search_url = f"{self.base_url}/search"
        params = {
            "q": keywords,
            "limit": 30,
            "key": self.api_key
        }

        response = requests.get(search_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Vlipsy API search error: {response.status_code} - {response.text}")

        data = response.json()
        search_results = data.get("data", [])
        results = []

        # Étape 2 : Pour chaque résultat, obtenir les détails via l'API "vlip by ID"
        for item in search_results[:max_results]:
            vlip_id = item.get("id")
            print(vlip_id)
            if not vlip_id:
                continue

            # Requête pour obtenir les détails du vlip
            vlip_url = f"{self.base_url}/{vlip_id}"
            params = {"key": self.api_key}
            vlip_response = requests.get(vlip_url, params=params)

            if vlip_response.status_code != 200:
                print(f"Error : {vlip_response}")
                continue  # Passe au suivant si la requête échoue

            vlip_data = vlip_response.json().get("data", {})
            title = vlip_data.get("title", f"Video {vlip_id}")
            video_url = vlip_data.get("media", {}).get("mp4", "").get("url", "")
            thumbnail_url = vlip_data.get("media", {}).get("preview_small").get("url", "")

            if not video_url:
                continue  # Ignore si pas de vidéo MP4

            results.append({
                "id": vlip_id,
                "url": thumbnail_url,
                "name": title,
                "date": vlip_data.get("created_at", ""),
                "original_url": video_url,  # URL directe du MP4
                "photographer": vlip_data.get("creator", {}).get("username", "Unknown"),  # Créateur
                "type": "video"
            })

        return results
