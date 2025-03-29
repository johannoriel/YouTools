import requests
import os
import shutil
from PIL import Image
import cv2
from io import BytesIO
from typing import Dict, List
import uuid


class PexelsAPI:
    def extract_title_from_url(self, url):
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
                title = self.extract_title_from_url(
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
                title = self.extract_title_from_url(
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

    def download(self, media_info: Dict, dest_dir: str) -> str:
        """Télécharge un média depuis Pexels"""
        os.makedirs(dest_dir, exist_ok=True)
        url = media_info["original_url"]

        # Déterminer l'extension à partir de l'URL ou utiliser .jpg par défaut
        ext = os.path.splitext(url.split('?')[0])[1] or ".jpg"
        filename = f"{media_info['name']}{ext}"
        filepath = os.path.join(dest_dir, filename)

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            return filepath
        raise Exception(f"Download failed: {response.status_code}")


class CanvaAPI:
    def search(self, keywords: str, api_key: str) -> List[Dict]:
        """Recherche d'images sur Canva"""
        # Implémentation simplifiée - à adapter avec la vraie API Canva
        return []

    def download(self, media_info: Dict, dest_dir: str) -> str:
        """Télécharge un média depuis Canva"""
        raise NotImplementedError("Canva API not fully implemented yet")
