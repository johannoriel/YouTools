import sqlite3
from datetime import datetime
import pytz
from typing import List, Dict, Any, Optional
from youtube_api import YoutubeAPI
import json

# Database file
DB_FILE = "youtube_database.db"
SCHEMA_VERSION = 2  # Nouvelle version avec le statut


def get_db_connection():
    """Create or connect to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn


def reset_database():
    """Reset the database structure to version 0 and recreate tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Supprimer toutes les tables
    cursor.execute("DROP TABLE IF EXISTS schema_version")
    cursor.execute("DROP TABLE IF EXISTS videos")
    cursor.execute("DROP TABLE IF EXISTS stats_snapshots")
    cursor.execute("DROP TABLE IF EXISTS campaign_cache")
    cursor.execute("DROP TABLE IF EXISTS target_channels")

    # Réinitialiser avec la version courante
    initialize_database()

    conn.commit()
    conn.close()


def upgrade_database(current_version: int, target_version: int, cursor):
    """Handle database schema upgrades."""
    if current_version < 2 < target_version:
        # Placeholder for future upgrades
        pass
    # Add more upgrade steps as schema evolves


def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        )
    """)

    cursor.execute("SELECT version FROM schema_version")
    current_version = cursor.fetchone()
    if not current_version:
        cursor.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

    cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                thumbnail_url TEXT,
                transcript TEXT,
                description TEXT,
                published_at TEXT,
                status TEXT,
                keywords TEXT DEFAULT '[]'  -- Nouveau champ pour les mots-clés, JSON par défaut une liste vide
            )
        """)

    # Nouvelle structure avec advanced_stats en JSON
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stats_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            timestamp TEXT,
            view_count INTEGER,
            retention_rate REAL,
            advanced_stats TEXT,  -- JSON pour les stats avancées
            FOREIGN KEY (video_id) REFERENCES videos (video_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaign_cache (
            cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_video_id TEXT,
            target_video_id TEXT,
            comment_id TEXT,
            comment_text TEXT,
            response_text TEXT,
            status TEXT,
            timestamp TEXT,
            FOREIGN KEY (campaign_video_id) REFERENCES videos (video_id)
        )
    """)

    cursor.execute("""
            CREATE TABLE IF NOT EXISTS target_channels (
                channel_id TEXT PRIMARY KEY,
                channel_title TEXT,
                channel_url TEXT UNIQUE,
                subscriber_count INTEGER DEFAULT 0,
                keywords TEXT,  -- JSON contenant la liste des mots-clés
                added_at TEXT,
                last_updated TEXT
            )
        """)

    cursor.execute("""
                CREATE TABLE IF NOT EXISTS campaign_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT,
                    comment_id TEXT,
                    comment_text TEXT,
                    response_text TEXT,
                    video_id TEXT,
                    channel_id TEXT,
                    author TEXT,
                    status TEXT,
                    timestamp TEXT
                )
            """)

    conn.commit()
    conn.close()


def get_latest_stats(video_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM stats_snapshots
        WHERE video_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (video_id,))
    stats = cursor.fetchone()

    conn.close()
    if stats:
        stats_dict = dict(stats)
        stats_dict['advanced_stats'] = json.loads(stats_dict['advanced_stats'])
        return stats_dict
    return None


def insert_stats_snapshot(video_id: str, timestamp: str, stats: Dict[str, Any]):
    """Insère un snapshot de statistiques dans la base."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO stats_snapshots (
            video_id, timestamp, view_count, retention_rate, advanced_stats
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        video_id,
        timestamp,
        stats['view_count'],
        stats['retention_rate'],
        json.dumps(stats['advanced_stats'])
    ))

    conn.commit()
    conn.close()


def get_stats_history(video_id: str) -> List[Dict[str, Any]]:
    """Get all historical stats snapshots for a video."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM stats_snapshots
        WHERE video_id = ?
        ORDER BY timestamp DESC
    """, (video_id,))
    history = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return history


def cache_campaign_data(campaign_video_id: str, target_video_id: str, comment_id: str, comment_text: str, response_text: str = "", status: str = "pending"):
    """Cache campaign data for later validation or reuse."""
    conn = get_db_connection()
    cursor = conn.cursor()

    timestamp = datetime.now(pytz.UTC).isoformat()
    cursor.execute("""
        INSERT INTO campaign_cache (campaign_video_id, target_video_id, comment_id, comment_text, response_text, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (campaign_video_id, target_video_id, comment_id, comment_text, response_text, status, timestamp))

    conn.commit()
    conn.close()


def update_campaign_response(cache_id: int, response_text: str, status: str = "validated"):
    """Update a cached response after user validation."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE campaign_cache
        SET response_text = ?, status = ?, timestamp = ?
        WHERE cache_id = ?
    """, (response_text, status, datetime.now(pytz.UTC).isoformat(), cache_id))

    conn.commit()
    conn.close()


def get_campaign_data(campaign_video_id: str, status: str = None) -> List[Dict[str, Any]]:
    """Retrieve cached campaign data."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM campaign_cache WHERE campaign_video_id = ?"
    params = [campaign_video_id]
    if status:
        query += " AND status = ?"
        params.append(status)

    cursor.execute(query, params)
    data = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return data


def mark_campaign_posted(cache_id: int, status: str = "posted"):
    """Mark a campaign response as posted."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE campaign_cache
        SET status = ?, timestamp = ?
        WHERE cache_id = ?
    """, (status, datetime.now(pytz.UTC).isoformat(), cache_id))

    conn.commit()
    conn.close()


def delete_stats_snapshot(timestamp: str):
    """Delete all stats snapshots for a specific timestamp."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM stats_snapshots
        WHERE timestamp = ?
    """, (timestamp,))

    conn.commit()
    conn.close()


def get_stats_snapshots_timestamps() -> List[str]:
    """Retrieve all unique timestamps from stats_snapshots."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT DISTINCT timestamp FROM stats_snapshots ORDER BY timestamp DESC")
    timestamps = [row["timestamp"] for row in cursor.fetchall()]

    conn.close()
    return timestamps


def sync_videos(channel_id: str, youtube_api: YoutubeAPI):
    """Sync all videos from the channel into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    videos = youtube_api.get_channel_videos(channel_id)
    for video in videos:
        cursor.execute("""
            INSERT OR REPLACE INTO videos (video_id, url, title, thumbnail_url, transcript, description, published_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video['video_id'],
            video['url'],
            video['title'],
            video['thumbnail'],
            "",  # Transcript placeholder
            video['description'],
            video['published_at'],
            video['status']  # Ajout du statut
        ))

    conn.commit()
    conn.close()


def update_video_keywords(video_id: str, keywords: List[str]):
    """Met à jour les mots-clés d'une vidéo."""
    conn = get_db_connection()
    cursor = conn.cursor()
    keywords_json = json.dumps(keywords)
    cursor.execute(
        "UPDATE videos SET keywords = ? WHERE video_id = ?", (keywords_json, video_id))
    conn.commit()
    conn.close()


def get_videos(filter_type: str = "title", keyword: str = "", page: int = 1, per_page: int = 100, keyword_filter: List[str] = None) -> List[Dict[str, Any]]:
    """Récupère les vidéos avec un filtre optionnel par mots-clés."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query_base = "SELECT * FROM videos WHERE "
    params = []
    conditions = []

    if filter_type == "title":
        conditions.append("title LIKE ?")
        params.append(f"%{keyword}%")
    elif filter_type == "title_description":
        conditions.append("(title LIKE ? OR description LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    elif filter_type == "all":
        conditions.append(
            "(title LIKE ? OR description LIKE ? OR transcript LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])

    if keyword_filter:
        # Filtrer par mots-clés (recherche dans le champ JSON)
        for kw in keyword_filter:
            conditions.append("keywords LIKE ?")
            params.append(f"%{kw}%")

    if conditions:
        query = query_base + " AND ".join(conditions)
    else:
        query = "SELECT * FROM videos"

    # Pagination
    offset = (page - 1) * per_page
    query += " LIMIT ? OFFSET ?"
    params.extend([per_page, offset])

    cursor.execute(query, params)
    videos = [dict(row) for row in cursor.fetchall()]
    for video in videos:
        video['keywords'] = json.loads(video['keywords'])

    conn.close()
    return videos


def add_target_channel(channel_id: str, channel_title: str, channel_url: str, keywords: List[str], subscriber_count: int = 0) -> None:
    """Ajoute une chaîne cible à la base de données."""
    conn = get_db_connection()
    cursor = conn.cursor()

    timestamp = datetime.now(pytz.UTC).isoformat()
    keywords_json = json.dumps(keywords)

    cursor.execute("""
        INSERT OR REPLACE INTO target_channels (channel_id, channel_title, channel_url, subscriber_count, keywords, added_at, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (channel_id, channel_title, channel_url, subscriber_count, keywords_json, timestamp, timestamp))

    conn.commit()
    conn.close()


def update_target_channel_keywords(channel_id: str, keywords: List[str]) -> None:
    """Met à jour les mots-clés d'une chaîne cible."""
    conn = get_db_connection()
    cursor = conn.cursor()

    timestamp = datetime.now(pytz.UTC).isoformat()
    keywords_json = json.dumps(keywords)

    cursor.execute("""
        UPDATE target_channels
        SET keywords = ?, last_updated = ?
        WHERE channel_id = ?
    """, (keywords_json, timestamp, channel_id))

    conn.commit()
    conn.close()


def update_target_channel_stats(channel_id: str, subscriber_count: int) -> None:
    """Met à jour les statistiques d'une chaîne cible."""
    conn = get_db_connection()
    cursor = conn.cursor()

    timestamp = datetime.now(pytz.UTC).isoformat()

    cursor.execute("""
        UPDATE target_channels
        SET subscriber_count = ?, last_updated = ?
        WHERE channel_id = ?
    """, (subscriber_count, timestamp, channel_id))

    conn.commit()
    conn.close()


def delete_target_channel(channel_id: str) -> None:
    """Supprime une chaîne cible de la base de données."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM target_channels WHERE channel_id = ?", (channel_id,))

    conn.commit()
    conn.close()


def get_target_channels() -> List[Dict[str, Any]]:
    """Récupère toutes les chaînes cibles avec leurs informations."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM target_channels")
    channels = [dict(row) for row in cursor.fetchall()]

    for channel in channels:
        channel['keywords'] = json.loads(channel['keywords'])

    conn.close()
    return channels


def get_target_channel(channel_id: str) -> Optional[Dict[str, Any]]:
    """Récupère les informations d'une chaîne cible spécifique."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM target_channels WHERE channel_id = ?", (channel_id,))
    channel = cursor.fetchone()

    conn.close()
    if channel:
        channel_dict = dict(channel)
        channel_dict['keywords'] = json.loads(channel_dict['keywords'])
        return channel_dict
    return None


def cache_campaign_response(campaign_id: str, comment_id: str, comment_text: str, response_text: str, video_id: str, channel_id: str, author: str, status: str = "pending"):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now(pytz.UTC).isoformat()
    cursor.execute("""
        INSERT INTO campaign_responses (campaign_id, comment_id, comment_text, response_text, video_id, channel_id, author, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (campaign_id, comment_id, comment_text, response_text, video_id, channel_id, author, status, timestamp))
    conn.commit()
    conn.close()


def update_campaign_response_status(comment_id: str, status: str, campaign_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now(pytz.UTC).isoformat()
    cursor.execute("""
        UPDATE campaign_responses
        SET status = ?, timestamp = ?
        WHERE comment_id = ? AND campaign_id = ?
    """, (status, timestamp, comment_id, campaign_id))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    initialize_database()
