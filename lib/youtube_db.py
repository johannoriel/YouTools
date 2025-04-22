import sqlite3
from datetime import datetime
import pytz
from typing import List, Dict, Any, Optional
from lib.youtube_api import YoutubeAPI
import json

# Database file
DB_FILE = "youtube_database.db"
SCHEMA_VERSION = 5  # Nouvelle version avec le statut


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
    cursor.execute("DROP TABLE IF EXISTS posted_responses")

    # Réinitialiser avec la version courante
    initialize_database()

    conn.commit()
    conn.close()


def auto_upgrade_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT version FROM schema_version")
    current_version = cursor.fetchone()['version']
    if current_version < SCHEMA_VERSION:
        upgrade_database(current_version, SCHEMA_VERSION, cursor)
        cursor.execute("UPDATE schema_version SET version = ?",
                       (SCHEMA_VERSION,))
        conn.commit()
        conn.close()


def upgrade_database(current_version: int, target_version: int, cursor):
    """Handle database schema upgrades."""
    if current_version < 2 and target_version >= 2:
        # Upgrade de la version 1 à 2 (si pertinent)
        pass
    if current_version < 3 and target_version >= 3:
        # Upgrade vers version 3 : ajout de la table posted_responses
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS posted_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_timestamp TEXT,  -- Timestamp de la campagne
                video_id TEXT,           -- ID de la vidéo commentée
                comment_id TEXT,         -- ID du commentaire répondu
                response_id TEXT,        -- ID de la réponse postée
                channel_id TEXT,         -- ID de la chaîne de la vidéo
                keyword TEXT,            -- Mot-clé ayant généré la réponse
                response_text TEXT,      -- Texte de la réponse
                posted_at TEXT           -- Timestamp de l'envoi
            )
        """)
    if current_version < 4 and target_version >= 4:
        # Nouvelle table pour les stats de campagne
        cursor.execute("""
                CREATE TABLE IF NOT EXISTS campaign_stats (
                    campaign_id TEXT PRIMARY KEY,
                    total_videos INTEGER,
                    excluded_videos INTEGER,
                    total_comments INTEGER,
                    stop_comments INTEGER,
                    excluded_comments INTEGER,
                    refused_responses INTEGER,
                    posted_responses INTEGER,
                    recorded_at TEXT
                )
            """)
    if current_version < 5 and target_version >= 5:
        # Ajout de la colonne moderation_status à posted_responses
        cursor.execute("""
                ALTER TABLE posted_responses
                ADD COLUMN moderation_status TEXT DEFAULT 'unknown'
            """)
        # Ajout de la colonne moderated_responses à campaign_stats
        cursor.execute("""
                ALTER TABLE campaign_stats
                ADD COLUMN moderated_responses INTEGER DEFAULT 0
            """)


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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posted_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_timestamp TEXT,
            video_id TEXT,
            comment_id TEXT,
            response_id TEXT,
            channel_id TEXT,
            keyword TEXT,
            response_text TEXT,
            posted_at TEXT,
            campaign_id TEXT,
            moderation_status TEXT DEFAULT 'unknown'  -- Nouvelle colonne
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaign_stats (
            campaign_id TEXT PRIMARY KEY,
            total_videos INTEGER,
            excluded_videos INTEGER,
            total_comments INTEGER,
            stop_comments INTEGER,
            excluded_comments INTEGER,
            refused_responses INTEGER,
            posted_responses INTEGER,
            moderated_responses INTEGER DEFAULT 0,  -- Nouvelle colonne
            recorded_at TEXT
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
        ) VALUES (?, ?, ?, ?, ?)
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
    """Sync all videos from the channel into the database without overwriting keywords or transcripts."""
    conn = get_db_connection()
    cursor = conn.cursor()

    videos = youtube_api.get_channel_videos(channel_id)
    for video in videos:
        cursor.execute("""
            INSERT OR REPLACE INTO videos (video_id, url, title, thumbnail_url, description, published_at, status, keywords, transcript)
            VALUES (?, ?, ?, ?, ?, ?, ?,
                COALESCE((SELECT keywords FROM videos WHERE video_id = ?), '[]'),
                COALESCE((SELECT transcript FROM videos WHERE video_id = ?), ''))
        """, (
            video['video_id'],
            video['url'],
            video['title'],
            video['thumbnail'],
            video['description'],
            video['published_at'],
            video['status'],
            video['video_id'],  # Pour COALESCE keywords
            video['video_id']   # Pour COALESCE transcript
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


def get_videos(filter_type: str = "title", keyword: str = "", page: int = 1, per_page: int = 0, keyword_filter: List[str] = None) -> List[Dict[str, Any]]:
    """Récupère les vidéos avec un filtre optionnel par mots-clés. Si per_page = 0, renvoie toutes les vidéos."""
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

    # Pagination : si per_page = 0, on ne met pas de LIMIT
    if per_page > 0:
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


def save_campaign_stats(campaign_id: str, stats: Dict[str, int]):
    """Sauvegarde les statistiques d'une campagne."""
    conn = get_db_connection()
    cursor = conn.cursor()
    recorded_at = datetime.now(pytz.UTC).isoformat()
    cursor.execute("""
        INSERT OR REPLACE INTO campaign_stats (
            campaign_id, total_videos, excluded_videos, total_comments,
            stop_comments, excluded_comments, refused_responses, posted_responses,
            moderated_responses, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        campaign_id, stats['total_videos'], stats['excluded_videos'], stats['total_comments'],
        stats['stop_comments'], stats['excluded_comments'], stats['refused_responses'],
        stats['posted_responses'], stats.get(
            'moderated_responses', 0), recorded_at
    ))
    conn.commit()
    conn.close()


def save_response(campaign_timestamp: str, video_id: str, comment_id: str, response_id: str, channel_id: str, keyword: str, response_text: str, moderation_status: str = "unknown"):
    """Sauvegarde une réponse postée avec statut de modération."""
    conn = get_db_connection()
    cursor = conn.cursor()
    posted_at = datetime.now(pytz.UTC).isoformat()
    cursor.execute("""
        INSERT INTO posted_responses (
            campaign_timestamp, video_id, comment_id, response_id, channel_id,
            keyword, response_text, posted_at, moderation_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (campaign_timestamp, video_id, comment_id, response_id, channel_id, keyword, response_text, posted_at, moderation_status))
    conn.commit()
    conn.close()


def check_existing_response(video_id: str, comment_id: str) -> bool:
    """Vérifie si une réponse existe déjà pour ce commentaire sur cette vidéo."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM posted_responses
        WHERE video_id = ? AND comment_id = ?
    """, (video_id, comment_id))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def get_posted_responses() -> List[Dict[str, Any]]:
    """Récupère toutes les réponses postées."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posted_responses")
    responses = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return responses


def get_response_count_by_video(video_id: str) -> int:
    """Compte le nombre de réponses postées pour une vidéo donnée."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM posted_responses WHERE video_id = ?", (video_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_response_count_by_channel(channel_id: str) -> int:
    """Compte le nombre de réponses postées pour une chaîne donnée."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM posted_responses WHERE channel_id = ?", (channel_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_video_transcript(video_id: str) -> Optional[str]:
    """Récupère le transcript d'une vidéo depuis la base de données."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT transcript FROM videos WHERE video_id = ?", (video_id,))
    result = cursor.fetchone()
    conn.close()
    return result['transcript'] if result and result['transcript'] else None


def save_transcript(video_id: str, transcript: str):
    """Sauvegarde ou met à jour la transcription d'une vidéo dans la base."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE videos SET transcript = ? WHERE video_id = ?
    """, (transcript, video_id))
    # Si la vidéo n'existe pas encore, on l'insère (optionnel, selon ton cas)
    if cursor.rowcount == 0:
        cursor.execute("""
            INSERT INTO videos (video_id, transcript) VALUES (?, ?)
            ON CONFLICT(video_id) DO UPDATE SET transcript = excluded.transcript
        """, (video_id, transcript))
    conn.commit()
    conn.close()


def get_response_moderation_status(video_id: str, comment_id: str) -> str:
    """Récupère le statut de modération d'une réponse postée pour un commentaire donné."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT moderation_status FROM posted_responses
        WHERE video_id = ? AND comment_id = ?
    """, (video_id, comment_id))
    result = cursor.fetchone()
    conn.close()
    return result['moderation_status'] if result else 'unknown'


def delete_video(video_id: str):
    """Supprime une vidéo de la base de données."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    initialize_database()
