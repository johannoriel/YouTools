import sqlite3
from datetime import datetime
import pytz
from typing import List, Dict, Any, Optional
from youtube_api import YoutubeAPI

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
            status TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stats_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            timestamp TEXT,
            view_count INTEGER,
            subscribers_gained INTEGER,
            subscribers_lost INTEGER,
            retention_rate REAL,
            avg_view_duration REAL,
            annotation_click_through_rate REAL,
            annotation_close_rate REAL,
            average_view_percentage REAL,
            comments INTEGER,
            dislikes INTEGER,
            estimated_minutes_watched REAL,
            estimated_ad_revenue REAL,
            likes INTEGER,
            shares INTEGER,
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

    conn.commit()
    conn.close()


def sync_stats(self, channel_id: str, progress_callback=None):
    """Sync stats for all videos with progress callback."""
    videos = self.get_channel_videos(channel_id)
    total_videos = len(videos)
    timestamp = datetime.now(pytz.UTC).isoformat()

    conn = get_db_connection()
    cursor = conn.cursor()

    for i, video in enumerate(videos):
        stats = self.get_advanced_video_stats(video['video_id'])
        if stats:
            cursor.execute("""
                INSERT INTO stats_snapshots (
                    video_id, timestamp, view_count, subscribers_gained, subscribers_lost, retention_rate, avg_view_duration,
                    annotation_click_through_rate, annotation_close_rate, average_view_percentage, comments, dislikes,
                    estimated_minutes_watched, estimated_ad_revenue, likes, shares
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video['video_id'],
                timestamp,
                stats['view_count'],
                stats['subscribers_gained'],
                stats['subscribers_lost'],
                stats['retention_rate'],
                stats['avg_view_duration'],
                stats['annotation_click_through_rate'],
                stats['annotation_close_rate'],
                stats['average_view_percentage'],
                stats['comments'],
                stats['dislikes'],
                stats['estimated_minutes_watched'],
                stats['estimated_ad_revenue'],
                stats['likes'],
                stats['shares']
            ))
        if progress_callback:
            progress_callback((i + 1) / total_videos)

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
    return dict(stats) if stats else None


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


def get_videos(filter_type: str = "title", keyword: str = "", page: int = 1, per_page: int = 100) -> List[Dict[str, Any]]:
    """Retrieve filtered and paginated videos from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query_base = "SELECT * FROM videos WHERE "
    if filter_type == "title":
        query = query_base + "title LIKE ?"
    elif filter_type == "title_description":
        query = query_base + "(title LIKE ? OR description LIKE ?)"
    elif filter_type == "all":
        query = query_base + \
            "(title LIKE ? OR description LIKE ? OR transcript LIKE ?)"
    else:
        query = "SELECT * FROM videos"

    keyword_param = f"%{keyword}%"
    params = [keyword_param] * (1 if filter_type ==
                                "title" else 2 if filter_type == "title_description" else 3)

    # Pagination
    offset = (page - 1) * per_page
    query += " LIMIT ? OFFSET ?"
    params.extend([per_page, offset])

    cursor.execute(query, params)
    videos = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return videos


if __name__ == "__main__":
    initialize_database()
