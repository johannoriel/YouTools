from urllib.parse import urlparse
import streamlit as st
from duckduckgo_search import DDGS
import random
import requests
from datetime import datetime, timedelta
from langdetect import detect
import yt_dlp
from lib.global_vars import t


def parse_date(date_str, debug=False):
    """Try parsing date in multiple formats"""
    date_formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.0000000"
    ]

    if "+" in date_str or "-" in date_str[-6:]:
        date_str = date_str.rsplit("+", 1)[0].rsplit("-", 1)[0]

    for fmt in date_formats:
        try:
            result = datetime.strptime(date_str, fmt)
            if debug:
                st.write(t("trendwatcher_debug_date").format(
                    date_str=date_str, result=result))
            return result
        except ValueError:
            continue

    if debug:
        st.write(t("trendwatcher_debug_date").format(
            date_str=date_str, result="Failed to parse"))
    return None


def search_videos_duckduckgo(query, keyword, useragents, valid_video_domains, debug=False):
    """Search for recent videos with URL validation using DuckDuckGo"""
    headers = {"User-Agent": random.choice(useragents)}
    ddgs = DDGS(headers=headers)

    video_results = ddgs.videos(
        keywords=query,
        region="fr-fr",
        timelimit="w",
        max_results=5
    )

    results = []
    cutoff_date = datetime.now() - timedelta(days=7)

    for video in video_results:
        try:
            parsed_url = urlparse(video["content"])
            domain = parsed_url.netloc.lower().replace("www.", "")
            if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                st.warning(
                    f"Non-video URL detected: {video['content']} "
                    f"for keyword '{keyword}'. Skipping."
                )
                continue
        except Exception as e:
            st.warning(
                f"Invalid URL: {video['content']} "
                f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
            )
            continue

        published_date = parse_date(video["published"], debug=debug)
        if published_date and published_date > cutoff_date:
            title = video["title"].replace("|", "")
            language = detect(video["title"]) if video["title"] else "unknown"
            is_youtube = domain in ["youtube.com", "youtu.be"]

            results.append({
                "keyword": keyword,
                "url": video["content"],
                "video_id": extract_youtube_id(video["content"]) if is_youtube else "N/A",
                "title": title,
                "view_count": "",
                "language": language,
                "published_at": "",
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "comment_count": "",
                "relevance_score": 0,
                "type": "video"
            })

    return results


def search_texts_duckduckgo(query, keyword, useragents, debug=False):
    """Search for recent text articles using DuckDuckGo"""
    headers = {"User-Agent": random.choice(useragents)}
    ddgs = DDGS(headers=headers)

    text_results = ddgs.text(
        keywords=query,
        region="fr-fr",
        timelimit="w",
        max_results=5
    )

    results = []
    for text in text_results:
        title = text["title"].replace("|", "")
        language = detect(text["title"]) if text["title"] else "unknown"
        if debug:
            st.write(f"Text article: [{title}]({text['href']})")
        results.append({
            "keyword": keyword,
            "url": text["href"],
            "title": title,
            "view_count": "",
            "language": language,
            "published_at": "",
            "channel_id": "N/A",
            "channel_title": "N/A",
            "subscriber_count": "",
            "comment_count": "",
            "relevance_score": 0,
            "type": "web"
        })

    return results


def search_videos_google(query, keyword, useragents, valid_video_domains, api_key, cx_id, debug=False):
    """Search for recent videos using Google Custom Search API"""
    if not api_key or not cx_id:
        st.error(t("trendwatcher_error").format(
            error="Google API key and CX ID are required."))
        return []

    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": f"{query} site:youtube.com",
        "key": api_key,
        "cx": cx_id,
        "num": 5,
        "dateRestrict": "w1"
    }

    try:
        headers = {"User-Agent": random.choice(useragents)}
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code != 200:
            if debug:
                st.error(f"Google API error: {response.text}")
            return []
        data = response.json()

        results = []
        cutoff_date = datetime.now() - timedelta(days=7)

        for item in data.get("items", [])[:5]:
            url = item.get("link", "")
            title = item.get("title", "").replace("|", "")
            if not url or not title:
                continue

            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower().replace("www.", "")
                if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                    if debug:
                        st.warning(
                            f"Non-video URL detected: {url} "
                            f"for keyword '{keyword}'. Skipping."
                        )
                    continue
            except Exception as e:
                if debug:
                    st.warning(
                        f"Invalid URL: {url} "
                        f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
                    )
                continue

            language = detect(title) if title else "unknown"
            is_youtube = domain in ["youtube.com", "youtu.be"]

            results.append({
                "keyword": keyword,
                "url": url,
                "video_id": extract_youtube_id(url) if is_youtube else "N/A",
                "title": title,
                "view_count": "",
                "language": language,
                "published_at": "",
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "comment_count": "",
                "relevance_score": 0,
                "type": "video"
            })

        return results
    except Exception as e:
        if debug:
            st.error(t("trendwatcher_error").format(
                error=f"Google Search error: {str(e)}"))
        return []


def search_texts_google(query, keyword, useragents, api_key, cx_id, debug=False):
    """Search for recent text articles using Google Custom Search API"""
    if not api_key or not cx_id:
        st.error(t("trendwatcher_error").format(
            error="Google API key and CX ID are required."))
        return []

    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cx_id,
        "num": 5,
        "dateRestrict": "w1"
    }

    try:
        headers = {"User-Agent": random.choice(useragents)}
        response = requests.get(base_url, params=params, headers=headers)
        if response.status_code != 200:
            if debug:
                st.error(f"Google API error: {response.text}")
            return []
        data = response.json()

        results = []
        for item in data.get("items", [])[:5]:
            url = item.get("link", "")
            title = item.get("title", "").replace("|", "")
            if not url or not title:
                continue

            language = detect(title) if title else "unknown"
            if debug:
                st.write(f"Text article: [{title}]({url})")
            results.append({
                "keyword": keyword,
                "url": url,
                "title": title,
                "view_count": "",
                "language": language,
                "published_at": "",
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "comment_count": "",
                "relevance_score": 0,
                "type": "web"
            })

        return results
    except Exception as e:
        if debug:
            st.error(t("trendwatcher_error").format(
                error=f"Google Search error: {str(e)}"))
        return []


def search_videos_ytdlp(query, keyword, useragents, valid_video_domains, debug=False):
    """Search for recent videos using yt-dlp"""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "max_downloads": 5,
        "dateafter": (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch5:{query}"
            info = ydl.extract_info(search_query, download=False)

        results = []
        cutoff_date = datetime.now() - timedelta(days=7)

        for entry in info.get("entries", [])[:5]:
            url = entry.get("url", "") or entry.get("webpage_url", "")
            title = entry.get("title", "").replace("|", "")
            if not url or not title:
                continue

            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower().replace("www.", "")
                if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                    if debug:
                        st.warning(
                            f"Non-video URL detected: {url} "
                            f"for keyword '{keyword}'. Skipping."
                        )
                    continue
            except Exception as e:
                if debug:
                    st.warning(
                        f"Invalid URL: {url} "
                        f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
                    )
                continue

            upload_date = entry.get("upload_date", "")
            if upload_date:
                try:
                    published_date = datetime.strptime(upload_date, "%Y%m%d")
                    if published_date <= cutoff_date:
                        continue
                except ValueError:
                    if debug:
                        st.warning(
                            f"Invalid date format for {url}: {upload_date}")
                    continue
            else:
                published_date = datetime.now()

            language = detect(title) if title else "unknown"
            is_youtube = domain in ["youtube.com", "youtu.be"]

            results.append({
                "keyword": keyword,
                "url": url,
                "video_id": entry.get("id", "N/A") if is_youtube else "N/A",
                "title": title,
                "view_count": str(entry.get("view_count", "")),
                "language": language,
                "published_at": upload_date,
                "channel_id": entry.get("channel_id", "N/A"),
                "channel_title": entry.get("uploader", "N/A"),
                "subscriber_count": "",
                "comment_count": "",
                "relevance_score": 0,
                "type": "video"
            })

        return results
    except Exception as e:
        if debug:
            st.error(t("trendwatcher_error").format(
                error=f"yt-dlp error: {str(e)}"))
        return []


def search_texts_ytdlp(query, keyword, useragents, debug=False):
    """Search for text articles using yt-dlp (not supported, returns empty)"""
    if debug:
        st.warning("yt-dlp does not support text article search.")
    return []


def search_videos_searxng(query, keyword, useragents, valid_video_domains, server_url, debug=False):
    """Search for recent videos using SearxNG"""
    if not server_url:
        st.error(t("trendwatcher_error").format(
            error="SearxNG server URL is required."))
        return []

    search_url = f"{server_url.rstrip('/')}/search"
    params = {
        "q": f"{query} site:youtube.com",
        "categories": "general,videos",
        "time_range": "week",
        "format": "json",
        "safesearch": 0,
        "language": "all"
    }

    try:
        headers = {"User-Agent": random.choice(useragents)}
        response = requests.get(search_url, params=params,
                                headers=headers, timeout=10)
        if response.status_code != 200:
            if debug:
                st.error(f"SearxNG API error: {response.text}")
            return []
        data = response.json()

        results = []
        cutoff_date = datetime.now() - timedelta(days=7)

        for item in data.get("results", [])[:5]:
            url = item.get("url", "")
            title = item.get("title", "").replace("|", "")
            if not url or not title:
                continue

            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower().replace("www.", "")
                if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                    if debug:
                        st.warning(
                            f"Non-video URL detected: {url} "
                            f"for keyword '{keyword}'. Skipping."
                        )
                    continue
            except Exception as e:
                if debug:
                    st.warning(
                        f"Invalid URL: {url} "
                        f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
                    )
                continue

            language = detect(title) if title else "unknown"
            is_youtube = domain in ["youtube.com", "youtu.be"]

            results.append({
                "keyword": keyword,
                "url": url,
                "video_id": extract_youtube_id(url) if is_youtube else "N/A",
                "title": title,
                "view_count": "",
                "language": language,
                "published_at": "",
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "comment_count": "",
                "relevance_score": 0,
                "type": "video"
            })

        return results
    except Exception as e:
        if debug:
            st.error(t("trendwatcher_error").format(
                error=f"SearxNG Search error: {str(e)}"))
        return []


def search_texts_searxng(query, keyword, useragents, server_url, debug=False):
    """Search for recent text articles using SearxNG"""
    if not server_url:
        st.error(t("trendwatcher_error").format(
            error="SearxNG server URL is required."))
        return []

    search_url = f"{server_url.rstrip('/')}/search"
    params = {
        "q": query,
        "categories": "general",
        "time_range": "week",
        "format": "json",
        "safesearch": 0,
        "language": "all"
    }

    try:
        headers = {"User-Agent": random.choice(useragents)}
        response = requests.get(search_url, params=params,
                                headers=headers, timeout=10)
        if response.status_code != 200:
            if debug:
                st.error(f"SearxNG API error: {response.text}")
            return []
        data = response.json()

        results = []
        for item in data.get("results", [])[:5]:
            url = item.get("url", "")
            title = item.get("title", "").replace("|", "")
            if not url or not title:
                continue

            language = detect(title) if title else "unknown"
            if debug:
                st.write(f"Text article: [{title}]({url})")
            results.append({
                "keyword": keyword,
                "url": url,
                "title": title,
                "view_count": "",
                "language": language,
                "published_at": "",
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "comment_count": "",
                "relevance_score": 0,
                "type": "web"
            })

        return results
    except Exception as e:
        if debug:
            st.error(t("trendwatcher_error").format(
                error=f"SearxNG Search error: {str(e)}"))
        return []


def search_videos_bing(query, keyword, useragents, valid_video_domains, api_key, debug=False):
    """Search for recent videos using Bing Web Search API"""
    if not api_key:
        st.error(t("trendwatcher_error").format(
            error="Bing API key is required."))
        return []

    search_url = "https://api.bing.microsoft.com/v7.0/search"
    params = {
        "q": f"{query} site:youtube.com",
        "count": 5,
        "freshness": "Week",
        "responseFilter": "Videos,Webpages"
    }
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "User-Agent": random.choice(useragents)
    }

    try:
        response = requests.get(search_url, params=params,
                                headers=headers, timeout=10)
        if response.status_code != 200:
            if debug:
                st.error(f"Bing API error: {response.text}")
            return []
        data = response.json()

        results = []
        cutoff_date = datetime.now() - timedelta(days=7)

        items = data.get("videos", {}).get("value", []) or data.get(
            "webPages", {}).get("value", [])

        for item in items[:5]:
            url = item.get("url") or item.get("contentUrl", "")
            title = item.get("name", "").replace("|", "")
            if not url or not title:
                continue

            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower().replace("www.", "")
                if not any(domain == valid_domain or domain.endswith("." + valid_domain) for valid_domain in valid_video_domains):
                    if debug:
                        st.warning(
                            f"Non-video URL detected: {url} "
                            f"for keyword '{keyword}'. Skipping."
                        )
                    continue
            except Exception as e:
                if debug:
                    st.warning(
                        f"Invalid URL: {url} "
                        f"for keyword '{keyword}'. Error: {str(e)}. Skipping."
                    )
                continue

            date_str = item.get("datePublished", "")
            published_date = parse_date(
                date_str, debug=debug) if date_str else None
            if published_date and published_date <= cutoff_date:
                continue

            language = detect(title) if title else "unknown"
            is_youtube = domain in ["youtube.com", "youtu.be"]

            results.append({
                "keyword": keyword,
                "url": url,
                "video_id": extract_youtube_id(url) if is_youtube else "N/A",
                "title": title,
                "view_count": "",
                "language": language,
                "published_at": "",
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "comment_count": "",
                "relevance_score": 0,
                "type": "video"
            })

        return results
    except Exception as e:
        if debug:
            st.error(t("trendwatcher_error").format(
                error=f"Bing Search error: {str(e)}"))
        return []


def search_texts_bing(query, keyword, useragents, api_key, debug=False):
    """Search for recent text articles using Bing Web Search API"""
    if not api_key:
        st.error(t("trendwatcher_error").format(
            error="Bing API key is required."))
        return []

    search_url = "https://api.bing.microsoft.com/v7.0/search"
    params = {
        "q": query,
        "count": 5,
        "freshness": "Week",
        "responseFilter": "Webpages"
    }
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "User-Agent": random.choice(useragents)
    }

    try:
        response = requests.get(search_url, params=params,
                                headers=headers, timeout=10)
        if response.status_code != 200:
            if debug:
                st.error(f"Bing API error: {response.text}")
            return []
        data = response.json()

        results = []
        for item in data.get("webPages", {}).get("value", [])[:5]:
            url = item.get("url", "")
            title = item.get("name", "").replace("|", "")
            if not url or not title:
                continue

            language = detect(title) if title else "unknown"
            if debug:
                st.write(f"Text article: [{title}]({url})")
            results.append({
                "keyword": keyword,
                "url": url,
                "title": title,
                "view_count": "",
                "language": language,
                "published_at": "",
                "channel_id": "N/A",
                "channel_title": "N/A",
                "subscriber_count": "",
                "comment_count": "",
                "relevance_score": 0,
                "type": "web"
            })

        return results
    except Exception as e:
        if debug:
            st.error(t("trendwatcher_error").format(
                error=f"Bing Search error: {str(e)}"))
        return []


def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    import re
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:watch\?v=)([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return "N/A"
