import asyncio
import hashlib
import json
import os
import tempfile
import re
import uuid
from typing import Literal
from difflib import SequenceMatcher
from urllib.parse import quote, urlencode

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

SLSKD_URL = os.getenv("SLSKD_URL", "http://slskd:5030").rstrip("/")
API_KEY = os.getenv("SLSKD_API_KEY", "")

LEGACY_SEARCH_TIMEOUT_MS = int(os.getenv("SEARCH_TIMEOUT_MS", "25000"))
SEARCH_DURATION_SECONDS = int(os.getenv(
    "SEARCH_DURATION_SECONDS",
    str(max(5, round(LEGACY_SEARCH_TIMEOUT_MS / 1000))),
))
SEARCH_WAIT_GRACE_SECONDS = int(os.getenv("SEARCH_WAIT_GRACE_SECONDS", "5"))
DEEP_SEARCH_DURATION_SECONDS = int(os.getenv("DEEP_SEARCH_DURATION_SECONDS", "60"))
DEEP_SEARCH_MIN_MATCH_SCORE = float(os.getenv("DEEP_SEARCH_MIN_MATCH_SCORE", "0.50"))
RETRY_NOT_FOUND_BY_DEFAULT = os.getenv("RETRY_NOT_FOUND_BY_DEFAULT", "true").lower() in {"1", "true", "yes", "on"}
MAX_PARALLEL_SEARCHES_LIMIT = 20
DEFAULT_MAX_PARALLEL_SEARCHES = max(1, min(MAX_PARALLEL_SEARCHES_LIMIT, int(os.getenv("MAX_PARALLEL_SEARCHES", "5"))))
SEARCH_START_INTERVAL_SECONDS = max(1.0, float(os.getenv("SEARCH_START_INTERVAL_SECONDS", "1")))

POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1"))
MIN_MATCH_SCORE = float(os.getenv("MIN_MATCH_SCORE", "0.52"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS_PER_SONG", "100"))
DEEP_SEARCH_RESPONSE_LIMIT = int(os.getenv("DEEP_SEARCH_RESPONSE_LIMIT", str(max(MAX_RESULTS, 250))))
TRANSFER_POLL_INTERVAL = float(os.getenv("TRANSFER_POLL_INTERVAL", "2"))
TRANSFER_MONITOR_HOURS = float(os.getenv("TRANSFER_MONITOR_HOURS", "24"))
MAX_429_RETRIES = int(os.getenv("SLSKD_429_RETRIES", "5"))

# Serialize short write requests, not their full search lifecycles. This allows
# active searches to overlap while their POST /searches calls are rate-spaced.
SLSKD_POST_LOCK = asyncio.Lock()
LAST_SLSKD_POST_AT = 0.0
ALLOWED_EXTENSIONS = {".mp3", ".flac"}

app = FastAPI(title="slskd Song Queue", version="2.8.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
BATCHES = {}
CURRENT_BATCH_ID = None
BATCH_START_LOCK = asyncio.Lock()


class BatchCancelled(Exception):
    """Raised when the user interrupts a batch."""


class Song(BaseModel):
    artist: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)


class SongList(BaseModel):
    songs: list[Song] = Field(min_length=1, max_length=100)
    search_duration_seconds: int = Field(default=SEARCH_DURATION_SECONDS, ge=5, le=180)
    retry_not_found: bool = Field(default=RETRY_NOT_FOUND_BY_DEFAULT)
    deep_search_duration_seconds: int = Field(default=DEEP_SEARCH_DURATION_SECONDS, ge=10, le=300)
    max_parallel_searches: int = Field(default=DEFAULT_MAX_PARALLEL_SEARCHES, ge=1, le=MAX_PARALLEL_SEARCHES_LIMIT)


class LastFMAuthStart(BaseModel):
    api_key: str = Field(min_length=8, max_length=128)
    api_secret: str = Field(min_length=8, max_length=128)


class LastFMAuthComplete(LastFMAuthStart):
    token: str = Field(min_length=8, max_length=256)


class LastFMRequest(BaseModel):
    mode: Literal[
        "personal_recommended_tracks",
        "history_mix",
        "artist_discovery",
        "tag_discovery",
    ]
    artist: str | None = Field(default=None, max_length=200)
    tag: str | None = Field(default=None, max_length=100)
    history_basis: Literal["recent", "top"] = "recent"
    period: Literal["7day", "1month", "3month", "6month", "12month", "overall"] = "3month"
    count: int = Field(default=40, ge=5, le=100)

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")


@app.get("/settings")
async def get_settings():
    return {
        "search_duration_seconds": SEARCH_DURATION_SECONDS,
        "retry_not_found": RETRY_NOT_FOUND_BY_DEFAULT,
        "deep_search_duration_seconds": DEEP_SEARCH_DURATION_SECONDS,
        "max_parallel_searches": DEFAULT_MAX_PARALLEL_SEARCHES,
        "search_start_interval_seconds": SEARCH_START_INTERVAL_SECONDS,
    }


def normalize(value: str) -> str:
    value = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", value.lower())
    value = re.sub(r"[^a-z0-9а-яё]+", " ", value)
    return " ".join(value.split())


def file_extension(filename: str) -> str:
    normalized_path = filename.replace("\\", "/")
    return os.path.splitext(normalized_path)[1].lower()


def format_priority(filename: str) -> int:
    return {".mp3": 2, ".flac": 1}.get(file_extension(filename), 0)


def match_score(song: Song, filename: str, bitrate: int = 0) -> float:
    leaf_name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    leaf_name = os.path.splitext(leaf_name)[0]
    target = normalize(f"{song.artist} {song.title}")
    candidate = normalize(leaf_name)
    score = SequenceMatcher(None, target, candidate).ratio()
    artist, title = normalize(song.artist), normalize(song.title)
    if artist and artist in candidate:
        score += 0.22
    if title and title in candidate:
        score += 0.28
    if bitrate >= 320:
        score += 0.03
    return min(score, 1.0)


def update_item(batch_id: str, index: int, **values):
    BATCHES[batch_id]["results"][index].update(values)


async def post_with_429_retry(client, path: str, *, json_body, cancel_event=None):
    """Serialize POSTs and leave a minimum one-second gap between their starts."""
    global LAST_SLSKD_POST_AT
    response = None
    async with SLSKD_POST_LOCK:
        loop = asyncio.get_running_loop()
        for attempt in range(MAX_429_RETRIES + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise BatchCancelled()

            wait_for_gap = SEARCH_START_INTERVAL_SECONDS - (loop.time() - LAST_SLSKD_POST_AT)
            if wait_for_gap > 0:
                await asyncio.sleep(wait_for_gap)

            if cancel_event is not None and cancel_event.is_set():
                raise BatchCancelled()

            LAST_SLSKD_POST_AT = loop.time()
            response = await client.post(path, json=json_body)
            if response.status_code != 429:
                response.raise_for_status()
                return response

            if attempt >= MAX_429_RETRIES:
                break
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(2 ** attempt, 8)
            except ValueError:
                delay = min(2 ** attempt, 8)
            await asyncio.sleep(max(1.0, min(delay, 10.0)))

    response.raise_for_status()
    return response


async def stop_search(client, search_id: str):
    """Cancel one active slskd search without racing other write requests."""
    global LAST_SLSKD_POST_AT
    response = None
    async with SLSKD_POST_LOCK:
        loop = asyncio.get_running_loop()
        for attempt in range(MAX_429_RETRIES + 1):
            wait_for_gap = SEARCH_START_INTERVAL_SECONDS - (loop.time() - LAST_SLSKD_POST_AT)
            if wait_for_gap > 0:
                await asyncio.sleep(wait_for_gap)
            LAST_SLSKD_POST_AT = loop.time()
            response = await client.delete(f"/api/v0/searches/{search_id}")
            if response.status_code in {200, 202, 204, 404}:
                return
            if response.status_code != 429 or attempt >= MAX_429_RETRIES:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(2 ** attempt, 8)
            except ValueError:
                delay = min(2 ** attempt, 8)
            await asyncio.sleep(max(1.0, min(delay, 10.0)))
    response.raise_for_status()


async def search_slskd(client, batch_id: str, query: str, duration_seconds: int, response_limit: int):
    """Start a search and poll it without locking out other active searches."""
    batch = BATCHES[batch_id]
    cancel_event = batch["cancel_event"]
    if cancel_event.is_set():
        raise BatchCancelled()

    search_id = str(uuid.uuid4())
    await post_with_429_retry(
        client,
        "/api/v0/searches",
        json_body={
            "id": search_id,
            "searchText": query,
            "searchTimeout": duration_seconds * 1000,
            "responseLimit": response_limit,
        },
        cancel_event=cancel_event,
    )

    batch["active_searches"].add(search_id)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration_seconds + SEARCH_WAIT_GRACE_SECONDS
    completed = False
    try:
        if cancel_event.is_set():
            raise BatchCancelled()

        while loop.time() < deadline:
            if cancel_event.is_set():
                raise BatchCancelled()
            state_response = await client.get(f"/api/v0/searches/{search_id}")
            state_response.raise_for_status()
            if state_response.json().get("isComplete") is True:
                completed = True
                break
            await asyncio.sleep(POLL_INTERVAL)

        if cancel_event.is_set():
            raise BatchCancelled()

        results_response = await client.get(f"/api/v0/searches/{search_id}/responses")
        results_response.raise_for_status()
        payload = results_response.json()
        if cancel_event.is_set():
            raise BatchCancelled()
        return payload
    finally:
        # Stop searches that timed out locally or were interrupted by the user.
        if not completed or cancel_event.is_set():
            try:
                await stop_search(client, search_id)
            except httpx.HTTPError:
                pass
        batch["active_searches"].discard(search_id)


def extract_audio_candidates(song: Song, payload, minimum_score: float):
    peers = payload if isinstance(payload, list) else payload.get("responses", []) if isinstance(payload, dict) else []
    candidates = []
    for peer in peers:
        username = peer.get("username")
        for file in peer.get("files", []) or []:
            filename = file.get("filename") or file.get("fileName")
            if not username or not filename:
                continue

            extension = file_extension(filename)
            if extension not in ALLOWED_EXTENSIONS:
                continue

            try:
                bitrate = int(file.get("bitRate") or file.get("bitrate") or 0)
            except (TypeError, ValueError):
                bitrate = 0

            score = match_score(song, filename, bitrate)
            if score < minimum_score:
                continue

            candidates.append({
                "username": username,
                "filename": filename,
                "size": file.get("size", 0),
                "bitrate": bitrate,
                "score": score,
                "format_priority": format_priority(filename),
                "format": extension.lstrip(".").upper(),
            })

    candidates.sort(
        key=lambda item: (item["format_priority"], item["score"], item["bitrate"]),
        reverse=True,
    )
    return candidates


def deep_search_queries(song: Song):
    title = song.title.strip()
    clean_title = re.sub(r"\s+(?:feat\.?|ft\.?|featuring)\s+.+$", "", title, flags=re.IGNORECASE)
    clean_title = re.sub(r"\s*[\[(].*?[\])]\s*", " ", clean_title)
    clean_title = " ".join(clean_title.split())

    queries = [f"{song.artist} {clean_title}".strip(), clean_title]
    unique = []
    seen = set()
    for query in queries:
        key = normalize(query)
        if key and key not in seen:
            seen.add(key)
            unique.append(query)
    return unique


async def get_download_files(client):
    response = await client.get("/api/v0/transfers/downloads")
    response.raise_for_status()
    payload = response.json()
    files = []
    for peer in payload if isinstance(payload, list) else []:
        username = peer.get("username")
        for directory in peer.get("directories", []) or []:
            for file in directory.get("files", []) or []:
                files.append({**file, "username": username})
    return files


def transfer_status(file):
    raw_state = str(file.get("state", "")).lower()
    state = re.sub(r"[^a-z]", "", raw_state)
    try:
        percent = float(file.get("percentComplete") or 0)
    except (TypeError, ValueError):
        percent = 0

    if any(word in state for word in ("errored", "failed", "error", "rejected", "timedout")):
        return "failed"
    if "cancelled" in state or "canceled" in state:
        return "cancelled"
    if "succeeded" in state or "completed" in state or percent >= 100:
        return "downloaded"
    if "paused" in state:
        return "paused"
    if any(word in state for word in ("inprogress", "transferring", "downloading")):
        return "downloading"
    if any(word in state for word in ("initializing", "connecting", "requested")):
        return "connecting"
    return "queued"


async def monitor_transfer(batch_id, index, username, filename):
    deadline = asyncio.get_running_loop().time() + TRANSFER_MONITOR_HOURS * 3600
    async with httpx.AsyncClient(
        base_url=SLSKD_URL,
        headers={"X-API-Key": API_KEY},
        timeout=20,
    ) as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                files = await get_download_files(client)
                found = next(
                    (item for item in files
                     if item.get("username") == username and item.get("filename") == filename),
                    None,
                )
                if found:
                    status = transfer_status(found)
                    update_item(
                        batch_id, index,
                        status=status,
                        percent=round(float(found.get("percentComplete", 0) or 0), 1),
                        bytes_transferred=found.get("bytesTransferred", 0),
                        size=found.get("size", 0),
                        speed=found.get("averageSpeed", 0),
                        remaining=found.get("bytesRemaining", 0),
                    )
                    if status in {"downloaded", "failed", "cancelled"}:
                        return
            except (httpx.HTTPError, ValueError, TypeError):
                pass
            await asyncio.sleep(TRANSFER_POLL_INTERVAL)

    if BATCHES[batch_id]["results"][index]["status"] in {"queued", "connecting", "downloading", "paused"}:
        update_item(
            batch_id, index,
            status="monitoring_timeout",
            message="Истекло время наблюдения; проверьте состояние в slskd",
        )


def mark_cancelled(batch_id: str, index: int, attempt: int):
    item = BATCHES[batch_id]["results"][index]
    if item["status"] not in {"queued", "connecting", "downloading", "paused", "downloaded"}:
        update_item(
            batch_id, index,
            status="cancelled",
            search_attempt=attempt,
            message="Прервано пользователем",
        )


async def process_song(
    batch_id: str,
    index: int,
    song: Song,
    search_duration_seconds: int,
    deep_search_duration_seconds: int,
    deep_search: bool = False,
    serial_retry: bool = False,
):
    batch = BATCHES[batch_id]
    cancel_event = batch["cancel_event"]
    attempt = 3 if deep_search and batch["results"][index].get("search_attempt", 1) >= 2 else (2 if deep_search or serial_retry else 1)
    threshold = DEEP_SEARCH_MIN_MATCH_SCORE if deep_search else MIN_MATCH_SCORE

    if deep_search:
        queries = deep_search_queries(song)
        duration = deep_search_duration_seconds
        response_limit = DEEP_SEARCH_RESPONSE_LIMIT
        status_message = f"Повторный расширенный поиск ({duration} сек.)"
    else:
        queries = [f"{song.artist} {song.title}"]
        duration = search_duration_seconds
        response_limit = MAX_RESULTS
        status_message = (
            f"Контрольный поиск без параллельного режима ({duration} сек.)"
            if serial_retry else f"Первичный поиск ({duration} сек.)"
        )

    update_item(
        batch_id, index,
        status="pending",
        search_attempt=attempt,
        message="Ожидает свободного слота поиска",
    )

    try:
        # The batch semaphore enforces the configured active-search limit.
        # A serial confirmation pass is used for first-pass misses because
        # slskd can occasionally complete concurrent searches with zero results.
        semaphore = (
            batch["serial_search_semaphore"]
            if serial_retry else batch["search_semaphore"]
        )
        async with semaphore:
            if cancel_event.is_set():
                raise BatchCancelled()

            update_item(
                batch_id, index,
                status="searching",
                search_attempt=attempt,
                message=status_message,
            )

            async with httpx.AsyncClient(
                base_url=SLSKD_URL,
                headers={"X-API-Key": API_KEY},
                timeout=30,
            ) as client:
                candidates_by_key = {}
                for query_index, query in enumerate(queries, start=1):
                    if cancel_event.is_set():
                        raise BatchCancelled()
                    update_item(
                        batch_id, index,
                        status="searching",
                        search_attempt=attempt,
                        message=(
                            f"Расширенный поиск: запрос {query_index}/{len(queries)}"
                            if deep_search
                            else status_message
                        ),
                    )
                    payload = await search_slskd(
                        client, batch_id, query, duration, response_limit
                    )
                    found = extract_audio_candidates(song, payload, threshold)
                    for candidate in found:
                        key = (candidate["username"], candidate["filename"])
                        previous = candidates_by_key.get(key)
                        if previous is None or candidate["score"] > previous["score"]:
                            candidates_by_key[key] = candidate

                    if deep_search and any(
                        candidate["format"] == "MP3"
                        for candidate in candidates_by_key.values()
                    ):
                        break

                if cancel_event.is_set():
                    raise BatchCancelled()

                candidates = sorted(
                    candidates_by_key.values(),
                    key=lambda item: (item["format_priority"], item["score"], item["bitrate"]),
                    reverse=True,
                )

                if not candidates:
                    update_item(
                        batch_id, index,
                        status="not_found",
                        search_attempt=attempt,
                        message=(
                            "Не найдено даже после расширенного поиска"
                            if deep_search else "Подходящий MP3/FLAC не найден"
                        ),
                        matches=0,
                    )
                    return None

                best = candidates[0]
                update_item(
                    batch_id, index,
                    status="adding",
                    search_attempt=attempt,
                    filename=best["filename"],
                    peer=best["username"],
                    size=best["size"],
                    match_score=round(best["score"], 3),
                    audio_format=best["format"],
                    message=f"Добавляем {best['format']} в очередь slskd",
                )

                await post_with_429_retry(
                    client,
                    f"/api/v0/transfers/downloads/{quote(best['username'], safe='')}",
                    json_body=[{"filename": best["filename"], "size": best["size"]}],
                    cancel_event=cancel_event,
                )

                update_item(
                    batch_id, index,
                    status="queued",
                    message=f"{best['format']} добавлен в очередь slskd",
                )

        return asyncio.create_task(
            monitor_transfer(batch_id, index, best["username"], best["filename"])
        )
    except BatchCancelled:
        mark_cancelled(batch_id, index, attempt)
    except httpx.HTTPStatusError as exc:
        update_item(
            batch_id, index,
            status="slskd_error",
            message=f"slskd HTTP {exc.response.status_code}",
            details=exc.response.text[:400],
            search_attempt=attempt,
        )
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        update_item(
            batch_id, index,
            status="error",
            message="Не удалось связаться со slskd",
            details=str(exc)[:400],
            search_attempt=attempt,
        )
    except Exception as exc:
        update_item(
            batch_id, index,
            status="error",
            message="Внутренняя ошибка",
            details=str(exc)[:400],
            search_attempt=attempt,
        )
    return None


async def run_batch(batch_id):
    global CURRENT_BATCH_ID
    batch = BATCHES[batch_id]
    settings = batch["settings"]
    monitor_tasks = []

    try:
        batch["phase"] = "initial_search"
        first_pass = await asyncio.gather(*(
            process_song(
                batch_id,
                index,
                item["song_obj"],
                settings["search_duration_seconds"],
                settings["deep_search_duration_seconds"],
                deep_search=False,
            )
            for index, item in enumerate(batch["results"])
        ))
        monitor_tasks.extend(task for task in first_pass if task is not None)

        # If a batch used parallel searching and some searches came back empty,
        # retry those exact queries one at a time before broadening the query.
        if (
            not batch["cancel_event"].is_set()
            and settings["max_parallel_searches"] > 1
        ):
            parallel_misses = [
                (index, item["song_obj"])
                for index, item in enumerate(batch["results"])
                if item["status"] == "not_found"
            ]
            if parallel_misses:
                batch["serial_retry_count"] = len(parallel_misses)
                batch["phase"] = "serial_retry"
                serial_pass = await asyncio.gather(*(
                    process_song(
                        batch_id,
                        index,
                        song,
                        settings["search_duration_seconds"],
                        settings["deep_search_duration_seconds"],
                        deep_search=False,
                        serial_retry=True,
                    )
                    for index, song in parallel_misses
                ))
                monitor_tasks.extend(task for task in serial_pass if task is not None)

        if not batch["cancel_event"].is_set() and settings["retry_not_found"]:
            missing = [
                (index, item["song_obj"])
                for index, item in enumerate(batch["results"])
                if item["status"] == "not_found"
            ]
            if missing:
                batch["phase"] = "deep_search"
                batch["retry_count"] = len(missing)
                second_pass = await asyncio.gather(*(
                    process_song(
                        batch_id,
                        index,
                        song,
                        settings["search_duration_seconds"],
                        settings["deep_search_duration_seconds"],
                        deep_search=True,
                    )
                    for index, song in missing
                ))
                monitor_tasks.extend(task for task in second_pass if task is not None)

        batch["search_finished"] = True
        batch["phase"] = (
            "cancelled_waiting_for_transfers"
            if batch["cancel_event"].is_set()
            else "waiting_for_transfers"
        )
        if CURRENT_BATCH_ID == batch_id:
            CURRENT_BATCH_ID = None
        batch["search_done_event"].set()
        if monitor_tasks:
            await asyncio.gather(*monitor_tasks, return_exceptions=True)
    finally:
        batch["search_finished"] = True
        batch["finished"] = True
        batch["phase"] = "cancelled" if batch["cancel_event"].is_set() else "finished"
        if CURRENT_BATCH_ID == batch_id:
            CURRENT_BATCH_ID = None
        batch["search_done_event"].set()
        for item in batch["results"]:
            item.pop("song_obj", None)


def lastfm_as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def lastfm_artist_name(value):
    if isinstance(value, dict):
        return str(value.get("name") or value.get("#text") or "").strip()
    return str(value or "").strip()


def lastfm_track_item(raw, rank, fallback_artist=""):
    if not isinstance(raw, dict):
        return None
    artist = lastfm_artist_name(raw.get("artist")) or fallback_artist
    title = str(raw.get("name") or raw.get("title") or "").strip()
    if not artist or not title:
        return None
    date = raw.get("date")
    if isinstance(date, dict):
        date = date.get("#text") or date.get("uts")
    attributes = raw.get("@attr") if isinstance(raw.get("@attr"), dict) else {}
    return {
        "kind": "track",
        "artist": artist,
        "title": title,
        "album": lastfm_artist_name(raw.get("album")),
        "rank": rank,
        "match": raw.get("match"),
        "playcount": raw.get("playcount"),
        "listeners": raw.get("listeners"),
        "date": date,
        "nowplaying": str(attributes.get("nowplaying", "")).lower() == "true",
        "loved": raw.get("loved"),
        "url": raw.get("url"),
    }


def lastfm_artist_item(raw, rank):
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or raw.get("#text") or "").strip()
    if not name:
        return None
    return {
        "kind": "artist",
        "name": name,
        "rank": rank,
        "match": raw.get("match"),
        "playcount": raw.get("playcount"),
        "listeners": raw.get("listeners"),
        "url": raw.get("url"),
    }


def lastfm_album_item(raw, rank):
    if not isinstance(raw, dict):
        return None
    artist = lastfm_artist_name(raw.get("artist"))
    title = str(raw.get("name") or raw.get("title") or "").strip()
    if not artist or not title:
        return None
    return {
        "kind": "album",
        "artist": artist,
        "title": title,
        "rank": rank,
        "playcount": raw.get("playcount"),
        "listeners": raw.get("listeners"),
        "url": raw.get("url"),
    }


def lastfm_tag_item(raw, rank):
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or raw.get("#text") or "").strip()
    if not name:
        return None
    return {
        "kind": "tag",
        "name": name,
        "rank": rank,
        "count": raw.get("count") or raw.get("taggings") or raw.get("reach"),
        "url": raw.get("url"),
    }

def lastfm_http_error_detail(response):
    body = response.text.strip()
    try:
        payload = response.json()
        if isinstance(payload, dict):
            if payload.get("error") is not None:
                body = f"API {payload.get('error')}: {payload.get('message') or 'ошибка API'}"
            elif payload.get("message"):
                body = str(payload["message"])
    except (ValueError, TypeError):
        pass
    diagnostic = f": {body[:400]}" if body else ""
    return f"Last.fm вернул HTTP {response.status_code}{diagnostic}"


async def lastfm_call(client, api_key: str, method: str, params: dict):
    query = {
        "method": method,
        "api_key": api_key,
        "format": "json",
        **params,
    }
    try:
        response = await client.get("/2.0/", params=query)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=lastfm_http_error_detail(exc.response),
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось получить ответ Last.fm: {str(exc)[:200]}",
        ) from exc

    if isinstance(payload, dict) and payload.get("error") is not None:
        error_code = payload.get("error")
        error_message = payload.get("message") or "Unknown Last.fm API error"
        raise HTTPException(
            status_code=502,
            detail=f"Last.fm API {error_code}: {error_message}",
        )
    return payload if isinstance(payload, dict) else {}


LASTFM_CREDENTIALS_PATH = os.getenv("LASTFM_CREDENTIALS_PATH", "/data/lastfm_credentials.json")


def load_lastfm_credentials():
    try:
        with open(LASTFM_CREDENTIALS_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if all(payload.get(key) for key in ("api_key", "api_secret", "session_key", "username")):
            return payload
    except (OSError, ValueError, TypeError):
        pass
    return {}


LASTFM_CREDENTIALS = load_lastfm_credentials()


def save_lastfm_credentials(credentials):
    directory = os.path.dirname(LASTFM_CREDENTIALS_PATH) or "."
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".lastfm-", suffix=".tmp", dir=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(credentials, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, LASTFM_CREDENTIALS_PATH)
        os.chmod(LASTFM_CREDENTIALS_PATH, 0o600)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def clear_lastfm_credentials():
    LASTFM_CREDENTIALS.clear()
    try:
        os.remove(LASTFM_CREDENTIALS_PATH)
    except FileNotFoundError:
        pass


def lastfm_signature(params, api_secret):
    signature_text = "".join(
        f"{key}{params[key]}"
        for key in sorted(params)
        if key not in {"format", "callback", "api_sig"}
    ) + api_secret
    return hashlib.md5(signature_text.encode("utf-8")).hexdigest()


async def lastfm_request(client, params, *, api_secret=None):
    query = {**params, "format": "json"}
    if api_secret:
        query["api_sig"] = lastfm_signature(params, api_secret)
    try:
        response = await client.get("/2.0/", params=query)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=lastfm_http_error_detail(exc.response),
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Не удалось получить ответ Last.fm: {str(exc)[:200]}",
        ) from exc
    if isinstance(payload, dict) and payload.get("error") is not None:
        raise HTTPException(
            status_code=502,
            detail=f"Last.fm API {payload.get('error')}: {payload.get('message') or 'ошибка API'}",
        )
    return payload if isinstance(payload, dict) else {}


async def lastfm_public_call(client, api_key, method, params=None):
    return await lastfm_request(
        client,
        {"method": method, "api_key": api_key, **(params or {})},
    )


async def lastfm_signed_call(client, credentials, method, params=None):
    signed = {
        "method": method,
        "api_key": credentials["api_key"],
        "sk": credentials["session_key"],
        **(params or {}),
    }
    return await lastfm_request(client, signed, api_secret=credentials["api_secret"])


def recommendation_match(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def add_recommendation(bucket, raw, *, source, weight=1.0, match=None):
    item = lastfm_track_item(raw, 0)
    if not item:
        return
    key = (normalize(item["artist"]), normalize(item["title"]))
    score = recommendation_match(match if match is not None else raw.get("match"))
    entry = bucket.setdefault(key, {
        **item,
        "score": 0.0,
        "support": 0,
        "reasons": [],
        "_sources": set(),
    })
    entry["score"] += weight * (0.35 + 0.65 * score)
    if source not in entry["_sources"]:
        entry["_sources"].add(source)
        entry["support"] += 1
        entry["reasons"].append(source)


async def add_artist_top_tracks(
    client, api_key, artist_name, bucket, *, source, weight=1.0, limit=5
):
    payload = await lastfm_public_call(
        client, api_key, "artist.getTopTracks",
        {"artist": artist_name, "autocorrect": 1, "limit": limit},
    )
    raw_items = lastfm_as_list((payload.get("toptracks") or {}).get("track"))
    for raw in raw_items:
        add_recommendation(
            bucket, raw, source=source, weight=weight,
            match=raw.get("match", 0.5),
        )
    await asyncio.sleep(0.2)


def finalize_recommendations(bucket, count):
    entries = sorted(
        bucket.values(),
        key=lambda item: (
            item["support"],
            item["score"],
            int(item.get("listeners") or 0) if str(item.get("listeners") or "").isdigit() else 0,
        ),
        reverse=True,
    )
    results = []
    for rank, item in enumerate(entries[:count], start=1):
        item["rank"] = rank
        item["recommendation_score"] = round(item.pop("score"), 3)
        item.pop("_sources", None)
        item["recommendation_reason"] = " • ".join(item.pop("reasons"))
        item["support"] = int(item["support"])
        results.append(item)
    return results


@app.post("/lastfm/auth/token")
async def lastfm_auth_token(request: LastFMAuthStart):
    async with httpx.AsyncClient(
        base_url="https://ws.audioscrobbler.com",
        headers={"User-Agent": "slskd-song-queue/2.8.0"},
        timeout=25,
    ) as client:
        payload = await lastfm_request(
            client,
            {"method": "auth.getToken", "api_key": request.api_key},
            api_secret=request.api_secret,
        )
    token = str(payload.get("token") or "").strip()
    if not token:
        raise HTTPException(status_code=502, detail="Last.fm не вернул токен авторизации.")
    authorize_url = "https://www.last.fm/api/auth/?" + urlencode({
        "api_key": request.api_key,
        "token": token,
    })
    return {"token": token, "authorize_url": authorize_url}


@app.post("/lastfm/auth/session")
async def lastfm_auth_session(request: LastFMAuthComplete):
    async with httpx.AsyncClient(
        base_url="https://ws.audioscrobbler.com",
        headers={"User-Agent": "slskd-song-queue/2.8.0"},
        timeout=25,
    ) as client:
        payload = await lastfm_request(
            client,
            {
                "method": "auth.getSession",
                "api_key": request.api_key,
                "token": request.token,
            },
            api_secret=request.api_secret,
        )
    session = payload.get("session") or {}
    session_key = str(session.get("key") or "").strip()
    username = str(session.get("name") or "").strip()
    if not session_key or not username:
        raise HTTPException(
            status_code=502,
            detail="Last.fm не вернул session key и имя пользователя.",
        )
    credentials = {
        "api_key": request.api_key,
        "api_secret": request.api_secret,
        "session_key": session_key,
        "username": username,
    }
    try:
        save_lastfm_credentials(credentials)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Не удалось безопасно сохранить подключение Last.fm в /data.",
        ) from exc
    LASTFM_CREDENTIALS.clear()
    LASTFM_CREDENTIALS.update(credentials)
    return {"connected": True, "username": username}


@app.get("/lastfm/auth/status")
async def lastfm_auth_status():
    return {
        "connected": bool(LASTFM_CREDENTIALS.get("session_key")),
        "username": LASTFM_CREDENTIALS.get("username"),
    }


@app.delete("/lastfm/auth/session")
async def lastfm_auth_disconnect():
    clear_lastfm_credentials()
    return {"connected": False}


@app.post("/lastfm/recommendations")
async def lastfm_recommendations(request: LastFMRequest):
    credentials = LASTFM_CREDENTIALS
    if not credentials.get("api_key") or not credentials.get("session_key"):
        raise HTTPException(
            status_code=401,
            detail="Сначала подключите аккаунт Last.fm через API key и shared secret.",
        )
    username = credentials["username"]
    api_key = credentials["api_key"]
    bucket = {}
    label = ""

    async with httpx.AsyncClient(
        base_url="https://ws.audioscrobbler.com",
        headers={"User-Agent": "slskd-song-queue/2.8.0"},
        timeout=25,
    ) as client:
        if request.mode in {"personal_recommended_tracks", "history_mix"}:
            personal_recommendations = request.mode == "personal_recommended_tracks"
            if personal_recommendations:
                label = f"Персональные рекомендации {username} по топ-трекам за 3 месяца"
                history_basis = "top"
                history_period = "3month"
            else:
                label = f"Рекомендации на основе прослушиваний {username}"
                history_basis = request.history_basis
                history_period = request.period

            if history_basis == "recent":
                payload = await lastfm_public_call(
                    client, api_key, "user.getRecentTracks",
                    {"user": username, "limit": 100, "extended": 1},
                )
                raw_history = lastfm_as_list((payload.get("recenttracks") or {}).get("track"))
            else:
                payload = await lastfm_public_call(
                    client, api_key, "user.getTopTracks",
                    {"user": username, "period": history_period, "limit": 100},
                )
                raw_history = lastfm_as_list((payload.get("toptracks") or {}).get("track"))

            track_frequency = {}
            artist_frequency = {}
            heard_keys = set()
            for raw in raw_history:
                track = lastfm_track_item(raw, 0)
                if not track:
                    continue
                track_key = (normalize(track["artist"]), normalize(track["title"]))
                heard_keys.add(track_key)
                track_frequency[track_key] = track_frequency.get(track_key, 0) + 1
                artist_key = normalize(track["artist"])
                artist_frequency[artist_key] = {
                    "name": track["artist"],
                    "count": artist_frequency.get(artist_key, {}).get("count", 0) + 1,
                }

            seeds = sorted(
                track_frequency.items(),
                key=lambda pair: pair[1],
                reverse=True,
            )[:6]
            artist_seeds = sorted(
                artist_frequency.values(),
                key=lambda item: item["count"],
                reverse=True,
            )[:3]

            for (artist_key, track_key), frequency in seeds:
                seed_artist = next(
                    (
                        lastfm_artist_name(raw.get("artist"))
                        for raw in raw_history
                        if normalize(lastfm_artist_name(raw.get("artist"))) == artist_key
                        and normalize(raw.get("name") or raw.get("title") or "") == track_key
                    ),
                    "",
                )
                seed_track = next(
                    (
                        str(raw.get("name") or raw.get("title") or "")
                        for raw in raw_history
                        if normalize(lastfm_artist_name(raw.get("artist"))) == artist_key
                        and normalize(raw.get("name") or raw.get("title") or "") == track_key
                    ),
                    "",
                )
                if not seed_artist or not seed_track:
                    continue
                similar = await lastfm_public_call(
                    client, api_key, "track.getSimilar",
                    {"artist": seed_artist, "track": seed_track, "autocorrect": 1, "limit": 18},
                )
                results = lastfm_as_list((similar.get("similartracks") or {}).get("track"))
                seed_weight = 1.0 + min(0.5, frequency / 5.0)
                for raw in results:
                    candidate = lastfm_track_item(raw, 0)
                    if not candidate:
                        continue
                    key = (normalize(candidate["artist"]), normalize(candidate["title"]))
                    if key in heard_keys:
                        continue
                    add_recommendation(
                        bucket, raw,
                        source=f"Похоже на: {seed_artist} — {seed_track}",
                        weight=seed_weight,
                    )
                await asyncio.sleep(0.2)

            similar_artists = {}
            for seed in artist_seeds:
                payload = await lastfm_public_call(
                    client, api_key, "artist.getSimilar",
                    {"artist": seed["name"], "autocorrect": 1, "limit": 5},
                )
                artists = lastfm_as_list((payload.get("similarartists") or {}).get("artist"))
                for raw_artist in artists:
                    name = lastfm_artist_name(raw_artist)
                    if not name or normalize(name) == normalize(seed["name"]):
                        continue
                    key = normalize(name)
                    match = recommendation_match(raw_artist.get("match"))
                    existing = similar_artists.get(key)
                    source = f"Похоже на исполнителя: {seed['name']}"
                    if existing is None or match > existing["match"]:
                        similar_artists[key] = {
                            "name": name,
                            "match": match,
                            "source": source,
                        }
                await asyncio.sleep(0.2)

            for similar_artist in sorted(
                similar_artists.values(),
                key=lambda item: item["match"],
                reverse=True,
            )[:6]:
                await add_artist_top_tracks(
                    client, api_key, similar_artist["name"], bucket,
                    source=similar_artist["source"],
                    weight=similar_artist["match"],
                    limit=4,
                )

            for key, entry in list(bucket.items()):
                if (normalize(entry["artist"]), normalize(entry["title"])) in heard_keys:
                    bucket.pop(key, None)

        elif request.mode == "artist_discovery":
            if not request.artist:
                raise HTTPException(status_code=422, detail="Укажите исполнителя.")
            label = f"Рекомендации по исполнителю {request.artist}"
            payload = await lastfm_public_call(
                client, api_key, "artist.getSimilar",
                {"artist": request.artist, "autocorrect": 1, "limit": 12},
            )
            similar_artists = lastfm_as_list((payload.get("similarartists") or {}).get("artist"))
            for artist in similar_artists[:min(12, max(5, request.count // 3))]:
                name = lastfm_artist_name(artist)
                if not name:
                    continue
                await add_artist_top_tracks(
                    client, api_key, name, bucket,
                    source=f"Похож на {request.artist}",
                    weight=recommendation_match(artist.get("match")),
                    limit=5,
                )

        else:  # tag_discovery
            if not request.tag:
                raise HTTPException(status_code=422, detail="Укажите жанр или тег.")
            label = f"Рекомендации по тегу {request.tag}"
            payload = await lastfm_public_call(
                client, api_key, "tag.getTopArtists",
                {"tag": request.tag, "limit": 12},
            )
            seed_artists = lastfm_as_list((payload.get("topartists") or {}).get("artist"))
            related_artists = {}
            for seed_artist in seed_artists[:5]:
                seed_name = lastfm_artist_name(seed_artist)
                if not seed_name:
                    continue
                similar = await lastfm_public_call(
                    client, api_key, "artist.getSimilar",
                    {"artist": seed_name, "autocorrect": 1, "limit": 4},
                )
                for raw_artist in lastfm_as_list((similar.get("similarartists") or {}).get("artist")):
                    name = lastfm_artist_name(raw_artist)
                    if not name:
                        continue
                    key = normalize(name)
                    score = recommendation_match(raw_artist.get("match"))
                    previous = related_artists.get(key)
                    if previous is None or score > previous["score"]:
                        related_artists[key] = {
                            "name": name,
                            "score": score,
                            "source": f"По тегу {request.tag} через {seed_name}",
                        }
                await asyncio.sleep(0.2)
            for artist in sorted(
                related_artists.values(),
                key=lambda item: item["score"],
                reverse=True,
            )[:8]:
                await add_artist_top_tracks(
                    client, api_key, artist["name"], bucket,
                    source=artist["source"], weight=artist["score"], limit=4,
                )

    results = finalize_recommendations(bucket, request.count)
    return {
        "mode": request.mode,
        "label": label,
        "kind": "track",
        "count": len(results),
        "items": results,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/songs")
async def queue_songs(request: SongList):
    global CURRENT_BATCH_ID
    if not API_KEY:
        raise HTTPException(status_code=500, detail="SLSKD_API_KEY is not configured")

    # Do not overlap the old search lifecycle with its replacement. Existing
    # transfers are deliberately left running and are not cancelled.
    async with BATCH_START_LOCK:
        previous_batch = BATCHES.get(CURRENT_BATCH_ID) if CURRENT_BATCH_ID else None
        if previous_batch and not previous_batch["search_finished"]:
            previous_batch["cancel_requested"] = True
            previous_batch["phase"] = "cancelling"
            previous_batch["cancel_event"].set()
            try:
                await asyncio.wait_for(
                    previous_batch["search_done_event"].wait(),
                    timeout=120,
                )
            except asyncio.TimeoutError as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Не удалось дождаться остановки предыдущего поиска. Новый поиск не запущен, чтобы не превысить лимит параллельности slskd.",
                ) from exc

        batch_id = str(uuid.uuid4())
        BATCHES[batch_id] = {
            "finished": False,
            "search_finished": False,
            "cancel_requested": False,
            "cancel_event": asyncio.Event(),
            "search_done_event": asyncio.Event(),
            "active_searches": set(),
            "search_semaphore": asyncio.Semaphore(request.max_parallel_searches),
            "serial_search_semaphore": asyncio.Semaphore(1),
            "phase": "queued",
            "retry_count": 0,
            "serial_retry_count": 0,
            "settings": {
                "search_duration_seconds": request.search_duration_seconds,
                "retry_not_found": request.retry_not_found,
                "deep_search_duration_seconds": request.deep_search_duration_seconds,
                "max_parallel_searches": request.max_parallel_searches,
                "search_start_interval_seconds": SEARCH_START_INTERVAL_SECONDS,
            },
            "results": [
                {
                    "song": song.model_dump(),
                    "song_obj": song,
                    "status": "pending",
                    "message": "Ожидает обработки",
                    "search_attempt": 0,
                    "percent": 0,
                    "bytes_transferred": 0,
                    "size": 0,
                }
                for song in request.songs
            ],
        }
        CURRENT_BATCH_ID = batch_id
        asyncio.create_task(run_batch(batch_id))
        return {"batch_id": batch_id, "submitted": len(request.songs)}


@app.post("/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    batch = BATCHES.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if batch["search_finished"] or batch["finished"]:
        return {
            "cancel_requested": False,
            "message": "Поисковая часть уже завершена; поставленные загрузки продолжаются",
        }

    batch["cancel_requested"] = True
    batch["phase"] = "cancelling"
    batch["cancel_event"].set()
    return {
        "cancel_requested": True,
        "active_searches": len(batch["active_searches"]),
        "message": "Прерываем активные и ожидающие поиски",
    }


@app.get("/batches/{batch_id}")
async def get_batch(batch_id: str):
    batch = BATCHES.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    results = [
        {key: value for key, value in item.items() if key != "song_obj"}
        for item in batch["results"]
    ]
    counts = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1

    return {
        "batch_id": batch_id,
        "submitted": len(results),
        "finished": batch["finished"],
        "search_finished": batch["search_finished"],
        "cancel_requested": batch["cancel_requested"],
        "phase": batch["phase"],
        "retry_count": batch["retry_count"],
        "serial_retry_count": batch["serial_retry_count"],
        "active_searches": len(batch["active_searches"]),
        "settings": batch["settings"],
        "counts": counts,
        "queued": sum(
            item["status"] in {"queued", "connecting", "downloading", "paused", "downloaded"}
            for item in results
        ),
        "downloaded": counts.get("downloaded", 0),
        "not_found": counts.get("not_found", 0),
        "results": results,
    }
