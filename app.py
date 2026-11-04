import asyncio
import os
import re
import uuid
from typing import Literal
from difflib import SequenceMatcher
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

SLSKD_URL = os.getenv("SLSKD_URL", "http://slskd:5030").rstrip("/")
API_KEY = os.getenv("SLSKD_API_KEY", "")
LASTFM_PROXY = os.getenv("LASTFM_PROXY", "").strip() or None

LEGACY_SEARCH_TIMEOUT_MS = int(os.getenv("SEARCH_TIMEOUT_MS", "25000"))
SEARCH_DURATION_SECONDS = int(os.getenv(
    "SEARCH_DURATION_SECONDS",
    str(max(5, round(LEGACY_SEARCH_TIMEOUT_MS / 1000))),
))
SEARCH_WAIT_GRACE_SECONDS = int(os.getenv("SEARCH_WAIT_GRACE_SECONDS", "5"))
DEEP_SEARCH_DURATION_SECONDS = int(os.getenv("DEEP_SEARCH_DURATION_SECONDS", "60"))
DEEP_SEARCH_MIN_MATCH_SCORE = float(os.getenv("DEEP_SEARCH_MIN_MATCH_SCORE", "0.50"))
RETRY_NOT_FOUND_BY_DEFAULT = os.getenv("RETRY_NOT_FOUND_BY_DEFAULT", "true").lower() in {"1", "true", "yes", "on"}
DEFAULT_MAX_PARALLEL_SEARCHES = max(1, min(2, int(os.getenv("MAX_PARALLEL_SEARCHES", "2"))))
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

app = FastAPI(title="slskd Song Queue", version="2.3.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
BATCHES = {}


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
    max_parallel_searches: int = Field(default=DEFAULT_MAX_PARALLEL_SEARCHES, ge=1, le=2)


class LastFMRequest(BaseModel):
    api_key: str = Field(min_length=8, max_length=128)
    mode: Literal[
        "similar_tracks",
        "artist_top_tracks",
        "similar_artists",
        "tag_top_tracks",
        "user_top_tracks",
        "loved_tracks",
        "weekly_track_chart",
        "global_chart",
        "country_chart",
    ]
    seed_tracks: list[Song] = Field(default_factory=list, max_length=10)
    artist: str | None = Field(default=None, max_length=200)
    tag: str | None = Field(default=None, max_length=100)
    username: str | None = Field(default=None, max_length=100)
    country: str | None = Field(default=None, max_length=80)
    period: Literal["overall", "7day", "1month", "3month", "6month", "12month"] = "overall"
    count: int = Field(default=30, ge=1, le=100)


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

    update_item(batch_id, index, status="searching", search_attempt=attempt, message=status_message)

    try:
        # Parallel mode is capped at two (the slskd client-side limit).
        # A serial confirmation pass is used for first-pass misses because
        # slskd can occasionally complete concurrent searches with zero results.
        semaphore = (
            batch["serial_search_semaphore"]
            if serial_retry else batch["search_semaphore"]
        )
        async with semaphore:
            if cancel_event.is_set():
                raise BatchCancelled()

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
        if monitor_tasks:
            await asyncio.gather(*monitor_tasks, return_exceptions=True)
    finally:
        batch["search_finished"] = True
        batch["finished"] = True
        batch["phase"] = "cancelled" if batch["cancel_event"].is_set() else "finished"
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


def lastfm_track_item(raw, rank):
    if not isinstance(raw, dict):
        return None
    artist = lastfm_artist_name(raw.get("artist"))
    title = str(raw.get("name") or raw.get("title") or "").strip()
    if not artist or not title:
        return None
    return {
        "kind": "track",
        "artist": artist,
        "title": title,
        "rank": rank,
        "match": raw.get("match"),
        "playcount": raw.get("playcount"),
        "listeners": raw.get("listeners"),
        "url": raw.get("url"),
    }


def lastfm_artist_item(raw, rank):
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
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
            detail=f"Last.fm вернул HTTP {exc.response.status_code}",
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


@app.post("/lastfm/recommendations")
async def lastfm_recommendations(request: LastFMRequest):
    mode = request.mode
    if mode == "similar_tracks" and not request.seed_tracks:
        raise HTTPException(
            status_code=422,
            detail="Для поиска похожих треков добавьте хотя бы одну строку «Исполнитель — Название».",
        )
    if mode in {"artist_top_tracks", "similar_artists"} and not request.artist:
        raise HTTPException(status_code=422, detail="Укажите исполнителя.")
    if mode == "tag_top_tracks" and not request.tag:
        raise HTTPException(status_code=422, detail="Укажите музыкальный тег.")
    if mode in {"user_top_tracks", "loved_tracks", "weekly_track_chart"} and not request.username:
        raise HTTPException(status_code=422, detail="Укажите имя пользователя Last.fm.")
    if mode == "country_chart" and not request.country:
        raise HTTPException(status_code=422, detail="Укажите страну на английском языке.")

    items = []
    label = ""
    async with httpx.AsyncClient(
        base_url="https://ws.audioscrobbler.com",
        proxy=LASTFM_PROXY,
        headers={"User-Agent": "slskd-song-queue/2.3.0"},
        timeout=25,
    ) as client:
        if mode == "similar_tracks":
            label = "Треки, похожие на выбранные композиции"
            for seed_index, seed in enumerate(request.seed_tracks):
                payload = await lastfm_call(
                    client,
                    request.api_key,
                    "track.getSimilar",
                    {
                        "artist": seed.artist,
                        "track": seed.title,
                        "autocorrect": 1,
                        "limit": min(request.count, 100),
                    },
                )
                container = payload.get("similartracks") or {}
                items.extend(
                    item for item in (
                        lastfm_track_item(raw, i + 1)
                        for i, raw in enumerate(lastfm_as_list(container.get("track")))
                    ) if item
                )
                # Keep multi-seed requests comfortably below Last.fm's request-rate limits.
                if seed_index + 1 < len(request.seed_tracks):
                    await asyncio.sleep(1.0)

        elif mode == "artist_top_tracks":
            label = f"Популярные треки: {request.artist}"
            payload = await lastfm_call(
                client, request.api_key, "artist.getTopTracks",
                {"artist": request.artist, "autocorrect": 1, "limit": request.count},
            )
            raw_items = lastfm_as_list((payload.get("toptracks") or {}).get("track"))
            items = [item for item in (lastfm_track_item(raw, i + 1) for i, raw in enumerate(raw_items)) if item]

        elif mode == "similar_artists":
            label = f"Исполнители, похожие на {request.artist}"
            payload = await lastfm_call(
                client, request.api_key, "artist.getSimilar",
                {"artist": request.artist, "autocorrect": 1, "limit": request.count},
            )
            raw_items = lastfm_as_list((payload.get("similarartists") or {}).get("artist"))
            items = [item for item in (lastfm_artist_item(raw, i + 1) for i, raw in enumerate(raw_items)) if item]

        elif mode == "tag_top_tracks":
            label = f"Популярное по тегу: {request.tag}"
            payload = await lastfm_call(
                client, request.api_key, "tag.getTopTracks",
                {"tag": request.tag, "limit": request.count},
            )
            raw_items = lastfm_as_list((payload.get("tracks") or {}).get("track"))
            items = [item for item in (lastfm_track_item(raw, i + 1) for i, raw in enumerate(raw_items)) if item]

        elif mode == "user_top_tracks":
            label = f"Топ треков пользователя {request.username}"
            payload = await lastfm_call(
                client, request.api_key, "user.getTopTracks",
                {"user": request.username, "period": request.period, "limit": request.count},
            )
            raw_items = lastfm_as_list((payload.get("toptracks") or {}).get("track"))
            items = [item for item in (lastfm_track_item(raw, i + 1) for i, raw in enumerate(raw_items)) if item]

        elif mode == "loved_tracks":
            label = f"Любимые треки пользователя {request.username}"
            payload = await lastfm_call(
                client, request.api_key, "user.getLovedTracks",
                {"user": request.username, "limit": request.count},
            )
            raw_items = lastfm_as_list((payload.get("lovedtracks") or {}).get("track"))
            items = [item for item in (lastfm_track_item(raw, i + 1) for i, raw in enumerate(raw_items)) if item]

        elif mode == "weekly_track_chart":
            label = f"Недельный чарт пользователя {request.username}"
            payload = await lastfm_call(
                client, request.api_key, "user.getWeeklyTrackChart",
                {"user": request.username},
            )
            raw_items = lastfm_as_list((payload.get("weeklytrackchart") or {}).get("track"))
            items = [item for item in (lastfm_track_item(raw, i + 1) for i, raw in enumerate(raw_items)) if item]

        elif mode == "global_chart":
            label = "Глобальный чарт Last.fm"
            payload = await lastfm_call(
                client, request.api_key, "chart.getTopTracks", {"limit": request.count},
            )
            raw_items = lastfm_as_list((payload.get("tracks") or {}).get("track"))
            items = [item for item in (lastfm_track_item(raw, i + 1) for i, raw in enumerate(raw_items)) if item]

        else:  # country_chart
            label = f"Чарт Last.fm: {request.country}"
            payload = await lastfm_call(
                client, request.api_key, "geo.getTopTracks",
                {"country": request.country, "limit": request.count},
            )
            raw_items = lastfm_as_list((payload.get("tracks") or {}).get("track"))
            items = [item for item in (lastfm_track_item(raw, i + 1) for i, raw in enumerate(raw_items)) if item]

    # A track can be returned for multiple seed songs. Keep its first appearance.
    unique = []
    seen = set()
    for item in items:
        key = (normalize(item.get("artist") or item.get("name") or ""),
               normalize(item.get("title") or ""))
        if key in seen:
            continue
        seen.add(key)
        item["rank"] = len(unique) + 1
        unique.append(item)

    return {
        "mode": mode,
        "label": label,
        "kind": "artist" if mode == "similar_artists" else "track",
        "count": len(unique),
        "items": unique[:request.count * (len(request.seed_tracks) if mode == "similar_tracks" else 1)],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/songs")
async def queue_songs(request: SongList):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="SLSKD_API_KEY is not configured")

    batch_id = str(uuid.uuid4())
    BATCHES[batch_id] = {
        "finished": False,
        "search_finished": False,
        "cancel_requested": False,
        "cancel_event": asyncio.Event(),
        "active_searches": set(),
        "search_semaphore": asyncio.Semaphore(min(request.max_parallel_searches, 2)),
        "serial_search_semaphore": asyncio.Semaphore(1),
        "phase": "queued",
        "retry_count": 0,
        "serial_retry_count": 0,
        "settings": {
            "search_duration_seconds": request.search_duration_seconds,
            "retry_not_found": request.retry_not_found,
            "deep_search_duration_seconds": request.deep_search_duration_seconds,
            "max_parallel_searches": min(request.max_parallel_searches, 2),
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
