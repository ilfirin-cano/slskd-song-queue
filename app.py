import asyncio
import os
import re
import uuid
from difflib import SequenceMatcher
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

SLSKD_URL = os.getenv("SLSKD_URL", "http://slskd:5030").rstrip("/")
API_KEY = os.getenv("SLSKD_API_KEY", "")

# SEARCH_TIMEOUT_MS and MAX_WAIT_SECONDS remain as backward-compatible defaults.
SEARCH_TIMEOUT_MS = int(os.getenv("SEARCH_TIMEOUT_MS", "25000"))
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "35"))
SEARCH_DURATION_SECONDS = int(os.getenv(
    "SEARCH_DURATION_SECONDS",
    str(max(5, round(SEARCH_TIMEOUT_MS / 1000))),
))
SEARCH_WAIT_GRACE_SECONDS = int(os.getenv("SEARCH_WAIT_GRACE_SECONDS", "5"))
DEEP_SEARCH_DURATION_SECONDS = int(os.getenv("DEEP_SEARCH_DURATION_SECONDS", "60"))
DEEP_SEARCH_MIN_MATCH_SCORE = float(os.getenv("DEEP_SEARCH_MIN_MATCH_SCORE", "0.50"))
RETRY_NOT_FOUND_BY_DEFAULT = os.getenv("RETRY_NOT_FOUND_BY_DEFAULT", "true").lower() in {"1", "true", "yes", "on"}

POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1"))
MIN_MATCH_SCORE = float(os.getenv("MIN_MATCH_SCORE", "0.52"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS_PER_SONG", "100"))
DEEP_SEARCH_RESPONSE_LIMIT = int(os.getenv("DEEP_SEARCH_RESPONSE_LIMIT", str(max(MAX_RESULTS, 250))))
TRANSFER_POLL_INTERVAL = float(os.getenv("TRANSFER_POLL_INTERVAL", "2"))
TRANSFER_MONITOR_HOURS = float(os.getenv("TRANSFER_MONITOR_HOURS", "24"))
MAX_429_RETRIES = int(os.getenv("SLSKD_429_RETRIES", "5"))

# slskd 0.24+ rejects overlapping search/download API operations with HTTP 429.
# Mutating API operations are serialized; actual file transfers continue independently.
SLSKD_OPERATION_LOCK = asyncio.Lock()
ALLOWED_EXTENSIONS = {".mp3", ".flac"}

app = FastAPI(title="slskd Song Queue", version="2.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
BATCHES = {}


class Song(BaseModel):
    artist: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)


class SongList(BaseModel):
    songs: list[Song] = Field(min_length=1, max_length=100)
    search_duration_seconds: int = Field(default=SEARCH_DURATION_SECONDS, ge=5, le=180)
    retry_not_found: bool = Field(default=RETRY_NOT_FOUND_BY_DEFAULT)
    deep_search_duration_seconds: int = Field(default=DEEP_SEARCH_DURATION_SECONDS, ge=10, le=300)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")


@app.get("/settings")
async def get_settings():
    return {
        "search_duration_seconds": SEARCH_DURATION_SECONDS,
        "retry_not_found": RETRY_NOT_FOUND_BY_DEFAULT,
        "deep_search_duration_seconds": DEEP_SEARCH_DURATION_SECONDS,
    }


def normalize(value: str) -> str:
    value = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", value.lower())
    value = re.sub(r"[^a-z0-9а-яё]+", " ", value)
    return " ".join(value.split())


def file_extension(filename: str) -> str:
    normalized_path = filename.replace("\\", "/")
    return os.path.splitext(normalized_path)[1].lower()


def format_priority(filename: str) -> int:
    # MP3 is the requested preferred format; FLAC is the fallback.
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


async def post_with_429_retry(client, path: str, *, json_body):
    """Retry slskd operations rejected while another operation is in flight."""
    response = None
    for attempt in range(MAX_429_RETRIES + 1):
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


async def search_slskd(client, query: str, duration_seconds: int, response_limit: int):
    """Run one search lifecycle. The caller must hold SLSKD_OPERATION_LOCK."""
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
    )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration_seconds + SEARCH_WAIT_GRACE_SECONDS
    completed = False
    while loop.time() < deadline:
        state_response = await client.get(f"/api/v0/searches/{search_id}")
        state_response.raise_for_status()
        if state_response.json().get("isComplete") is True:
            completed = True
            break
        await asyncio.sleep(POLL_INTERVAL)

    results_response = await client.get(f"/api/v0/searches/{search_id}/responses")
    results_response.raise_for_status()
    payload = results_response.json()

    # If our local deadline expired before slskd completed the search,
    # cancel the lingering search before starting the next one.
    if not completed:
        try:
            cancel_response = await client.delete(f"/api/v0/searches/{search_id}")
            if cancel_response.status_code not in {200, 202, 204, 404}:
                cancel_response.raise_for_status()
        except httpx.HTTPError:
            pass

    return payload


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

    # Prefer MP3 over FLAC; within a format, use filename match and bitrate.
    candidates.sort(
        key=lambda item: (item["format_priority"], item["score"], item["bitrate"]),
        reverse=True,
    )
    return candidates


def deep_search_queries(song: Song):
    title = song.title.strip()
    # Remove common parenthetical tags and featured-artist suffixes for a broader query.
    clean_title = re.sub(r"\s+(?:feat\.?|ft\.?|featuring)\s+.+$", "", title, flags=re.IGNORECASE)
    clean_title = re.sub(r"\s*[\[(].*?[\])]\s*", " ", clean_title)
    clean_title = " ".join(clean_title.split())

    queries = [
        f"{song.artist} {clean_title}".strip(),
        clean_title,
    ]
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
    if "succeeded" in state or (
        "completed" in state and not any(word in state for word in ("rejected", "errored", "failed", "timedout"))
    ) or file.get("percentComplete", 0) >= 100:
        return "downloaded"
    if any(word in state for word in ("errored", "failed", "error", "rejected", "timedout")):
        return "failed"
    if "cancelled" in state or "canceled" in state:
        return "cancelled"
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


async def process_song(
    batch_id: str,
    index: int,
    song: Song,
    search_duration_seconds: int,
    deep_search_duration_seconds: int,
    deep_search: bool = False,
):
    attempt = 2 if deep_search else 1
    threshold = DEEP_SEARCH_MIN_MATCH_SCORE if deep_search else MIN_MATCH_SCORE
    if deep_search:
        queries = deep_search_queries(song)
        duration = deep_search_duration_seconds
        response_limit = DEEP_SEARCH_RESPONSE_LIMIT
        update_item(
            batch_id, index,
            status="searching",
            search_attempt=attempt,
            message=f"Повторный расширенный поиск ({duration} сек.)",
        )
    else:
        queries = [f"{song.artist} {song.title}"]
        duration = search_duration_seconds
        response_limit = MAX_RESULTS
        update_item(
            batch_id, index,
            status="searching",
            search_attempt=attempt,
            message=f"Первичный поиск ({duration} сек.)",
        )

    try:
        async with httpx.AsyncClient(
            base_url=SLSKD_URL,
            headers={"X-API-Key": API_KEY},
            timeout=30,
        ) as client:
            candidates_by_key = {}

            # Hold the operation lock through search AND enqueue. This ensures
            # the selected file is queued immediately before the next search
            # starts, so real transfers overlap with subsequent searches.
            async with SLSKD_OPERATION_LOCK:
                for query_index, query in enumerate(queries, start=1):
                    update_item(
                        batch_id, index,
                        status="searching",
                        search_attempt=attempt,
                        message=(
                            f"Расширенный поиск: запрос {query_index}/{len(queries)}"
                            if deep_search else f"Поиск: {duration} сек."
                        ),
                    )
                    payload = await search_slskd(client, query, duration, response_limit)
                    found = extract_audio_candidates(song, payload, threshold)
                    for candidate in found:
                        key = (candidate["username"], candidate["filename"])
                        previous = candidates_by_key.get(key)
                        if previous is None or candidate["score"] > previous["score"]:
                            candidates_by_key[key] = candidate

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
                )

                update_item(
                    batch_id, index,
                    status="queued",
                    message=f"{best['format']} добавлен в очередь slskd",
                )

        # The transfer keeps running while the next song is searched.
        return asyncio.create_task(
            monitor_transfer(batch_id, index, best["username"], best["filename"])
        )
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

        # The first-pass searches are complete. Re-search misses now, without
        # waiting for active downloads: their transfer tasks run in the background.
        if settings["retry_not_found"]:
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

        batch["phase"] = "waiting_for_transfers"
        if monitor_tasks:
            await asyncio.gather(*monitor_tasks, return_exceptions=True)
    finally:
        batch["finished"] = True
        batch["phase"] = "finished"
        for item in batch["results"]:
            item.pop("song_obj", None)


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
        "phase": "queued",
        "retry_count": 0,
        "settings": {
            "search_duration_seconds": request.search_duration_seconds,
            "retry_not_found": request.retry_not_found,
            "deep_search_duration_seconds": request.deep_search_duration_seconds,
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
        "phase": batch["phase"],
        "retry_count": batch["retry_count"],
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
