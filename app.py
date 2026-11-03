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
SEARCH_TIMEOUT_MS = int(os.getenv("SEARCH_TIMEOUT_MS", "25000"))
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "35"))
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1"))
MIN_MATCH_SCORE = float(os.getenv("MIN_MATCH_SCORE", "0.52"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS_PER_SONG", "100"))
TRANSFER_POLL_INTERVAL = float(os.getenv("TRANSFER_POLL_INTERVAL", "2"))
TRANSFER_MONITOR_HOURS = float(os.getenv("TRANSFER_MONITOR_HOURS", "24"))

app = FastAPI(title="slskd Song Queue", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
BATCHES = {}

class Song(BaseModel):
    artist: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)

class SongList(BaseModel):
    songs: list[Song] = Field(min_length=1, max_length=100)

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")

def normalize(value: str) -> str:
    value = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", value.lower())
    value = re.sub(r"[^a-z0-9а-яё]+", " ", value)
    return " ".join(value.split())

def match_score(song: Song, filename: str, bitrate: int = 0) -> float:
    target = normalize(f"{song.artist} {song.title}")
    candidate = normalize(filename.rsplit("/", 1)[-1])
    score = SequenceMatcher(None, target, candidate).ratio()
    artist, title = normalize(song.artist), normalize(song.title)
    if artist and artist in candidate:
        score += 0.22
    if title and title in candidate:
        score += 0.28
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext in {"flac", "wav", "alac"}:
        score += 0.06
    elif ext in {"mp3", "m4a", "aac", "ogg", "opus"}:
        score += 0.03
    if bitrate >= 320:
        score += 0.03
    return min(score, 1.0)

def update_item(batch_id: str, index: int, **values):
    BATCHES[batch_id]["results"][index].update(values)

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
    state = str(file.get("state", "")).lower().replace("_", "").replace(" ", "")
    if state in {"succeeded", "completed", "complete"} or file.get("percentComplete", 0) >= 100:
        return "downloaded"
    if state in {"errored", "failed", "error"}:
        return "failed"
    if state in {"cancelled", "canceled"}:
        return "cancelled"
    if state == "paused":
        return "paused"
    if state in {"inprogress", "transferring", "downloading"}:
        return "downloading"
    if state in {"initializing", "connecting"}:
        return "connecting"
    return "queued"

async def monitor_transfer(batch_id, index, username, filename):
    deadline = asyncio.get_running_loop().time() + TRANSFER_MONITOR_HOURS * 3600
    async with httpx.AsyncClient(base_url=SLSKD_URL, headers={"X-API-Key": API_KEY}, timeout=20) as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                files = await get_download_files(client)
                found = next((f for f in files if f.get("username") == username and f.get("filename") == filename), None)
                if found:
                    status = transfer_status(found)
                    update_item(batch_id, index, status=status,
                        percent=round(float(found.get("percentComplete", 0) or 0), 1),
                        bytes_transferred=found.get("bytesTransferred", 0),
                        size=found.get("size", 0),
                        speed=found.get("averageSpeed", 0),
                        remaining=found.get("bytesRemaining", 0))
                    if status in {"downloaded", "failed", "cancelled"}:
                        return
            except (httpx.HTTPError, ValueError, TypeError):
                pass
            await asyncio.sleep(TRANSFER_POLL_INTERVAL)
    if BATCHES[batch_id]["results"][index]["status"] in {"queued", "connecting", "downloading", "paused"}:
        update_item(batch_id, index, status="monitoring_timeout", message="Истекло время наблюдения; проверьте состояние в slskd")

async def process_song(batch_id, index, song):
    update_item(batch_id, index, status="searching", message="Ищем совпадения в Soulseek")
    try:
        async with httpx.AsyncClient(base_url=SLSKD_URL, headers={"X-API-Key": API_KEY}, timeout=20) as client:
            search_id = str(uuid.uuid4())
            response = await client.post("/api/v0/searches", json={
                "id": search_id, "searchText": f"{song.artist} {song.title}",
                "searchTimeout": SEARCH_TIMEOUT_MS, "responseLimit": MAX_RESULTS})
            response.raise_for_status()
            deadline = asyncio.get_running_loop().time() + MAX_WAIT_SECONDS
            while asyncio.get_running_loop().time() < deadline:
                state = await client.get(f"/api/v0/searches/{search_id}")
                state.raise_for_status()
                if state.json().get("isComplete") is True:
                    break
                await asyncio.sleep(POLL_INTERVAL)
            response = await client.get(f"/api/v0/searches/{search_id}/responses")
            response.raise_for_status()
            candidates = []
            for peer in response.json() if isinstance(response.json(), list) else []:
                username = peer.get("username")
                for file in peer.get("files", []) or []:
                    filename = file.get("filename") or file.get("fileName")
                    if not username or not filename:
                        continue
                    try:
                        bitrate = int(file.get("bitRate") or file.get("bitrate") or 0)
                    except (TypeError, ValueError):
                        bitrate = 0
                    candidates.append({"username": username, "filename": filename,
                        "size": file.get("size", 0), "score": match_score(song, filename, bitrate)})
            candidates.sort(key=lambda x: x["score"], reverse=True)
            if not candidates or candidates[0]["score"] < MIN_MATCH_SCORE:
                update_item(batch_id, index, status="not_found", message="Подходящее совпадение не найдено", matches=len(candidates))
                return
            best = candidates[0]
            update_item(batch_id, index, status="adding", filename=best["filename"], peer=best["username"],
                size=best["size"], match_score=round(best["score"], 3), message="Добавляем файл в очередь slskd")
            queued = await client.post(f"/api/v0/transfers/downloads/{quote(best['username'], safe='')}",
                json=[{"filename": best["filename"], "size": best["size"]}])
            queued.raise_for_status()
            update_item(batch_id, index, status="queued", message="Файл добавлен в очередь slskd")
        await monitor_transfer(batch_id, index, best["username"], best["filename"])
    except httpx.HTTPStatusError as exc:
        update_item(batch_id, index, status="slskd_error", message=f"slskd HTTP {exc.response.status_code}", details=exc.response.text[:400])
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        update_item(batch_id, index, status="error", message="Не удалось связаться со slskd", details=str(exc)[:400])
    except Exception as exc:
        update_item(batch_id, index, status="error", message="Внутренняя ошибка", details=str(exc)[:400])

async def run_batch(batch_id):
    batch = BATCHES[batch_id]
    await asyncio.gather(*(process_song(batch_id, i, item["song_obj"]) for i, item in enumerate(batch["results"])))
    batch["finished"] = True
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
    BATCHES[batch_id] = {"finished": False, "results": [
        {"song": song.model_dump(), "song_obj": song, "status": "pending", "message": "Ожидает обработки",
         "percent": 0, "bytes_transferred": 0, "size": 0}
        for song in request.songs]}
    asyncio.create_task(run_batch(batch_id))
    return {"batch_id": batch_id, "submitted": len(request.songs)}

@app.get("/batches/{batch_id}")
async def get_batch(batch_id: str):
    batch = BATCHES.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    results = [{k: v for k, v in item.items() if k != "song_obj"} for item in batch["results"]]
    counts = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"batch_id": batch_id, "submitted": len(results), "finished": batch["finished"],
        "counts": counts, "queued": sum(x["status"] in {"queued", "connecting", "downloading", "paused", "downloaded"} for x in results),
        "downloaded": counts.get("downloaded", 0), "not_found": counts.get("not_found", 0), "results": results}
