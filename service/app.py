"""Violation image generation service: a web UI over synth.try_astra_edit_batch.edit_frame.

Run: .venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8000   (then open http://127.0.0.1:8000)
Inputs are sorted by input kind (synth.violation_cases.INPUT_KINDS: what the frame shows, e.g. a forklift with an
LSP and cargo), given in one of three ways: a folder path on this machine (free text) whose sub-folders are named by
kind, a zip of such a folder, or images uploaded per kind. A job takes one engine, some of its cases and a number of
variants: each input kind feeds one engine, so only frames of that engine's kinds are used, and every (frame, case
that takes the frame's kind, variant) is one task that makes exactly one output with one Astra edit. Tasks of all
jobs share one pool of WORKERS threads (3 at a time); each task's state goes to the job's WebSocket subscribers as
soon as it changes, its output included, and a job's outputs download one by one or as one zip (one folder per
input kind).
Jobs live in memory (a restart forgets them); their files stay in work/service/jobs/<job id>/: inputs/<kind>/
(uploads, the unpacked zip), runs/<kind>/ (per task: references, prompt, Astra log, raw image), outputs/<kind>/ and
thumbs/.
"""

from __future__ import annotations

import asyncio
import io
import re
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from starlette.datastructures import UploadFile

from synth.try_astra_edit_batch import IMAGE_TYPES, ROOT, edit_frame, frames
from synth.violation_cases import ENGINES, INPUT_KINDS, catalogue

JOBS_DIR = ROOT / "work/service/jobs"
STATIC = Path(__file__).resolve().parent / "static"
WORKERS = 3
MAX_VARIANTS = 10
MAX_TASKS = 500

app = FastAPI(title="Violation generator")
executor = ThreadPoolExecutor(WORKERS, thread_name_prefix="astra")
jobs: dict[str, "Job"] = {}


class Job:
    def __init__(self, engine: str, cases: list[str], variants: int, inputs: dict[str, list[Path]], notes: list[str],
                 folder: Path, loop: asyncio.AbstractEventLoop):
        self.id = folder.name
        self.folder = folder
        self.engine, self.cases, self.variants = engine, cases, variants
        self.inputs, self.notes = inputs, notes
        self.loop = loop
        self.lock = threading.Lock()
        self.subscribers: set[WebSocket] = set()
        self.tasks = []
        frame_index = 0
        for kind, images in inputs.items():
            for image in images:
                for case_id in cases:
                    case = ENGINES[engine]["cases"][case_id]
                    if kind not in case["changes"]:
                        continue
                    for k in range(variants):
                        name = f"{image.stem}__{case_id}__v{k + 1}.png"
                        self.tasks.append({
                            "id": f"t{len(self.tasks) + 1:04d}", "input": image.name, "input_kind": kind,
                            "case": case_id, "case_title": case["title"], "variant": k + 1,
                            "seed": frame_index * variants + k, "status": "queued", "output_name": name,
                            "path": f"{kind}/{name}", "_image": str(image)})
                frame_index += 1

    def task_view(self, task: dict) -> dict:
        return {key: value for key, value in task.items() if not key.startswith("_")}

    def public(self) -> dict:
        with self.lock:
            tasks = [self.task_view(task) for task in self.tasks]
        counts = {status: sum(t["status"] == status for t in tasks) for status in ("queued", "running", "done", "failed")}
        return {"id": self.id, "engine": self.engine, "engine_name": ENGINES[self.engine]["name"], "cases": self.cases,
                "variants": self.variants, "inputs": {kind: len(images) for kind, images in self.inputs.items()},
                "notes": self.notes, "total": len(tasks), "counts": counts,
                "finished": counts["queued"] == counts["running"] == 0, "tasks": tasks}

    def notify(self, message: dict) -> None:
        """Send message to every subscriber, from any thread."""
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def _broadcast(self, message: dict) -> None:
        for socket in list(self.subscribers):
            try:
                await socket.send_json(message)
            except Exception:             # a closed socket; its handler removes it
                self.subscribers.discard(socket)


def run_task(job: Job, task: dict) -> None:
    """One task in a pool thread: one edit_frame call, its state sent before and after."""
    with job.lock:
        task["status"] = "running"
        task["started_at"] = time.time()
        view = job.task_view(task)
    job.notify({"type": "task", "task": view})
    case = ENGINES[job.engine]["cases"][task["case"]]
    kind = task["input_kind"]
    try:
        row = edit_frame(Path(task["_image"]), job.folder / "runs" / kind / task["output_name"].removesuffix(".png"),
                         job.folder / "outputs" / task["path"], task["seed"], case["changes"][kind],
                         INPUT_KINDS[kind]["scene"], refs=case["refs"])
    except Exception as error:            # edit_frame reports its own errors; this guards the pool
        row = {"error": f"{type(error).__name__}: {error}"}
    with job.lock:
        if row.get("output"):
            task["status"] = "done"
            task["url"] = f"/api/jobs/{job.id}/outputs/{task['path']}"
            task["thumb"] = f"/api/jobs/{job.id}/thumbs/{task['path']}"
        else:
            task["status"] = "failed"
            task["error"] = row.get("error", "no output")
        task["seconds"] = row.get("seconds", round(time.time() - task["started_at"]))
        view = job.task_view(task)
        finished = all(t["status"] in ("done", "failed") for t in job.tasks)
    job.notify({"type": "task", "task": view})
    if finished:
        job.notify({"type": "job", "job": job.public()})


def safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "image"
    return stem + Path(name).suffix.lower()


def unique(path: Path) -> Path:
    n = 1
    target = path
    while target.exists():
        n += 1
        target = path.with_name(f"{path.stem}_{n}{path.suffix}")
    return target


def kind_folders(root: Path) -> tuple[dict[str, list[Path]], list[str]]:
    """The images of root's sub-folders named by input kind (root may wrap them in one more folder, as the zip of
    a folder does), and notes on what was left out."""
    def named(base: Path) -> dict[str, Path]:
        return {p.name: p for p in sorted(base.iterdir()) if p.is_dir() and p.name in INPUT_KINDS}
    found = named(root)
    if not found:
        wrappers = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "__MACOSX"))]
        if len(wrappers) == 1:
            root, found = wrappers[0], named(wrappers[0])
    if not found:
        raise HTTPException(400, f"no input-kind folders in {root.name or root}: the input needs sub-folders named "
                                 + ", ".join(INPUT_KINDS))
    notes = []
    other = sorted(p.name for p in root.iterdir() if p.name not in INPUT_KINDS
                   and not p.name.startswith((".", "__MACOSX")))
    if other:
        notes.append(f"left out (not an input-kind folder): {', '.join(other)}")
    return {kind: frames([folder]) for kind, folder in found.items()}, notes


def unpack_zip(upload: bytes, target: Path) -> None:
    """The images of a zip, keeping their folders; nothing outside target, nothing but images."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(upload))
    except zipfile.BadZipFile:
        raise HTTPException(400, "the uploaded file is not a zip")
    with archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename.replace("\\", "/")).parts
            if (member.is_dir() or not parts or any(p in ("", ".", "..") for p in parts) or parts[0].startswith("/")
                    or Path(parts[-1]).suffix.lower() not in IMAGE_TYPES):
                continue
            destination = target.joinpath(*[safe_name(p) if i == len(parts) - 1 else re.sub(r"[^A-Za-z0-9._-]+", "_", p)
                                            for i, p in enumerate(parts)])
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source:
                destination.write_bytes(source.read())


def get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return job


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/engines")
def engines() -> dict:
    return catalogue()


@app.post("/api/jobs")
async def create_job(request: Request) -> dict:
    """Form fields: engine, cases (repeated), variants, and the input as ONE of: folder (a path on the server),
    zip (a zip file), files:<input kind> (images, repeated, for each kind)."""
    form = await request.form()
    engine = str(form.get("engine", ""))
    if engine not in ENGINES:
        raise HTTPException(400, f"unknown engine {engine!r}")
    cases = list(dict.fromkeys(str(c) for c in form.getlist("cases")))
    unknown = [c for c in cases if c not in ENGINES[engine]["cases"]]
    if unknown or not cases:
        raise HTTPException(400, f"choose cases of {engine}" + (f"; unknown: {', '.join(unknown)}" if unknown else ""))
    try:
        variants = int(form.get("variants", 1))
    except ValueError:
        variants = 0
    if not 1 <= variants <= MAX_VARIANTS:
        raise HTTPException(400, f"variants must be 1-{MAX_VARIANTS}")
    folder = str(form.get("folder", "")).strip()
    archive = form.get("zip")
    archive = archive if isinstance(archive, UploadFile) and archive.filename else None
    uploads = {kind: [f for f in form.getlist(f"files:{kind}") if isinstance(f, UploadFile) and f.filename]
               for kind in INPUT_KINDS}
    uploads = {kind: files for kind, files in uploads.items() if files}
    if sum((bool(folder), archive is not None, bool(uploads))) != 1:
        raise HTTPException(400, "give the input one way: a folder path, a zip, or images uploaded per input kind")

    job_folder = JOBS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    notes: list[str] = []
    if folder:
        source = Path(folder).expanduser()
        if not source.is_absolute():
            source = ROOT / source
        if not source.is_dir():
            raise HTTPException(400, f"no such folder on the server: {source}")
        inputs, notes = kind_folders(source)
    elif archive is not None:
        unpacked = job_folder / "inputs"
        unpacked.mkdir(parents=True)
        unpack_zip(await archive.read(), unpacked)
        inputs, notes = kind_folders(unpacked)
    else:
        inputs = {}
        for kind, files in uploads.items():
            (job_folder / "inputs" / kind).mkdir(parents=True, exist_ok=True)
            for upload in files:
                name = safe_name(upload.filename)
                if Path(name).suffix not in IMAGE_TYPES:
                    continue
                target = unique(job_folder / "inputs" / kind / name)
                target.write_bytes(await upload.read())
                inputs.setdefault(kind, []).append(target)

    # each input kind feeds one engine: the job uses the chosen engine's kinds only
    other = {kind: images for kind, images in inputs.items() if INPUT_KINDS[kind]["engine"] != engine and images}
    for kind, images in other.items():
        notes.append(f"{len(images)} {kind} image(s) left out: that kind feeds "
                     f"{ENGINES[INPUT_KINDS[kind]['engine']]['name']}")
    inputs = {kind: images for kind, images in inputs.items() if INPUT_KINDS[kind]["engine"] == engine and images}
    if not inputs:
        kinds = [kind for kind, k in INPUT_KINDS.items() if k["engine"] == engine]
        raise HTTPException(400, f"no images for {ENGINES[engine]['name']}: it takes {', '.join(kinds)} inputs"
                                 + (f" ({'; '.join(notes)})" if notes else ""))
    job_folder.mkdir(parents=True, exist_ok=True)
    job = Job(engine, cases, variants, inputs, notes, job_folder, asyncio.get_running_loop())
    if not job.tasks:
        raise HTTPException(400, "none of the chosen cases takes these input kinds")
    if len(job.tasks) > MAX_TASKS:
        raise HTTPException(400, f"{len(job.tasks)} outputs is over {MAX_TASKS}; narrow the input")
    jobs[job.id] = job
    for task in job.tasks:
        executor.submit(run_task, job, task)
    return job.public()


@app.get("/api/jobs/{job_id}")
def job_state(job_id: str) -> dict:
    return get_job(job_id).public()


def done_output(job_id: str, kind: str, name: str) -> tuple[Job, Path]:
    job = get_job(job_id)
    with job.lock:
        known = {t["path"] for t in job.tasks if t["status"] == "done"}
    if f"{kind}/{name}" not in known:
        raise HTTPException(404, "no such output")
    return job, job.folder / "outputs" / kind / name


@app.get("/api/jobs/{job_id}/outputs/{kind}/{name}")
def output(job_id: str, kind: str, name: str) -> FileResponse:
    _, path = done_output(job_id, kind, name)
    return FileResponse(path, media_type="image/png", filename=name)


@app.get("/api/jobs/{job_id}/thumbs/{kind}/{name}")
def thumbnail(job_id: str, kind: str, name: str) -> FileResponse:
    """A small JPEG of an output for the UI grid (the PNGs are full frames of a few MB)."""
    job, path = done_output(job_id, kind, name)
    thumb = job.folder / "thumbs" / kind / f"{Path(name).stem}.jpg"
    if not thumb.is_file():
        thumb.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            small = image.convert("RGB")
            small.thumbnail((640, 360))
            small.save(thumb, quality=82)
    return FileResponse(thumb, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/download.zip")
def download_all(job_id: str) -> StreamingResponse:
    job = get_job(job_id)
    with job.lock:
        paths = [t["path"] for t in job.tasks if t["status"] == "done"]
    if not paths:
        raise HTTPException(404, "no outputs yet")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:     # PNGs are compressed already
        for path in paths:
            archive.write(job.folder / "outputs" / path, path)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{job.id}_outputs.zip"'})


@app.websocket("/ws/jobs/{job_id}")
async def job_socket(socket: WebSocket, job_id: str) -> None:
    job = jobs.get(job_id)
    await socket.accept()
    if job is None:
        await socket.send_json({"type": "error", "error": f"no job {job_id}"})
        await socket.close()
        return
    job.subscribers.add(socket)
    await socket.send_json({"type": "job", "job": job.public()})   # the state so far, then every change
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        job.subscribers.discard(socket)
