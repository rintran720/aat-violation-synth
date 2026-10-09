"""Violation image generation service: a web UI over synth.try_astra_edit_batch.edit_frame.

Run: .venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8000   (then open http://127.0.0.1:8000)
Inputs are sorted by input kind (synth.violation_cases.INPUT_KINDS: what the frame shows, e.g. a forklift with an
LSP and cargo), given in one of three ways: a folder path on this machine (free text) whose sub-folders are named by
kind, a zip of such a folder, or images uploaded per kind. A job takes one engine, some of its cases and a number of
outputs per frame (`variants`): each input kind feeds one engine, so only frames of that engine's kinds are used, and
each frame gets `variants` tasks, each of one chosen case that takes the frame's kind (the cases take turns across
frames), each task making exactly one output with one Astra edit. Tasks of all
jobs share one pool of WORKERS threads (3 at a time); each task's state goes to the job's WebSocket subscribers as
soon as it changes, its output included, and a job's outputs download one by one or as one zip (one folder per
input kind).
Every job is kept on disk in work/service/jobs/<job id>/: job.json (its settings, inputs, every task's state and
the tokens Astra reported), inputs/<kind>/ (a copy of every input frame, whatever way it came), runs/<kind>/ (per
task: references, prompt, Astra log, raw image), outputs/<kind>/ and thumbs/. On start the service loads every
job.json, so a job's URL keeps working; a task that was queued or running when the service stopped is marked
interrupted and is not run again (a job only runs when someone starts it).
Cancel (POST /api/jobs/<id>/cancel) marks a job's waiting outputs interrupted too; running ones finish.
Tokens: each attempt records structured Codex input/output/cache usage; legacy counts are input-only.
An input-token estimate for a new job is its
number of outputs times the median of the counts recorded so far (with the 10th-90th percentile as its range).
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hmac
import io
import json
import os
import random
import re
import shutil
import statistics
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile

from service.codex_status import CodexUsage
from synth.token_usage import legacy_usage, migrate_task, read_usage, summarize, update_task
from synth.try_astra_edit_batch import IMAGE_TYPES, ROOT, edit_frame, frames
from synth.violation_cases import ENGINES, INPUT_KINDS, catalogue, prompt_context

JOBS_DIR = ROOT / "work/service/jobs"
STATIC = Path(__file__).resolve().parent / "static"
WORKERS = 3
MAX_VARIANTS = 10
MAX_TASKS = 500
# the Codex usage limit only warns (user, 2026-10-09: do not block generating when the quota is used up); with
# ENFORCE_CODEX_LIMIT=1 in the environment a job, resume or rebuild over it is refused (409) unless forced
ENFORCE_CODEX_LIMIT = os.environ.get("ENFORCE_CODEX_LIMIT") == "1"
FALLBACK_TOKENS = 22_000          # median Astra tokens per output over the first 83 runs (2026-10-06)

app = FastAPI(title="Violation generator")


class BasicAuth:
    """HTTP Basic auth over every request and WebSocket, when VG_USER and VG_PASSWORD are set (the service listens
    on a public address: without it anyone could spend the Codex plan and read the server's image folders)."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        user, password = os.environ.get("VG_USER"), os.environ.get("VG_PASSWORD")
        if scope["type"] not in ("http", "websocket") or not (user and password):
            return await self.inner(scope, receive, send)
        header = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
        expected = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
        if hmac.compare_digest(header, expected):
            return await self.inner(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"www-authenticate", b'Basic realm="Violation Generator"'),
                                (b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"Sign in to use the Violation Generator."})


app.add_middleware(BasicAuth)
app.mount("/label-preview", StaticFiles(directory=ROOT / "work/label_exports", html=True, check_dir=False),
          name="label-preview")
executor = ThreadPoolExecutor(WORKERS, thread_name_prefix="astra")
jobs: dict[str, "Job"] = {}


def service_totals() -> tuple[int, int]:
    """Outputs made and tokens used by all jobs so far: the counters the Codex usage calibration reads against."""
    outputs = tokens = 0
    for job in list(jobs.values()):
        with job.lock:
            for task in job.tasks + job.deleted:
                outputs += task["status"] == "done"
                tokens += task.get("tokens") or 0
    return outputs, tokens


def pending_outputs() -> int:
    """Outputs queued or running in any job: they will use part of what the Codex plan has left."""
    count = 0
    for job in list(jobs.values()):
        with job.lock:
            count += sum(t["status"] in ("queued", "running") for t in job.tasks)
    return count


USAGE = CodexUsage(ROOT / "work/service/usage.json", service_totals)


async def codex_fit(outputs: int) -> dict:
    """USAGE.check for `outputs` more; unless ENFORCE_CODEX_LIMIT, over the limit still fits, its reason a warning."""
    fit = await asyncio.to_thread(USAGE.check, outputs, pending_outputs())
    if not fit["fits"] and not ENFORCE_CODEX_LIMIT:
        fit = fit | {"fits": True, "over": True, "reason": f"{fit['reason']} Not enforced: it runs anyway."}
    return fit


class Job:
    def __init__(self, folder: Path, engine: str, cases: list[str], variants: int, inputs: dict[str, list[Path]],
                 notes: list[str], created: float | None = None, tasks: list[dict] | None = None,
                 excluded: list[str] | None = None, deleted: list[dict] | None = None,
                 rates: dict[str, float] | None = None):
        self.id = folder.name
        self.folder = folder
        self.engine, self.cases, self.variants = engine, cases, variants
        self.rates = {case_id: (rates or {}).get(case_id, 1.0) for case_id in cases}   # chance a case is drawn
        self.inputs, self.notes = inputs, notes
        self.created = created or time.time()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.lock = threading.Lock()
        self.subscribers: set[WebSocket] = set()
        self.tasks = tasks if tasks is not None else self._plan()
        for task in self.tasks:
            migrate_task(task)
        self.excluded: set[str] = set(excluded or [])     # "<kind>/<name>" inputs the reviewer removed (X)
        # outputs the user deleted: off the page and out of downloads, kept here for their tokens, ids and names
        self.deleted: list[dict] = deleted or []
        self.tickets: dict[str, int] = {}       # task id -> its latest submit; an older (cancelled) submit does nothing

    def _plan(self) -> list[dict]:
        """At most `variants` outputs per frame (user, 2026-10-09), each of a different chosen case that takes the
        frame's kind. Per frame every such case is drawn with its rate (0-1, default 1); when more are drawn than
        `variants`, the ones the kind has used least so far are kept (ties at random), so with rate 1 everywhere
        and 1 output per frame each case comes once every 4 frames (in a random order). A frame where no case is drawn gets no output.
        The draws are seeded by the settings and the frame names, so a dry run counts exactly what the job makes."""
        rng = random.Random(json.dumps([self.engine, self.cases, self.variants, self.rates,
                                        {kind: [image.name for image in images] for kind, images in self.inputs.items()}]))
        tasks, frame_index = [], 0
        for kind, images in self.inputs.items():
            takes = [c for c in self.cases if kind in ENGINES[self.engine]["cases"][c]["changes"]]
            used = dict.fromkeys(takes, 0)
            for image in images:
                drawn = [c for c in takes if rng.random() < self.rates[c]]
                rng.shuffle(drawn)
                drawn = sorted(drawn, key=lambda c: used[c])[:self.variants]     # stable: ties stay shuffled
                for k, case_id in enumerate(sorted(drawn, key=takes.index)):
                    used[case_id] += 1
                    name = f"{image.stem}__{case_id}__v1.png"
                    tasks.append({"id": f"t{len(tasks) + 1:04d}", "input": image.name, "input_kind": kind,
                                  "case": case_id, "case_title": ENGINES[self.engine]["cases"][case_id]["title"],
                                  "variant": 1, "seed": frame_index * self.variants + k, "status": "queued",
                                  "output_name": name, "path": f"{kind}/{name}"})
                frame_index += 1
        return tasks

    def image(self, task: dict) -> Path:
        return self.folder / "inputs" / task["input_kind"] / task["input"]

    # ---- views ----
    def public(self) -> dict:
        with self.lock:
            tasks = copy.deepcopy(self.tasks)
            spent = tasks + copy.deepcopy(self.deleted)      # a deleted output's tokens were still used
        counts = {s: sum(t["status"] == s for t in tasks) for s in ("queued", "running", "done", "failed", "interrupted")}
        usage = summarize(a["usage"] for t in spent for a in t["token_attempts"])
        used = [t["tokens"] for t in spent if t["token_attempts"]]
        pending = counts["queued"] + counts["running"]
        per_output = token_stats()
        excluded = sorted(self.excluded)
        kept = [t for t in tasks if t["status"] == "done" and f"{t['input_kind']}/{t['input']}" not in self.excluded]
        review = {"good": sum(t.get("review") == "good" for t in kept), "bad": sum(t.get("review") == "bad" for t in kept),
                  "unreviewed": sum(not t.get("review") for t in kept), "outputs": len(kept)}
        return {
            "id": self.id, "created": self.created, "engine": self.engine,
            "engine_name": ENGINES[self.engine]["name"], "cases": self.cases, "variants": self.variants,
            "rates": self.rates,
            "rates": self.rates, "inputs": [{"kind": kind, "name": image.name, "url": f"/api/jobs/{self.id}/inputs/{kind}/{image.name}",
                        "thumb": f"/api/jobs/{self.id}/input-thumbs/{kind}/{image.name}"}
                       for kind, images in self.inputs.items() for image in images],
            "notes": self.notes, "total": len(tasks), "counts": counts, "finished": pending == 0,
            "review": review, "excluded_inputs": excluded, "deleted": len(spent) - len(tasks),
            "tokens": usage | {"used": sum(used), "outputs_counted": len(used),
                       "estimate_total": estimate(len(tasks)), "estimate_left": estimate(pending),
                       "per_output": per_output},
            "tasks": tasks,
        }

    def summary(self) -> dict:
        state = self.public()
        return {key: state[key] for key in ("id", "created", "engine_name", "cases", "variants", "total", "counts",
                                            "finished")} | {"inputs": len(state["inputs"]),
                                                            "tokens_used": state["tokens"]["used"],
                                                            "token_usage": state["tokens"]}

    # ---- disk ----
    def save(self) -> None:
        """job.json, written whole and then renamed into place, so a reader never sees half of it."""
        with self.lock:
            data = {"id": self.id, "created": self.created, "engine": self.engine, "cases": self.cases,
                    "variants": self.variants, "rates": self.rates, "notes": self.notes,
                    "inputs": {kind: [image.name for image in images] for kind, images in self.inputs.items()},
                    "tasks": [dict(task) for task in self.tasks], "excluded_inputs": sorted(self.excluded),
                    "deleted_tasks": [dict(task) for task in self.deleted]}
            temporary = self.folder / f"job.json.{threading.get_ident()}.tmp"
            temporary.write_text(json.dumps(data, indent=1))
            os.replace(temporary, self.folder / "job.json")

    @classmethod
    def load(cls, folder: Path) -> "Job":
        data = json.loads((folder / "job.json").read_text())
        inputs = {kind: [folder / "inputs" / kind / name for name in names] for kind, names in data["inputs"].items()}
        needs_migration = any("token_attempts" not in t for t in data["tasks"])
        job = cls(folder, data["engine"], data["cases"], data["variants"], inputs, data.get("notes", []),
                  data.get("created"), data["tasks"], data.get("excluded_inputs"), data.get("deleted_tasks"),
                  data.get("rates"))
        changed = needs_migration
        for task in job.tasks:       # the service stopped while these were waiting or running: they are not rerun
            if task["status"] in ("queued", "running"):
                task["status"] = "interrupted"
                for attempt in task["token_attempts"]:
                    if attempt.get("status") == "running":
                        attempt["status"] = "interrupted"
                        if attempt.get("log"):
                            attempt["usage"] = read_usage(folder / attempt["log"])
                update_task(task)
                task["error"] = "The service stopped before this output was made; it was not run again."
                changed = True
        if changed:
            job.save()
        return job

    # ---- live updates ----
    def notify(self, message: dict) -> None:
        """Send message to every subscriber, from any thread."""
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def _broadcast(self, message: dict) -> None:
        for socket in list(self.subscribers):
            try:
                await socket.send_json(message)
            except Exception:             # a closed socket; its handler removes it
                self.subscribers.discard(socket)


# ---------- tokens ----------
_token_lock = threading.Lock()
_token_samples: list[int] = []


def remember_tokens(count: int | None) -> None:
    if isinstance(count, int):
        with _token_lock:
            _token_samples.append(count)


def load_token_history() -> None:
    """Every Astra token count recorded so far: the tasks of stored jobs and the Astra logs under work/."""
    counts = []
    for log in (ROOT / "work").rglob("astra.log"):
        count = read_usage(log).get("input_tokens")
        if count is not None:
            counts.append(count)
    with _token_lock:
        _token_samples[:] = counts


def token_stats() -> dict:
    with _token_lock:
        samples = sorted(_token_samples)
    if len(samples) < 5:
        return {"median": FALLBACK_TOKENS, "low": FALLBACK_TOKENS // 2, "high": FALLBACK_TOKENS * 2,
                "samples": len(samples), "basis": "first measurements"}
    deciles = statistics.quantiles(samples, n=10)
    return {"median": int(statistics.median(samples)), "low": int(deciles[0]), "high": int(deciles[-1]),
            "samples": len(samples), "basis": "recorded runs"}


def estimate(outputs: int) -> dict:
    stats = token_stats()
    return {"outputs": outputs, "median": outputs * stats["median"], "low": outputs * stats["low"],
            "high": outputs * stats["high"]}


# ---------- running ----------
def submit(job: Job, task: dict) -> None:
    """Queue a task in the pool. Its ticket lets a cancel (and a resume after it) leave the older submit idle."""
    with job.lock:
        ticket = job.tickets[task["id"]] = job.tickets.get(task["id"], 0) + 1
    executor.submit(run_task, job, task, ticket)


def run_task(job: Job, task: dict, ticket: int) -> None:
    """One task in a pool thread: one edit_frame call, its state sent and saved before and after."""
    with job.lock:
        if task["status"] != "queued" or job.tickets.get(task["id"]) != ticket:
            return                        # cancelled while it waited, or queued again by a later submit
        task["status"] = "running"
        task["started_at"] = time.time()
        attempt_number = len(task["token_attempts"]) + 1
        run = job.folder / "runs" / task["input_kind"] / task["output_name"].removesuffix(".png") / f"attempt-{attempt_number:04d}"
        attempt = {"usage": legacy_usage(None), "status": "running", "started_at": task["started_at"],
                   "log": str((run / "astra.log").relative_to(job.folder))}
        task["token_attempts"].append(attempt)
        update_task(task)
        view = copy.deepcopy(task)
    job.save()
    job.notify({"type": "task", "task": view})
    case = ENGINES[job.engine]["cases"][task["case"]]
    kind = task["input_kind"]
    change = case["changes"][kind]
    if task.get("note"):                  # a rebuild's own request for this image comes after the case's change
        change += (f" Additional request for this image, which takes priority over anything above it conflicts "
                   f"with: {task['note']}")
    try:
        row = edit_frame(job.image(task), run,
                         job.folder / "outputs" / task["path"], task["seed"], change,
                         INPUT_KINDS[kind]["scene"], refs=case["refs"], context=prompt_context(job.engine))
    except Exception as error:            # edit_frame reports its own errors; this guards the pool
        row = {"error": f"{type(error).__name__}: {error}"}
    remember_tokens((row.get("token_usage") or legacy_usage(row.get("tokens"))).get("input_tokens"))
    with job.lock:
        if row.get("output"):
            task["status"] = "done"
            task["url"] = f"/api/jobs/{job.id}/outputs/{task['path']}"
            task["thumb"] = f"/api/jobs/{job.id}/thumbs/{task['path']}"
        else:
            task["status"] = "failed"
            task["error"] = row.get("error", "no output")
        attempt.update(usage=row.get("token_usage") or legacy_usage(row.get("tokens")),
                       status=task["status"], seconds=row.get("seconds"))
        update_task(task)
        task["seconds"] = row.get("seconds", round(time.time() - task["started_at"]))
        view = copy.deepcopy(task)
        finished = all(t["status"] not in ("queued", "running") for t in job.tasks)
    job.save()
    job.notify({"type": "task", "task": view})
    state = job.public()
    job.notify({"type": "tokens", "tokens": state["tokens"]})
    job.notify({"type": "review", "review": state["review"]})     # a new output joins the unreviewed ones
    if finished:
        job.notify({"type": "job", "job": job.public()})
        USAGE.refresh(force=True)         # a calibration point: the percent used once these outputs are made


@app.on_event("startup")
def load_jobs() -> None:
    load_token_history()
    if not JOBS_DIR.is_dir():
        return
    for folder in sorted(JOBS_DIR.iterdir()):
        if (folder / "job.json").is_file() and folder.name not in jobs:
            try:
                jobs[folder.name] = Job.load(folder)
            except (OSError, ValueError, KeyError) as error:
                print(f"job {folder.name} not loaded: {error}", flush=True)


# ---------- inputs ----------
def safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "image"
    return stem + Path(name).suffix.lower()


def unique(path: Path) -> Path:
    n, target = 1, path
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


def keep_inputs(inputs: dict[str, list[Path]], job_folder: Path) -> dict[str, list[Path]]:
    """Every input frame as a file of the job, in inputs/<kind>/ (copied when it is elsewhere), so the job keeps
    its inputs whatever happens to the folder they came from."""
    kept: dict[str, list[Path]] = {}
    for kind, images in inputs.items():
        folder = job_folder / "inputs" / kind
        folder.mkdir(parents=True, exist_ok=True)
        for image in images:
            if image.parent == folder:
                kept.setdefault(kind, []).append(image)
                continue
            target = unique(folder / safe_name(image.name))
            shutil.copy2(image, target)
            kept.setdefault(kind, []).append(target)
    return kept


def get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return job


# ---------- routes ----------
@app.get("/")
def index() -> FileResponse:
    # no-store: after a restart with a new page, a reload never runs the old script from the browser cache
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/codex")
async def codex_status(refresh: bool = False) -> dict:
    """The Codex plan's usage limit, the calibration and the room left in outputs after the queued ones."""
    return await asyncio.to_thread(USAGE.status, pending_outputs(), refresh) | {"enforced": ENFORCE_CODEX_LIMIT}


@app.get("/api/engines")
def engines() -> dict:
    return catalogue() | {"tokens_per_output": token_stats()}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [job.summary() for job in sorted(jobs.values(), key=lambda j: j.created, reverse=True)]


@app.post("/api/jobs")
async def create_job(request: Request) -> dict:
    """Form fields: engine, cases (repeated), variants (at most this many outputs per frame), rate:<case id> (0-1, the chance
    the case is drawn per frame, default 1), and the input as ONE of: folder (a path on the server),
    zip (a zip file), files:<input kind> (images, repeated, for each kind). With dry_run=1 nothing is kept or run:
    the answer is the frames found, the outputs a job would make and their token estimate."""
    form = await request.form()
    dry_run = str(form.get("dry_run", "")) in ("1", "true")
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
    rates = {}
    for case_id in cases:                 # rate:<case id>, the chance the case is drawn per frame; 1 when not given
        try:
            rates[case_id] = float(form.get(f"rate:{case_id}", 1))
        except ValueError:
            rates[case_id] = -1.0
        if not 0 <= rates[case_id] <= 1:
            raise HTTPException(400, f"the rate of {case_id} must be 0-1")
    folder = str(form.get("folder", "")).strip()
    archive = form.get("zip")
    archive = archive if isinstance(archive, UploadFile) and archive.filename else None
    uploads = {kind: [f for f in form.getlist(f"files:{kind}") if isinstance(f, UploadFile) and f.filename]
               for kind in INPUT_KINDS}
    uploads = {kind: files for kind, files in uploads.items() if files}
    if sum((bool(folder), archive is not None, bool(uploads))) != 1:
        raise HTTPException(400, "give the input one way: a folder path, a zip, or images uploaded per input kind")

    job_folder = JOBS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    scratch = job_folder if not dry_run else JOBS_DIR / f".dry-{uuid.uuid4().hex[:8]}"
    notes: list[str] = []
    try:
        if folder:
            source = Path(folder).expanduser()
            if not source.is_absolute():
                source = ROOT / source
            if not source.is_dir():
                raise HTTPException(400, f"no such folder on the server: {source}")
            inputs, notes = kind_folders(source)
        elif archive is not None:
            unpacked = scratch / "inputs"
            unpacked.mkdir(parents=True)
            unpack_zip(await archive.read(), unpacked)
            inputs, notes = kind_folders(unpacked)
        else:
            inputs = {}
            for kind, files in uploads.items():
                (scratch / "inputs" / kind).mkdir(parents=True, exist_ok=True)
                for upload in files:
                    name = safe_name(upload.filename)
                    if Path(name).suffix not in IMAGE_TYPES:
                        continue
                    target = unique(scratch / "inputs" / kind / name)
                    target.write_bytes(await upload.read())
                    inputs.setdefault(kind, []).append(target)

        # each input kind feeds one engine: the job uses the chosen engine's kinds only
        for kind, images in inputs.items():
            if INPUT_KINDS[kind]["engine"] != engine and images:
                notes.append(f"{len(images)} {kind} image(s) left out: that kind feeds "
                             f"{ENGINES[INPUT_KINDS[kind]['engine']]['name']}")
        inputs = {kind: images for kind, images in inputs.items() if INPUT_KINDS[kind]["engine"] == engine and images}
        if not inputs:
            kinds = [kind for kind, k in INPUT_KINDS.items() if k["engine"] == engine]
            raise HTTPException(400, f"no images for {ENGINES[engine]['name']}: it takes {', '.join(kinds)} inputs"
                                     + (f" ({'; '.join(notes)})" if notes else ""))
        planned = Job(scratch, engine, cases, variants, inputs, notes, rates=rates)
        if not planned.tasks:
            raise HTTPException(400, "none of the chosen cases takes these input kinds, or their rates drew no case")
        if len(planned.tasks) > MAX_TASKS:
            raise HTTPException(400, f"{len(planned.tasks)} outputs is over {MAX_TASKS}; narrow the input")
        # the Codex plan's usage limit: when enforced, a job that would not fit is refused unless the request says force
        fit = await codex_fit(len(planned.tasks))
        if dry_run:
            return {"frames": {kind: len(images) for kind, images in inputs.items()}, "outputs": len(planned.tasks),
                    "notes": notes, "tokens": estimate(len(planned.tasks)) | {"per_output": token_stats()},
                    "codex": fit}
        if not fit["fits"] and str(form.get("force", "")) not in ("1", "true"):
            raise HTTPException(409, {"message": fit["reason"], "codex": fit})
    finally:
        if dry_run:
            shutil.rmtree(scratch, ignore_errors=True)

    job_folder.mkdir(parents=True, exist_ok=True)
    job = Job(job_folder, engine, cases, variants, keep_inputs(inputs, job_folder), notes, rates=rates)
    job.loop = asyncio.get_running_loop()
    job.save()
    await asyncio.to_thread(USAGE.refresh, True)      # a calibration point: the percent used before these outputs
    jobs[job.id] = job
    for task in job.tasks:
        submit(job, task)
    return job.public()


MAX_NOTE = 1000


@app.post("/api/jobs/{job_id}/resume")
async def resume(job_id: str, request: Request) -> dict:
    """Run again the outputs a job did not make: the interrupted ones (the service stopped while they were waiting or
    running), and the failed ones too when failed=1. Each keeps its name, seed and prompt; finished outputs are not
    touched. Form fields: failed, force (to run past the Codex usage limit)."""
    job = get_job(job_id)
    form = await request.form()
    statuses = {"interrupted"} | ({"failed"} if str(form.get("failed", "")) in ("1", "true") else set())
    with job.lock:
        todo = [t for t in job.tasks if t["status"] in statuses]
    if not todo:
        raise HTTPException(400, "nothing to resume: no interrupted" + (" or failed" if "failed" in statuses else "")
                                 + " outputs in this job")
    fit = await codex_fit(len(todo))
    if not fit["fits"] and str(form.get("force", "")) not in ("1", "true"):
        raise HTTPException(409, {"message": fit["reason"], "codex": fit})
    with job.lock:
        for task in todo:
            task["status"] = "queued"
            for key in ("error", "seconds", "started_at", "url", "thumb"):
                task.pop(key, None)
    job.loop = job.loop or asyncio.get_running_loop()
    job.save()
    job.notify({"type": "job", "job": job.public()})
    await asyncio.to_thread(USAGE.refresh, True)      # a calibration point before these outputs
    for task in todo:
        submit(job, task)
    return {"resumed": len(todo), "job": job.public()}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel(job_id: str) -> dict:
    """Cancel every output of the job still waiting for a free thread: each becomes interrupted (Resume runs it
    later), with the same name, seed and prompt. Outputs already running are left to finish."""
    job = get_job(job_id)
    with job.lock:
        todo = [t for t in job.tasks if t["status"] == "queued"]
        for task in todo:
            task["status"] = "interrupted"
            task["error"] = "Cancelled before it ran; Resume runs it."
    if not todo:
        raise HTTPException(400, "nothing to cancel: no outputs of this job are waiting")
    job.save()
    state = job.public()
    job.notify({"type": "job", "job": state})
    return {"cancelled": len(todo), "job": state}


def rebuild_note(form) -> str:
    """The user's request for a rebuild, on one line: required, at most MAX_NOTE characters."""
    note = " ".join(str(form.get("note", "")).split())
    if not note:
        raise HTTPException(400, "write what to change in this image")
    if len(note) > MAX_NOTE:
        raise HTTPException(400, f"the request is over {MAX_NOTE} characters")
    return note


def rebuild_sources(job: Job, task_ids: list[str], action: str = "rebuild") -> list[dict]:
    """The outputs to rebuild (or delete), in the order given: each must be a task of the job no longer being made."""
    with job.lock:
        by_id = {t["id"]: t for t in job.tasks}
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            raise HTTPException(404, f"no task {', '.join(missing)} in job {job.id}")
        busy = [task_id for task_id in task_ids if by_id[task_id]["status"] in ("queued", "running")]
        if busy:
            raise HTTPException(409, f"still being made, {action} once done: {', '.join(busy)}")
        return [by_id[task_id] for task_id in task_ids]


def new_rebuild_task(job: Job, source: dict, note: str) -> dict:
    """One more task for the source's frame and case, appended to the job (the caller holds job.lock). Rounds count
    from the first output, so a rebuild of a rebuild is ...__r2.png, ...__r3.png."""
    origin = source.get("rebuild_of") or source["id"]
    rounds = 1 + sum(t.get("rebuild_of") == origin for t in job.tasks + job.deleted)     # never reuse a name
    stem = source["output_name"].removesuffix(".png").split("__r")[0]
    name = f"{stem}__r{rounds}.png"
    task = {"id": f"t{len(job.tasks) + len(job.deleted) + 1:04d}", "input": source["input"],
            "input_kind": source["input_kind"],
            "case": source["case"], "case_title": source["case_title"], "variant": source["variant"],
            "seed": source["seed"] + 1000 * rounds, "status": "queued", "output_name": name,
            "path": f"{source['input_kind']}/{name}", "rebuild_of": origin, "rebuild_round": rounds,
            "rebuilt_from": source["id"], "note": note}
    migrate_task(task)
    job.tasks.append(task)
    return task


async def queue_rebuilds(job: Job, sources: list[dict], note: str, force: bool) -> list[dict]:
    """Check the Codex room for len(sources) outputs (409 unless forced), then add and start one rebuild each."""
    fit = await codex_fit(len(sources))
    if not fit["fits"] and not force:
        raise HTTPException(409, {"message": fit["reason"], "codex": fit})
    with job.lock:
        tasks = [new_rebuild_task(job, source, note) for source in sources]
    job.loop = job.loop or asyncio.get_running_loop()
    job.save()
    job.notify({"type": "job", "job": job.public()})
    for task in tasks:
        submit(job, task)
    return [dict(task) for task in tasks]


@app.post("/api/jobs/{job_id}/tasks/{task_id}/rebuild")
async def rebuild(job_id: str, task_id: str, request: Request) -> dict:
    """One more output for the same frame and case, with the user's own request for that image added to the prompt.
    The earlier output stays; the new one is a task of the same job (output name ...__r<n>.png) and runs in the
    shared pool. Form fields: note (the request, required), force (to run past the Codex usage limit)."""
    job = get_job(job_id)
    form = await request.form()
    note = rebuild_note(form)
    sources = rebuild_sources(job, [task_id])
    tasks = await queue_rebuilds(job, sources, note, str(form.get("force", "")) in ("1", "true"))
    return {"task": tasks[0], "job": job.public()}


@app.post("/api/jobs/{job_id}/rebuild")
async def rebuild_many(job_id: str, request: Request) -> dict:
    """One more output for each chosen output, all with the same request (a reviewer's feedback on a set of outputs).
    Each is a rebuild as above; nothing is added when any choice is refused. Form fields: task (one per chosen
    output, repeated; a repeat is made once), note (required), force (to run past the Codex usage limit)."""
    job = get_job(job_id)
    form = await request.form()
    note = rebuild_note(form)
    task_ids = list(dict.fromkeys(str(value) for value in form.getlist("task")))
    if not task_ids:
        raise HTTPException(400, "choose at least one output to rebuild")
    sources = rebuild_sources(job, task_ids)
    tasks = await queue_rebuilds(job, sources, note, str(form.get("force", "")) in ("1", "true"))
    return {"tasks": tasks, "job": job.public()}


@app.post("/api/jobs/{job_id}/delete")
async def delete_outputs(job_id: str, request: Request) -> dict:
    """Delete the chosen outputs (form field task, repeated): each leaves the page, the review and the downloads, and
    its image and thumbnail are removed. Its run folder and record stay (in deleted_tasks of job.json), so the tokens it
    used still count and no later rebuild reuses its name. An output still waiting or running is refused (cancel it
    first); nothing is deleted when any choice is refused."""
    job = get_job(job_id)
    form = await request.form()
    task_ids = list(dict.fromkeys(str(value) for value in form.getlist("task")))
    if not task_ids:
        raise HTTPException(400, "choose at least one output to delete")
    chosen = rebuild_sources(job, task_ids, "delete")
    with job.lock:
        gone = {task["id"] for task in chosen}
        job.tasks = [task for task in job.tasks if task["id"] not in gone]
        for task in chosen:
            task["deleted_at"] = time.time()
            job.deleted.append(task)
    for task in chosen:
        (job.folder / "outputs" / task["path"]).unlink(missing_ok=True)
        (job.folder / "thumbs" / task["input_kind"] / f"{Path(task['output_name']).stem}.jpg").unlink(missing_ok=True)
    job.save()
    state = job.public()
    job.notify({"type": "job", "job": state})
    return {"deleted": len(chosen), "job": state}


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


def job_input(job_id: str, kind: str, name: str) -> tuple[Job, Path]:
    job = get_job(job_id)
    if name not in {image.name for image in job.inputs.get(kind, [])}:
        raise HTTPException(404, "no such input")
    return job, job.folder / "inputs" / kind / name


def thumb_of(source: Path, thumb: Path) -> FileResponse:
    if not thumb.is_file():
        thumb.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            small = image.convert("RGB")
            small.thumbnail((640, 360))
            small.save(thumb, quality=82)
    return FileResponse(thumb, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/outputs/{kind}/{name}")
def output(job_id: str, kind: str, name: str) -> FileResponse:
    _, path = done_output(job_id, kind, name)
    return FileResponse(path, media_type="image/png", filename=name)


@app.get("/api/jobs/{job_id}/thumbs/{kind}/{name}")
def thumbnail(job_id: str, kind: str, name: str) -> FileResponse:
    """A small JPEG of an output for the UI grid (the PNGs are full frames of a few MB)."""
    job, path = done_output(job_id, kind, name)
    return thumb_of(path, job.folder / "thumbs" / kind / f"{Path(name).stem}.jpg")


@app.get("/api/jobs/{job_id}/inputs/{kind}/{name}")
def input_image(job_id: str, kind: str, name: str) -> FileResponse:
    _, path = job_input(job_id, kind, name)
    return FileResponse(path, filename=name)


@app.get("/api/jobs/{job_id}/input-thumbs/{kind}/{name}")
def input_thumbnail(job_id: str, kind: str, name: str) -> FileResponse:
    job, path = job_input(job_id, kind, name)
    return thumb_of(path, job.folder / "thumbs" / "_inputs" / kind / f"{Path(name).stem}.jpg")


@app.post("/api/jobs/{job_id}/tasks/{task_id}/review")
async def review_task(job_id: str, task_id: str, request: Request) -> dict:
    """Mark a finished output good or bad (verdict=good|bad), or clear its mark (verdict=clear)."""
    job = get_job(job_id)
    verdict = str((await request.form()).get("verdict", ""))
    if verdict not in ("good", "bad", "clear"):
        raise HTTPException(400, "verdict must be good, bad or clear")
    with job.lock:
        task = next((t for t in job.tasks if t["id"] == task_id), None)
        if task is None:
            raise HTTPException(404, f"no task {task_id} in job {job_id}")
        if task["status"] != "done":
            raise HTTPException(409, "only a finished output can be reviewed")
        if verdict == "clear":
            task.pop("review", None)
        else:
            task["review"] = verdict
        view = dict(task)
    job.loop = job.loop or asyncio.get_running_loop()
    job.save()
    job.notify({"type": "task", "task": view})
    job.notify({"type": "review", "review": job.public()["review"]})
    return {"task": view, "review": job.public()["review"]}


@app.post("/api/jobs/{job_id}/inputs/{kind}/{name}/exclude")
async def exclude_input(job_id: str, kind: str, name: str, request: Request) -> dict:
    """Take an input frame out of the job's review and downloads (excluded=1), or put it back (excluded=0). Its
    outputs stay on disk."""
    job, _ = job_input(job_id, kind, name)
    excluded = str((await request.form()).get("excluded", "1")) in ("1", "true")
    key = f"{kind}/{name}"
    with job.lock:
        (job.excluded.add if excluded else job.excluded.discard)(key)
    job.loop = job.loop or asyncio.get_running_loop()
    job.save()
    state = job.public()
    job.notify({"type": "job", "job": state})
    return {"excluded_inputs": state["excluded_inputs"], "review": state["review"]}


def selected_outputs(job: Job, which: str) -> list[dict]:
    """The finished outputs a download takes: all, or only those marked good / bad; never those of excluded inputs."""
    with job.lock:
        tasks = [dict(t) for t in job.tasks if t["status"] == "done"
                 and f"{t['input_kind']}/{t['input']}" not in job.excluded]
    if which == "good":
        return [t for t in tasks if t.get("review") == "good"]
    if which == "bad":
        return [t for t in tasks if t.get("review") == "bad"]
    return tasks


@app.get("/api/jobs/{job_id}/download.zip")
def download_all(job_id: str, filter: str = "all", layout: str = "flat") -> FileResponse:
    """A zip of the job's outputs: filter=all|good|bad, layout=flat (one folder) or by_case (a folder per violation
    case). Written to a temporary file (a job's PNGs can be hundreds of MB) that is deleted once sent."""
    job = get_job(job_id)
    if filter not in ("all", "good", "bad") or layout not in ("flat", "by_case"):
        raise HTTPException(400, "filter must be all, good or bad, and layout flat or by_case")
    tasks = selected_outputs(job, filter)
    if not tasks:
        raise HTTPException(404, {"all": "no outputs yet", "good": "no outputs marked good yet",
                                  "bad": "no outputs marked bad yet"}[filter])
    handle, temporary = tempfile.mkstemp(prefix=f"{job.id}_", suffix=".zip", dir=JOBS_DIR)
    os.close(handle)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:     # PNGs are compressed already
        for task in tasks:
            name = task["output_name"] if layout == "flat" else f"{task['case']}/{task['output_name']}"
            archive.write(job.folder / "outputs" / task["path"], name)
    label = f"{job.id}_{filter}{'_by_case' if layout == 'by_case' else ''}.zip"
    return FileResponse(temporary, media_type="application/zip", filename=label,
                        background=BackgroundTask(os.remove, temporary))


@app.websocket("/ws/jobs/{job_id}")
async def job_socket(socket: WebSocket, job_id: str) -> None:
    job = jobs.get(job_id)
    await socket.accept()
    if job is None:
        await socket.send_json({"type": "error", "error": f"no job {job_id}"})
        await socket.close()
        return
    job.loop = job.loop or asyncio.get_running_loop()   # a job loaded from disk gets the loop on its first watcher
    job.subscribers.add(socket)
    await socket.send_json({"type": "job", "job": job.public()})   # the state so far, then every change
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        job.subscribers.discard(socket)
