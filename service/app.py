"""Synthetic Data Generation service: a web UI over synth.try_astra_edit_batch.edit_frame.

Run: .venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8000   (then open http://127.0.0.1:8000)
The database (MySQL, configured in .env; service.store) holds the projects, the engines each project may use, the
engine catalogue (input kinds, cases and every prompt text, edited on the Engines and Prompts pages) and which
project each job belongs to.
Inputs are sorted by input kind (what the frame shows, e.g. a forklift with an LSP and cargo), given in one of three
ways: a folder path on this machine (free text) whose sub-folders are named by kind, a zip of such a folder, or images
uploaded per kind. A job belongs to one project and takes one engine the project may use and some of its cases: each
input kind feeds one engine, so only frames of that engine's kinds are used, and every chosen case that takes a
frame's kind makes exactly ONE output of that frame (user, 2026-10-09: no outputs-per-frame setting, no case rates),
each with one Astra edit. Tasks of all
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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile

from service import job_records
from service import reference_store
from service import store as db
from service.catalogue_api import router as catalogue_router
from service.codex_status import CodexUsage
from service.settings import database_url
from synth.token_usage import legacy_usage, migrate_task, read_usage, summarize, update_task
from synth.try_astra_edit_batch import IMAGE_TYPES, ROOT, edit_frame, frames, reference_labels, with_labels

JOBS_DIR = ROOT / "work/service/jobs"
STATIC = Path(__file__).resolve().parent / "static"
WORKERS = 3
PAGES = {"": "index.html", "projects": "projects.html", "engines": "engines.html", "prompts": "prompts.html",
         "references": "references.html"}
MAX_TASKS = 500
# the Codex usage limit only warns (user, 2026-10-09: do not block generating when the quota is used up); with
# ENFORCE_CODEX_LIMIT=1 in the environment a job, resume or rebuild over it is refused (409) unless forced
ENFORCE_CODEX_LIMIT = os.environ.get("ENFORCE_CODEX_LIMIT") == "1"
# median input tokens of one output (cached ones included) over the 152 fully counted runs of 2026-10-08/09
FALLBACK_TOKENS = 124_000

app = FastAPI(title="Synthetic Data Generation")


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
                    "headers": [(b"www-authenticate", b'Basic realm="Synthetic Data Generation"'),
                                (b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"Sign in to use Synthetic Data Generation."})


app.add_middleware(BasicAuth)
app.include_router(catalogue_router)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(db.StoreError)
async def store_error(_: Request, error: db.StoreError) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=error.status)


def catalog() -> dict:
    """The engines, input kinds, cases and prompts as the database holds them now (service.store.Store.catalogue)."""
    return db.current().catalogue()


def engine_name(engine_id: str) -> str:
    """The engine's display name; its id when the database cannot be read (a job view must not fail on it)."""
    try:
        return db.current().engine_name(engine_id)
    except Exception:
        return engine_id

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
    def __init__(self, folder: Path, engine: str, cases: list[str], inputs: dict[str, list[Path]],
                 notes: list[str], project_id: int | None = None, created: float | None = None,
                 tasks: list[dict] | None = None, excluded: list[str] | None = None,
                 deleted: list[dict] | None = None):
        self.id = folder.name
        self.folder = folder
        self.engine, self.cases, self.project_id = engine, cases, project_id
        self.inputs, self.notes = inputs, notes
        self.created = created or time.time()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.lock = threading.Lock()
        self.save_lock = threading.Lock()       # one save at a time, so an older state never lands last
        self.db_sent: dict[str, str] = {}       # task id -> what the database holds of it (job_records)
        self.subscribers: set[WebSocket] = set()
        self.tasks = tasks if tasks is not None else self._plan()
        for task in self.tasks:
            migrate_task(task)
        self.excluded: set[str] = set(excluded or [])     # "<kind>/<name>" inputs the reviewer removed (X)
        # outputs the user deleted: off the page and out of downloads, kept here for their tokens, ids and names
        self.deleted: list[dict] = deleted or []
        self.tickets: dict[str, int] = {}       # task id -> its latest submit; an older (cancelled) submit does nothing

    def _plan(self) -> list[dict]:
        """ONE output per frame and chosen case that takes the frame's kind (user, 2026-10-09), in the order the
        cases are listed; each gets its own seed."""
        cases = catalog()["engines"][self.engine]["cases"]
        tasks = []
        for kind, images in self.inputs.items():
            takes = [c for c in self.cases if kind in cases[c]["changes"]]
            for image in images:
                for case_id in takes:
                    name = f"{image.stem}__{case_id}__v1.png"
                    tasks.append({"id": f"t{len(tasks) + 1:04d}", "input": image.name, "input_kind": kind,
                                  "case": case_id, "case_title": cases[case_id]["title"],
                                  "variant": 1, "seed": len(tasks), "status": "queued",
                                  "output_name": name, "path": f"{kind}/{name}"})
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
            "engine_name": engine_name(self.engine), "cases": self.cases, "project_id": self.project_id,
            "inputs": [{"kind": kind, "name": image.name, "url": f"/api/jobs/{self.id}/inputs/{kind}/{image.name}",
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
        return {key: state[key] for key in ("id", "created", "engine", "engine_name", "cases", "project_id",
                                            "total", "counts", "finished")} | {"inputs": len(state["inputs"]),
                                                            "tokens_used": state["tokens"]["used"],
                                                            "token_usage": state["tokens"]}

    # ---- disk and database ----
    def state(self) -> dict:
        """Everything about the job but its files: the job.json dict (the caller holds self.lock)."""
        return {"id": self.id, "created": self.created, "engine": self.engine, "cases": self.cases,
                "project_id": self.project_id, "notes": self.notes,
                "inputs": {kind: [image.name for image in images] for kind, images in self.inputs.items()},
                "tasks": [copy.deepcopy(task) for task in self.tasks], "excluded_inputs": sorted(self.excluded),
                "deleted_tasks": [copy.deepcopy(task) for task in self.deleted]}

    def save(self, strict: bool = False) -> None:
        """The job's full state into the database (its row and the tasks changed since the last save) and into
        job.json, written whole and then renamed into place. A database error is logged and the tasks it missed are
        sent with the next save; with strict it is raised (a new job that cannot be recorded is not run)."""
        with self.save_lock:
            with self.lock:
                data = self.state()
            self.folder.mkdir(parents=True, exist_ok=True)
            temporary = self.folder / f"job.json.{threading.get_ident()}.tmp"
            temporary.write_text(json.dumps(data, indent=1))
            os.replace(temporary, self.folder / "job.json")
            sent = {task["id"]: json.dumps([deleted, task], sort_keys=True)
                    for deleted, tasks in ((False, data["tasks"]), (True, data["deleted_tasks"])) for task in tasks}
            changed = [(deleted, task) for deleted, tasks in ((False, data["tasks"]), (True, data["deleted_tasks"]))
                       for task in tasks if self.db_sent.get(task["id"]) != sent[task["id"]]]
            try:
                job_records.save_job(db.current().db, data, changed)
            except Exception as error:
                if strict:
                    raise
                print(f"job {self.id} not saved to the database: {type(error).__name__}: {error}", flush=True)
                return
            self.db_sent = sent

    @classmethod
    def load(cls, folder: Path) -> "Job":
        """A job from its job.json (one the database does not hold yet)."""
        return cls.from_state(folder, json.loads((folder / "job.json").read_text()), in_db=False)

    @classmethod
    def from_state(cls, folder: Path, data: dict, in_db: bool) -> "Job":
        """A job from its state (the job.json dict, from the database or the file); saved again when it changed."""
        inputs = {kind: [folder / "inputs" / kind / name for name in names] for kind, names in data["inputs"].items()}
        needs_migration = any("token_attempts" not in t for t in data["tasks"])
        # jobs made before projects also hold variants and rates: they only shaped the plan, now in data["tasks"]
        job = cls(folder, data["engine"], data["cases"], inputs, data.get("notes", []), data.get("project_id"),
                  data.get("created"), data["tasks"], data.get("excluded_inputs"), data.get("deleted_tasks"))
        if in_db:
            job.db_sent = {task["id"]: json.dumps([deleted, task], sort_keys=True)
                           for deleted, tasks in ((False, job.tasks), (True, job.deleted)) for task in tasks}
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
        if changed and in_db:
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
_token_samples: list[dict] = []         # the token usage of each fully counted Astra run


def complete(usage: dict | None) -> bool:
    """A reading Codex reported in full (input with its cached part, and output). Older logs only kept a partial
    input count (about the uncached part: ~24k against ~124k), so they would make the estimate 5x too low."""
    return bool(usage) and usage.get("source") == "codex_json" and isinstance(usage.get("input_tokens"), int) \
        and isinstance(usage.get("output_tokens"), int)


def remember_tokens(usage: dict | None) -> None:
    if complete(usage):
        with _token_lock:
            _token_samples.append(usage)


def load_token_history() -> None:
    """Every fully counted Astra run so far: the Astra logs under work/ (the stored jobs' runs among them)."""
    usages = [usage for log in (ROOT / "work").rglob("astra.log") if complete(usage := read_usage(log))]
    with _token_lock:
        _token_samples[:] = usages


def token_stats() -> dict:
    """Tokens of one output: the median input (cached included) with its 10th-90th percentile range, and the
    median cached part and output, over the fully counted runs."""
    with _token_lock:
        samples = list(_token_samples)
    if len(samples) < 5:
        return {"median": FALLBACK_TOKENS, "low": FALLBACK_TOKENS * 3 // 4, "high": FALLBACK_TOKENS * 3 // 2,
                "cached_median": None, "output_median": None, "samples": len(samples), "basis": "first measurements"}
    inputs = sorted(u["input_tokens"] for u in samples)
    cached = [u["cached_input_tokens"] for u in samples if isinstance(u.get("cached_input_tokens"), int)]
    deciles = statistics.quantiles(inputs, n=10)
    return {"median": int(statistics.median(inputs)), "low": int(deciles[0]), "high": int(deciles[-1]),
            "cached_median": int(statistics.median(cached)) if cached else None,
            "output_median": int(statistics.median(u["output_tokens"] for u in samples)),
            "samples": len(samples), "basis": "fully counted runs"}


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
    try:
        current = catalog()
    except Exception as error:            # the database is down: the task fails instead of hanging as running
        finish_task(job, task, attempt, {"error": f"the database could not be read: {error}", "seconds": 0})
        return
    kind = task["input_kind"]
    case = current["engines"].get(job.engine, {}).get("cases", {}).get(task["case"])
    if case is None or kind not in case["changes"] or kind not in current["input_kinds"]:
        finish_task(job, task, attempt, {"error": f"case {task['case']} of {job.engine} has no prompt for "
                                                  f"{kind} any more (edited on the Engines page)", "seconds": 0})
        return
    try:                                  # the object references as they are now, named by their image numbers
        references, labels = reference_store.for_case(current, case["refs"])
    except db.StoreError as error:
        finish_task(job, task, attempt, {"error": str(error), "seconds": 0})
        return
    rejected = job.folder / task["rejected_image"] if task.get("rejected_image") else None
    if rejected is not None and rejected.is_file():       # a rebuild: the output it redoes goes in as the last image
        references = references + [{"id": "REJECTED", "text": REJECTED_TEXT, "render": False, "images": [rejected]}]
        labels = reference_labels([(r["id"], len(r["images"]) or 1) for r in references])
    change = with_labels(case["changes"][kind], labels)
    if task.get("note") and "REJECTED" in labels:
        change += (f" This is a redo: {labels['REJECTED']} is an earlier result of this same edit that the reviewer "
                   f"rejected, and their feedback on it is: {task['note']} Fix exactly that in this new edit of image 1, "
                   "and keep everything the change above asks for; where the feedback conflicts with it, the feedback "
                   "wins.")
    elif task.get("note"):                # the rejected output is gone: the feedback alone
        change += (f" Additional request for this image, which takes priority over anything above it conflicts "
                   f"with: {task['note']}")
    try:
        row = edit_frame(job.image(task), run,
                         job.folder / "outputs" / task["path"], task["seed"], change,
                         current["input_kinds"][kind]["scene"], refs=case["refs"],
                         context=current["engines"][job.engine]["prompt"], references=references)
    except Exception as error:            # edit_frame reports its own errors; this guards the pool
        row = {"error": f"{type(error).__name__}: {error}"}
    remember_tokens(row.get("token_usage"))
    finish_task(job, task, attempt, row)


def finish_task(job: Job, task: dict, attempt: dict, row: dict) -> None:
    """Record a task's result row (output or error), then save the job and tell its watchers."""
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
def startup() -> None:
    """Open the database (it fails here, with what is missing, when .env does not configure it), then the jobs."""
    store = db.Store(database_url())
    store.setup()
    db.use(store)
    load_token_history()
    load_jobs(store)


def load_jobs(store: db.Store) -> None:
    """Every job: from the database, which holds each job's full state; then each job on disk the database does not
    hold yet (made before it did) from its job.json, put into the database under the project it names, or Default."""
    states = job_records.job_states(store.db)
    for job_id, state in states.items():
        if state is None or job_id in jobs:
            continue
        try:
            jobs[job_id] = Job.from_state(JOBS_DIR / job_id, state, in_db=True)
        except Exception as error:                # one bad job never stops the service
            print(f"job {job_id} not loaded from the database: {type(error).__name__}: {error}", flush=True)
    if not JOBS_DIR.is_dir():
        return
    known, projects, default = store.job_projects(), {p["id"] for p in store.projects()}, None
    for folder in sorted(JOBS_DIR.iterdir()):
        if not (folder / "job.json").is_file() or folder.name in jobs:
            continue
        try:
            job = Job.load(folder)
            if job.id in known:
                job.project_id = known[job.id]
            elif job.project_id not in projects:
                default = default or store.default_project_id()
                job.project_id = default
            backup = folder / "job.json.before-projects"
            if not backup.exists():               # the first rewrite drops the old keys (variants, rates): keep them
                shutil.copy2(folder / "job.json", backup)
            job.save(strict=True)
        except Exception as error:
            print(f"job {folder.name} not loaded: {type(error).__name__}: {error}", flush=True)
            continue
        jobs[job.id] = job


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
    INPUT_KINDS = catalog()["input_kinds"]

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
@app.get("/{page}")
def index(page: str = "") -> FileResponse:
    """The pages: Generate (/), Projects, Engines and Prompts; the menu on each links them."""
    if page not in PAGES:
        raise HTTPException(404, f"no page {page}")
    # no-store: after a restart with a new page, a reload never runs the old script from the browser cache
    return FileResponse(STATIC / PAGES[page], headers={"Cache-Control": "no-store"})


@app.get("/api/codex")
async def codex_status(refresh: bool = False) -> dict:
    """The Codex plan's usage limit, the calibration and the room left in outputs after the queued ones."""
    return await asyncio.to_thread(USAGE.status, pending_outputs(), refresh) | {"enforced": ENFORCE_CODEX_LIMIT}


def project_engines(project_id: int | None) -> list[str] | None:
    """The engines a project may use (404 for an unknown project); None (every engine) without a project."""
    return None if project_id is None else db.current().project(project_id)["engines"]


@app.get("/api/engines")
def engines(project_id: int | None = None) -> dict:
    """The input kinds, engines and cases without the prompt text, for the Generate page; with project_id only the
    engines that project may use."""
    current, allowed = catalog(), project_engines(project_id)
    shown = [e for e in current["engines"] if allowed is None or e in allowed]
    return {
        "input_kinds": [{"id": kind, "title": k["title"], "engine": k["engine"]}
                        for kind, k in current["input_kinds"].items() if k["engine"] in shown],
        "engines": [{"id": engine_id, "name": engine["name"], "color": engine["color"],
                     "input_kinds": [kind for kind, k in current["input_kinds"].items() if k["engine"] == engine_id],
                     "cases": [{"id": case_id, "title": c["title"], "catalogue": c["catalogue"],
                                "status": c["status"], "input_kinds": list(c["changes"])}
                               for case_id, c in engine["cases"].items()]}
                    for engine_id, engine in current["engines"].items() if engine_id in shown],
        "tokens_per_output": token_stats(),
    }


@app.get("/api/jobs")
def list_jobs(project_id: int | None = None) -> list[dict]:
    return [job.summary() for job in sorted(jobs.values(), key=lambda j: j.created, reverse=True)
            if project_id is None or job.project_id == project_id]


def job_choices(form) -> tuple[int, str, list[str]]:
    """The project, its engine and the chosen cases of a new job's form, each checked against the database."""
    try:
        project_id = int(str(form.get("project_id", "")))
    except ValueError:
        raise HTTPException(400, "choose a project")
    if db.current().project(project_id)["hidden"]:
        raise HTTPException(403, f"project {project_id} is hidden: it takes no new jobs")
    allowed = project_engines(project_id)
    engine = str(form.get("engine", ""))
    engine_cases = catalog()["engines"].get(engine, {}).get("cases")
    if engine_cases is None:
        raise HTTPException(400, f"unknown engine {engine!r}")
    if engine not in allowed:
        raise HTTPException(403, f"project {project_id} may not use engine {engine!r}; add it on the Projects page")
    cases = list(dict.fromkeys(str(c) for c in form.getlist("cases")))
    unknown = [c for c in cases if c not in engine_cases]
    if unknown or not cases:
        raise HTTPException(400, f"choose cases of {engine}" + (f"; unknown: {', '.join(unknown)}" if unknown else ""))
    return project_id, engine, cases


@app.post("/api/jobs")
async def create_job(request: Request) -> dict:
    """Form fields: project_id, engine (one the project may use, else 403), cases (repeated: each makes ONE output
    of every frame of a kind it takes), and the input as ONE of: folder (a path on the server),
    zip (a zip file), files:<input kind> (images, repeated, for each kind). With dry_run=1 nothing is kept or run:
    the answer is the frames found, the outputs a job would make and their token estimate."""
    form = await request.form()
    dry_run = str(form.get("dry_run", "")) in ("1", "true")
    project_id, engine, cases = job_choices(form)
    INPUT_KINDS, engine_label = catalog()["input_kinds"], engine_name(engine)
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
                             f"{engine_name(INPUT_KINDS[kind]['engine'])}")
        inputs = {kind: images for kind, images in inputs.items() if INPUT_KINDS[kind]["engine"] == engine and images}
        if not inputs:
            kinds = [kind for kind, k in INPUT_KINDS.items() if k["engine"] == engine]
            raise HTTPException(400, f"no images for {engine_label}: it takes {', '.join(kinds)} inputs"
                                     + (f" ({'; '.join(notes)})" if notes else ""))
        planned = Job(scratch, engine, cases, inputs, notes, project_id)
        if not planned.tasks:
            raise HTTPException(400, "none of the chosen cases takes these input kinds")
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
    job = Job(job_folder, engine, cases, keep_inputs(inputs, job_folder), notes, project_id)
    job.loop = asyncio.get_running_loop()
    try:
        job.save(strict=True)
    except Exception as error:            # e.g. the project was deleted meanwhile: nothing is run
        shutil.rmtree(job_folder, ignore_errors=True)
        raise HTTPException(409, f"the job could not be recorded: {type(error).__name__}") from error
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


# the output a rebuild redoes, given to Astra after the references with the reviewer's feedback (user, 2026-10-09)
REJECTED_TEXT = ("An earlier result of this same edit of image 1, rejected by a reviewer; the feedback in the change below "
                 "says what is wrong with it. It only shows the mistake to avoid: never edit it, never copy it or start "
                 "from it, and take nothing else from it; edit image 1.")


def new_rebuild_task(job: Job, source: dict, note: str) -> dict:
    """One more task for the source's frame and case, appended to the job (the caller holds job.lock). Rounds count
    from the first output, so a rebuild of a rebuild is ...__r2.png, ...__r3.png. The source's output, when it has
    one, is copied to rebuild_refs/ (kept even if the output is deleted later) and goes in with the feedback."""
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
    made = job.folder / "outputs" / source["path"]
    if source["status"] == "done" and made.is_file():
        kept = job.folder / "rebuild_refs" / source["input_kind"] / f"{Path(name).stem}__rejected{made.suffix}"
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(made, kept)
        task["rejected_image"] = str(kept.relative_to(job.folder))
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
