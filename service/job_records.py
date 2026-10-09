"""Every job's full state in the database: its row in `jobs` (settings, inputs, excluded inputs, counts) and one row
per task in `job_tasks` (its whole record as JSON, plus the columns worth querying). The service loads jobs from
here on start; job.json in each job folder stays a copy on disk.

A save sends only the tasks that changed since the job's last save (the caller keeps what it sent), so a running
job of hundreds of outputs costs one small transaction per state change.
"""

from __future__ import annotations

import json
import re
import time

import sqlalchemy as sa

from service.store import LONG_TEXT, jobs_t, meta, projects_t

job_tasks_t = sa.Table(
    "job_tasks", meta,
    sa.Column("job_id", sa.String(64), sa.ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("task_id", sa.String(32), primary_key=True),
    sa.Column("position", sa.Integer, nullable=False),
    sa.Column("deleted", sa.Boolean, nullable=False),          # an output the user deleted (kept for its tokens)
    sa.Column("input_kind", sa.String(64), nullable=False),
    sa.Column("input_name", sa.String(255), nullable=False),
    sa.Column("case_id", sa.String(64), nullable=False),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("review", sa.String(8)),
    sa.Column("output_path", sa.String(400)),
    sa.Column("tokens", sa.Integer),
    sa.Column("data", LONG_TEXT, nullable=False))               # the task record as the service keeps it


def upgrade(db: sa.Engine) -> None:
    """Add the columns added since to tables made before them (create_all never alters a table): the jobs' state
    columns, projects.hidden."""
    for table in (jobs_t, projects_t):
        have = {column["name"] for column in sa.inspect(db).get_columns(table.name)}
        missing = [column for column in table.columns if column.name not in have]
        if not missing:
            continue
        with db.begin() as conn:
            for column in missing:
                kind = column.type.compile(dialect=db.dialect)
                conn.execute(sa.text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {kind} NULL"))


def task_row(job_id: str, task: dict, deleted: bool) -> dict:
    """position: the number in the task id (t0012), as ids grow in the order tasks are added."""
    position = int(re.sub(r"\D", "", task["id"]) or 0)
    return {"job_id": job_id, "task_id": task["id"], "position": position, "deleted": deleted,
            "input_kind": task["input_kind"], "input_name": task["input"], "case_id": task["case"],
            "status": task["status"], "review": task.get("review"), "output_path": task.get("path"),
            "tokens": task.get("tokens"), "data": json.dumps(task)}


def save_job(db: sa.Engine, state: dict, changed: list[tuple[bool, dict]]) -> None:
    """Write a job's row (state: the job.json dict) and the changed tasks, each (deleted, task)."""
    tasks = state["tasks"]
    counts = {s: sum(t["status"] == s for t in tasks) for s in ("done", "failed", "interrupted")}
    row = {"project_id": state["project_id"], "engine_id": state["engine"], "created": state["created"],
           "cases": json.dumps(state["cases"]), "notes": json.dumps(state["notes"]),
           "inputs": json.dumps(state["inputs"]), "excluded_inputs": json.dumps(state["excluded_inputs"]),
           "total": len(tasks), "waiting": sum(t["status"] in ("queued", "running") for t in tasks),
           "updated": time.time(), **counts}
    with db.begin() as conn:
        if conn.execute(sa.select(jobs_t.c.id).where(jobs_t.c.id == state["id"])).first():
            conn.execute(jobs_t.update().where(jobs_t.c.id == state["id"]).values(**row))
        else:
            conn.execute(jobs_t.insert().values(id=state["id"], **row))
        if not changed:
            return
        ids = [task["id"] for _, task in changed]
        stored = {r[0] for r in conn.execute(sa.select(job_tasks_t.c.task_id).where(
            job_tasks_t.c.job_id == state["id"], job_tasks_t.c.task_id.in_(ids)))}
        for deleted, task in changed:
            values = task_row(state["id"], task, deleted)
            if task["id"] in stored:
                conn.execute(job_tasks_t.update().where(job_tasks_t.c.job_id == state["id"],
                                                        job_tasks_t.c.task_id == task["id"]).values(**values))
            else:
                conn.execute(job_tasks_t.insert().values(**values))


def job_states(db: sa.Engine) -> dict[str, dict | None]:
    """Every job the database knows, as the job.json dict; None for a job registered before its state was kept."""
    with db.connect() as conn:
        rows = conn.execute(sa.select(jobs_t)).mappings().all()
        tasks: dict[str, list] = {}
        for t in conn.execute(sa.select(job_tasks_t.c.job_id, job_tasks_t.c.deleted, job_tasks_t.c.data)
                              .order_by(job_tasks_t.c.job_id, job_tasks_t.c.position)):
            tasks.setdefault(t.job_id, []).append((bool(t.deleted), json.loads(t.data)))
    states: dict[str, dict | None] = {}
    for row in rows:
        if row["cases"] is None:
            states[row["id"]] = None
            continue
        own = tasks.get(row["id"], [])
        states[row["id"]] = {
            "id": row["id"], "created": row["created"], "engine": row["engine_id"], "project_id": row["project_id"],
            "cases": json.loads(row["cases"]), "notes": json.loads(row["notes"] or "[]"),
            "inputs": json.loads(row["inputs"] or "{}"), "excluded_inputs": json.loads(row["excluded_inputs"] or "[]"),
            "tasks": [task for deleted, task in own if not deleted],
            "deleted_tasks": [task for deleted, task in own if deleted]}
    return states
