"""The service database (MySQL in use, any SQLAlchemy URL works): projects, the engines each project may use, the
engine catalogue (engines, their input kinds, their cases = output types) with every prompt text, and which project
each job belongs to.

A job's own state (tasks, reviews, tokens) stays in work/service/jobs/<id>/job.json, written many times a second
while it runs; the database indexes the jobs by project. On first start the catalogue is filled from
synth.violation_cases (the engines, kinds, cases and prompts as they were in code) and a "Default" project with
every engine is made; jobs found on disk that the database does not know join that project.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time

from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.pool import StaticPool

from synth import violation_cases
from synth.try_astra_edit_batch import REFERENCE_TOKEN, WAREHOUSE

SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
STATUSES = ("active", "pending_rule", "draft")
PROMPT_FIELDS = ("setting", "scene_prefix", "objects", "keep")
DEFAULT_PROJECT = "Default"
LONG_TEXT = sa.Text().with_variant(mysql.MEDIUMTEXT(), "mysql")     # TEXT is 64 KB, too small for 20000 utf8mb4 chars

meta = sa.MetaData()
projects_t = sa.Table(
    "projects", meta,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("name", sa.String(120), nullable=False, unique=True),
    sa.Column("description", sa.Text, nullable=False),
    sa.Column("created", sa.Double, nullable=False),
    sa.Column("hidden", sa.Boolean))          # kept with its jobs but off the pages, and no new job (Default)
engines_t = sa.Table(
    "engines", meta,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("name", sa.String(120), nullable=False),
    sa.Column("color", sa.String(7), nullable=False),
    sa.Column("setting", LONG_TEXT, nullable=False),
    sa.Column("scene_prefix", LONG_TEXT, nullable=False),
    sa.Column("objects", LONG_TEXT, nullable=False),
    sa.Column("keep", LONG_TEXT, nullable=False),
    sa.Column("position", sa.Integer, nullable=False))
input_kinds_t = sa.Table(
    "input_kinds", meta,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("engine_id", sa.String(64), sa.ForeignKey("engines.id", ondelete="CASCADE"), nullable=False),
    sa.Column("title", sa.String(160), nullable=False),
    sa.Column("scene", LONG_TEXT, nullable=False),
    sa.Column("position", sa.Integer, nullable=False))
cases_t = sa.Table(
    "cases", meta,
    sa.Column("engine_id", sa.String(64), sa.ForeignKey("engines.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("title", sa.String(160), nullable=False),
    sa.Column("catalogue", sa.String(32), nullable=False),
    sa.Column("status", sa.String(32), nullable=False),
    sa.Column("refs", sa.Text, nullable=False),            # a JSON list of reference ids (references_t)
    sa.Column("position", sa.Integer, nullable=False))
case_prompts_t = sa.Table(       # a case takes an input kind when it has a change prompt for it
    "case_prompts", meta,
    sa.Column("engine_id", sa.String(64), primary_key=True),
    sa.Column("case_id", sa.String(64), primary_key=True),
    sa.Column("input_kind_id", sa.String(64), sa.ForeignKey("input_kinds.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("change_text", LONG_TEXT, nullable=False),
    sa.ForeignKeyConstraint(["engine_id", "case_id"], ["cases.engine_id", "cases.id"], ondelete="CASCADE"))
references_t = sa.Table(        # object references: the images a case adds after the frame (service.reference_store)
    "references", meta,
    sa.Column("id", sa.String(32), primary_key=True),
    sa.Column("title", sa.String(160), nullable=False),
    sa.Column("text", LONG_TEXT, nullable=False),           # how the prompt describes them; may hold {skid}
    sa.Column("images", sa.Text, nullable=False),           # JSON list of file names in work/service/references/<id>/
    sa.Column("render", sa.Boolean, nullable=False),        # with no image: the frame's 3D catalogue render (SKID)
    sa.Column("position", sa.Integer, nullable=False))
project_engines_t = sa.Table(
    "project_engines", meta,
    sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("engine_id", sa.String(64), sa.ForeignKey("engines.id", ondelete="CASCADE"), primary_key=True))
jobs_t = sa.Table(
    "jobs", meta,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id"), nullable=False, index=True),
    sa.Column("engine_id", sa.String(64), nullable=False),
    sa.Column("created", sa.Double, nullable=False),
    # the job's own state (service.job_records): settings, inputs and a count per task status; tasks in job_tasks
    sa.Column("cases", sa.Text),                       # JSON list of case ids
    sa.Column("notes", LONG_TEXT),                     # JSON list
    sa.Column("inputs", LONG_TEXT),                    # JSON {input kind: [file names]}
    sa.Column("excluded_inputs", LONG_TEXT),           # JSON list of "<kind>/<name>"
    sa.Column("total", sa.Integer), sa.Column("done", sa.Integer), sa.Column("failed", sa.Integer),
    sa.Column("waiting", sa.Integer), sa.Column("interrupted", sa.Integer),
    sa.Column("updated", sa.Double))


class StoreError(Exception):
    status = 400


class NotFound(StoreError):
    status = 404


class Conflict(StoreError):
    status = 409


def text(value, field: str, limit: int | None = None, required: bool = True) -> str:
    value = str(value if value is not None else "").strip()
    if required and not value:
        raise StoreError(f"{field} is required")
    if limit and len(value) > limit:
        raise StoreError(f"{field} is over {limit} characters")
    return value


def named_refs(change: str) -> list[str]:
    """The object references a prompt text names ({LSP}, {SKID}, ...), in the order it first names them."""
    return list(dict.fromkeys(REFERENCE_TOKEN.findall(change)))


def check_named(change: str, refs: list[str], where: str) -> None:
    """Every reference a prompt names must be one the output gives, or its image number would be wrong."""
    missing = [ref for ref in named_refs(change) if ref not in refs]
    if missing:
        raise StoreError(f"{where} names {', '.join('{' + r + '}' for r in missing)}, which this output does not "
                         f"give (it gives {', '.join(refs) or 'no reference'}): tick it on the Engines page first")


def slug(value, field: str) -> str:
    value = str(value or "").strip()
    # no "__": output names are <frame>__<case>__v1.png, split on it
    if not SLUG.match(value) or "__" in value:
        raise StoreError(f"{field} must be 1-64 characters of a-z, 0-9, _ and - (no double _), starting with a "
                         "letter or digit")
    return value


class Store:
    def __init__(self, url: str):
        options = ({"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
                   if url.startswith("sqlite") else {"pool_pre_ping": True, "pool_recycle": 3600})
        self.db = sa.create_engine(url, **options)
        if url.startswith("sqlite"):
            sa.event.listen(self.db, "connect", lambda conn, _: conn.execute("PRAGMA foreign_keys=ON"))
        self._lock = threading.Lock()
        self._snapshot: dict | None = None

    @contextmanager
    def _tx(self):
        """A write transaction; a unique or foreign key clash (two people saving at once) is a 409, not a 500."""
        try:
            with self.db.begin() as conn:
                yield conn
        except sa.exc.IntegrityError as error:
            raise Conflict(f"it was changed meanwhile: {error.orig}") from error

    def setup(self) -> None:
        """Create the tables that are missing, and fill a new database's catalogue from synth.violation_cases (a
        database with any engine or project is never filled again, so deleting every engine does not bring them back)."""
        from service import job_records, reference_store      # job_records' table joins meta before create_all

        meta.create_all(self.db)
        job_records.upgrade(self.db)
        with self.db.begin() as conn:         # Default holds the jobs made before projects: hidden (user, 2026-10-09)
            conn.execute(projects_t.update().where(projects_t.c.name == DEFAULT_PROJECT, projects_t.c.hidden.is_(None))
                         .values(hidden=True))
        with self.db.begin() as conn:
            if not (conn.execute(sa.select(sa.func.count()).select_from(engines_t)).scalar()
                    or conn.execute(sa.select(sa.func.count()).select_from(projects_t)).scalar()):
                self._seed(conn)
        reference_store.seed(self)

    def _seed(self, conn) -> None:
        for e_pos, (engine_id, engine) in enumerate(violation_cases.ENGINES.items()):
            prompt = {**WAREHOUSE, **(engine.get("prompt") or {})}
            conn.execute(engines_t.insert().values(id=engine_id, name=engine["name"], color=engine["color"],
                                                   position=e_pos, **{f: prompt[f] for f in PROMPT_FIELDS}))
        for k_pos, (kind_id, kind) in enumerate(violation_cases.INPUT_KINDS.items()):
            conn.execute(input_kinds_t.insert().values(id=kind_id, engine_id=kind["engine"], title=kind["title"],
                                                       scene=kind["scene"], position=k_pos))
        for engine_id, engine in violation_cases.ENGINES.items():
            for c_pos, (case_id, case) in enumerate(engine["cases"].items()):
                conn.execute(cases_t.insert().values(engine_id=engine_id, id=case_id, title=case["title"],
                                                     catalogue=case["catalogue"], status=case["status"],
                                                     refs=json.dumps(case["refs"]), position=c_pos))
                for kind_id, change in case["changes"].items():
                    conn.execute(case_prompts_t.insert().values(engine_id=engine_id, case_id=case_id,
                                                                input_kind_id=kind_id, change_text=change))
        if not conn.execute(sa.select(projects_t.c.id).where(projects_t.c.name == DEFAULT_PROJECT)).first():
            project_id = conn.execute(projects_t.insert().values(
                name=DEFAULT_PROJECT, description="Jobs made before projects existed.", created=time.time(),
                hidden=True)
            ).inserted_primary_key[0]
            for engine_id in violation_cases.ENGINES:
                conn.execute(project_engines_t.insert().values(project_id=project_id, engine_id=engine_id))
        self._changed()

    # ---------- the catalogue the generator reads ----------
    def _changed(self) -> None:
        with self._lock:
            self._snapshot = None

    def catalogue(self) -> dict:
        """{"engines": {id: {name, color, prompt, cases: {id: {title, catalogue, status, refs, changes}}}},
        "input_kinds": {id: {title, scene, engine}}, "references": {id: {title, text, images, render}}}, in their
        set order: the shape of synth.violation_cases."""
        with self._lock:
            if self._snapshot is None:
                self._snapshot = self._read_catalogue()
            return copy.deepcopy(self._snapshot)

    def _read_catalogue(self) -> dict:
        with self.db.connect() as conn:
            engines = conn.execute(sa.select(engines_t).order_by(engines_t.c.position, engines_t.c.id)).mappings()
            result = {"engines": {e["id"]: {"name": e["name"], "color": e["color"], "position": e["position"],
                                            "prompt": {f: e[f] for f in PROMPT_FIELDS}, "cases": {}}
                                  for e in engines},
                      "input_kinds": {}, "references": {}}
            for r in conn.execute(sa.select(references_t).order_by(references_t.c.position, references_t.c.id)).mappings():
                result["references"][r["id"]] = {"title": r["title"], "text": r["text"], "render": bool(r["render"]),
                                                 "images": json.loads(r["images"])}
            for k in conn.execute(sa.select(input_kinds_t).order_by(input_kinds_t.c.position, input_kinds_t.c.id)).mappings():
                result["input_kinds"][k["id"]] = {"title": k["title"], "scene": k["scene"], "engine": k["engine_id"]}
            for c in conn.execute(sa.select(cases_t).order_by(cases_t.c.position, cases_t.c.id)).mappings():
                result["engines"][c["engine_id"]]["cases"][c["id"]] = {
                    "title": c["title"], "catalogue": c["catalogue"], "status": c["status"],
                    "refs": json.loads(c["refs"]), "changes": {}}
            kind_order = list(result["input_kinds"])
            prompts = sorted(conn.execute(sa.select(case_prompts_t)).mappings(),
                             key=lambda p: kind_order.index(p["input_kind_id"]))
            for p in prompts:
                result["engines"][p["engine_id"]]["cases"][p["case_id"]]["changes"][p["input_kind_id"]] = p["change_text"]
        return result

    # ---------- projects ----------
    def projects(self, hidden: bool = True) -> list[dict]:
        """Every project with its engines and job count; hidden=False leaves out the hidden ones (Default)."""
        with self.db.connect() as conn:
            rows = conn.execute(sa.select(projects_t).order_by(projects_t.c.created)).mappings().all()
            grants = conn.execute(sa.select(project_engines_t)).mappings().all()
            counts = dict(conn.execute(sa.select(jobs_t.c.project_id, sa.func.count()).group_by(jobs_t.c.project_id)).all())
        order = {engine_id: i for i, engine_id in enumerate(self.catalogue()["engines"])}
        return [dict(row) | {"hidden": bool(row["hidden"]),
                             "engines": sorted((g["engine_id"] for g in grants if g["project_id"] == row["id"]),
                                               key=lambda e: order.get(e, len(order))),
                             "jobs": counts.get(row["id"], 0)} for row in rows if hidden or not row["hidden"]]

    def project(self, project_id: int) -> dict:
        found = next((p for p in self.projects() if p["id"] == project_id), None)
        if found is None:
            raise NotFound(f"no project {project_id}")
        return found

    def default_project_id(self) -> int:
        with self.db.connect() as conn:
            row = conn.execute(sa.select(projects_t.c.id).where(projects_t.c.name == DEFAULT_PROJECT)).first()
        if row:
            return row[0]
        return self.create_project({"name": DEFAULT_PROJECT, "description": "Jobs made before projects existed."})["id"]

    def engine_name(self, engine_id: str) -> str:
        """An engine's name from the cached catalogue, without copying it; the id when it is unknown."""
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            snapshot = self.catalogue()
        return snapshot["engines"].get(engine_id, {}).get("name", engine_id)

    def _engine_ids(self, values) -> list[str]:
        known = self.catalogue()["engines"]
        chosen = list(dict.fromkeys(str(v) for v in values or []))
        unknown = [v for v in chosen if v not in known]
        if unknown:
            raise StoreError(f"unknown engine {', '.join(unknown)}")
        return chosen

    def create_project(self, data: dict) -> dict:
        name = text(data.get("name"), "name", 120)
        description = text(data.get("description"), "description", 2000, required=False)
        engines = self._engine_ids(data.get("engines"))
        with self._tx() as conn:
            if conn.execute(sa.select(projects_t.c.id).where(projects_t.c.name == name)).first():
                raise Conflict(f"a project named {name!r} exists")
            project_id = conn.execute(projects_t.insert().values(name=name, description=description,
                                                                 created=time.time())).inserted_primary_key[0]
            for engine_id in engines:
                conn.execute(project_engines_t.insert().values(project_id=project_id, engine_id=engine_id))
        return self.project(project_id)

    def update_project(self, project_id: int, data: dict) -> dict:
        self.project(project_id)
        values = {}
        if "name" in data:
            values["name"] = text(data["name"], "name", 120)
        if "description" in data:
            values["description"] = text(data["description"], "description", 2000, required=False)
        if "hidden" in data:
            values["hidden"] = bool(data["hidden"])
        engines = self._engine_ids(data["engines"]) if "engines" in data else None
        with self._tx() as conn:
            if "name" in values and conn.execute(sa.select(projects_t.c.id).where(
                    projects_t.c.name == values["name"], projects_t.c.id != project_id)).first():
                raise Conflict(f"a project named {values['name']!r} exists")
            if values:
                conn.execute(projects_t.update().where(projects_t.c.id == project_id).values(**values))
            if engines is not None:
                conn.execute(project_engines_t.delete().where(project_engines_t.c.project_id == project_id))
                for engine_id in engines:
                    conn.execute(project_engines_t.insert().values(project_id=project_id, engine_id=engine_id))
        return self.project(project_id)

    def delete_project(self, project_id: int) -> None:
        project = self.project(project_id)
        if project["jobs"]:
            raise Conflict(f"project {project['name']!r} has {project['jobs']} job(s); it can only be deleted empty")
        with self._tx() as conn:
            conn.execute(project_engines_t.delete().where(project_engines_t.c.project_id == project_id))
            conn.execute(projects_t.delete().where(projects_t.c.id == project_id))

    # ---------- jobs ----------
    def job_projects(self) -> dict[str, int]:
        with self.db.connect() as conn:
            return dict(conn.execute(sa.select(jobs_t.c.id, jobs_t.c.project_id)).all())

    def engines_in_use(self) -> set[str]:
        with self.db.connect() as conn:
            return {row[0] for row in conn.execute(sa.select(jobs_t.c.engine_id).distinct())}

    # ---------- engines ----------
    def _engine_values(self, data: dict, partial: bool) -> dict:
        values = {}
        if not partial or "name" in data:
            values["name"] = text(data.get("name"), "name", 120)
        if not partial or "color" in data:
            color = str(data.get("color") or "#5b6b80").strip()
            if not COLOR.match(color):
                raise StoreError("color must be like #1f62b5")
            values["color"] = color
        for field in PROMPT_FIELDS:
            if not partial or field in data:
                values[field] = text(data.get(field, WAREHOUSE[field] if not partial else ""), field, 20000,
                                     required=field != "scene_prefix")
        return values

    def create_engine(self, data: dict) -> None:
        engine_id = slug(data.get("id"), "engine id")
        values = self._engine_values(data, partial=False)
        with self._tx() as conn:
            if conn.execute(sa.select(engines_t.c.id).where(engines_t.c.id == engine_id)).first():
                raise Conflict(f"an engine {engine_id!r} exists")
            position = conn.execute(sa.select(sa.func.coalesce(sa.func.max(engines_t.c.position), -1))).scalar() + 1
            conn.execute(engines_t.insert().values(id=engine_id, position=position, **values))
            for project_id in self._engine_ids_to_projects(data.get("projects"), conn):
                conn.execute(project_engines_t.insert().values(project_id=project_id, engine_id=engine_id))
        self._changed()

    def _engine_ids_to_projects(self, values, conn) -> list[int]:
        try:
            chosen = list(dict.fromkeys(int(v) for v in values or []))
        except (TypeError, ValueError):
            raise StoreError("projects must be project ids")
        known = {row[0] for row in conn.execute(sa.select(projects_t.c.id))}
        unknown = [v for v in chosen if v not in known]
        if unknown:
            raise StoreError(f"unknown project {', '.join(map(str, unknown))}")
        return chosen

    def update_engine(self, engine_id: str, data: dict) -> None:
        self._need_engine(engine_id)
        values = self._engine_values(data, partial=True)
        if values:
            with self._tx() as conn:
                conn.execute(engines_t.update().where(engines_t.c.id == engine_id).values(**values))
        self._changed()

    def delete_engine(self, engine_id: str) -> None:
        self._need_engine(engine_id)
        if engine_id in self.engines_in_use():
            raise Conflict(f"engine {engine_id!r} has jobs; it cannot be deleted")
        with self._tx() as conn:
            kinds = [row[0] for row in conn.execute(sa.select(input_kinds_t.c.id).where(input_kinds_t.c.engine_id == engine_id))]
            conn.execute(case_prompts_t.delete().where(case_prompts_t.c.engine_id == engine_id))
            if kinds:
                conn.execute(case_prompts_t.delete().where(case_prompts_t.c.input_kind_id.in_(kinds)))
            conn.execute(cases_t.delete().where(cases_t.c.engine_id == engine_id))
            conn.execute(input_kinds_t.delete().where(input_kinds_t.c.engine_id == engine_id))
            conn.execute(project_engines_t.delete().where(project_engines_t.c.engine_id == engine_id))
            conn.execute(engines_t.delete().where(engines_t.c.id == engine_id))
        self._changed()

    def _need_engine(self, engine_id: str) -> dict:
        engine = self.catalogue()["engines"].get(engine_id)
        if engine is None:
            raise NotFound(f"no engine {engine_id!r}")
        return engine

    # ---------- input kinds ----------
    def create_input_kind(self, engine_id: str, data: dict) -> None:
        self._need_engine(engine_id)
        kind_id = slug(data.get("id"), "input kind id")
        if kind_id in self.catalogue()["input_kinds"]:
            raise Conflict(f"an input kind {kind_id!r} exists (each kind feeds one engine)")
        with self._tx() as conn:
            position = conn.execute(sa.select(sa.func.coalesce(sa.func.max(input_kinds_t.c.position), -1))).scalar() + 1
            conn.execute(input_kinds_t.insert().values(id=kind_id, engine_id=engine_id, position=position,
                                                       title=text(data.get("title"), "title", 160),
                                                       scene=text(data.get("scene"), "scene", 20000)))
        self._changed()

    def update_input_kind(self, kind_id: str, data: dict) -> None:
        if kind_id not in self.catalogue()["input_kinds"]:
            raise NotFound(f"no input kind {kind_id!r}")
        values = {}
        if "title" in data:
            values["title"] = text(data["title"], "title", 160)
        if "scene" in data:
            values["scene"] = text(data["scene"], "scene", 20000)
        if values:
            with self._tx() as conn:
                conn.execute(input_kinds_t.update().where(input_kinds_t.c.id == kind_id).values(**values))
        self._changed()

    def delete_input_kind(self, kind_id: str) -> None:
        if kind_id not in self.catalogue()["input_kinds"]:
            raise NotFound(f"no input kind {kind_id!r}")
        with self._tx() as conn:
            conn.execute(case_prompts_t.delete().where(case_prompts_t.c.input_kind_id == kind_id))
            conn.execute(input_kinds_t.delete().where(input_kinds_t.c.id == kind_id))
        self._changed()

    # ---------- cases (the engine's output types) ----------
    def _case_values(self, data: dict, partial: bool) -> dict:
        values = {}
        if not partial or "title" in data:
            values["title"] = text(data.get("title"), "title", 160)
        if not partial or "catalogue" in data:
            values["catalogue"] = text(data.get("catalogue") or "-", "catalogue", 32)
        if not partial or "status" in data:
            status = str(data.get("status") or "draft")
            if status not in STATUSES:
                raise StoreError(f"status must be one of {', '.join(STATUSES)}")
            values["status"] = status
        if not partial or "refs" in data:
            refs = list(dict.fromkeys(str(r) for r in data.get("refs") or []))
            known = list(self.catalogue()["references"])
            unknown = [r for r in refs if r not in known]
            if unknown:
                raise StoreError(f"unknown reference {', '.join(unknown)}; known: {', '.join(known)}")
            values["refs"] = json.dumps(refs)
        return values

    def create_case(self, engine_id: str, data: dict) -> None:
        engine = self._need_engine(engine_id)
        case_id = slug(data.get("id"), "case id")
        if case_id in engine["cases"]:
            raise Conflict(f"engine {engine_id!r} has a case {case_id!r}")
        values = self._case_values(data, partial=False)
        with self._tx() as conn:
            conn.execute(cases_t.insert().values(engine_id=engine_id, id=case_id, position=len(engine["cases"]),
                                                 **values))
        self._changed()

    def update_case(self, engine_id: str, case_id: str, data: dict) -> None:
        if case_id not in self._need_engine(engine_id)["cases"]:
            raise NotFound(f"engine {engine_id!r} has no case {case_id!r}")
        values = self._case_values(data, partial=True)
        if "refs" in values:                  # a reference its prompts still name stays given
            refs = json.loads(values["refs"])
            for kind_id, change in self.catalogue()["engines"][engine_id]["cases"][case_id]["changes"].items():
                check_named(change, refs, f"its prompt for {kind_id}")
        if values:
            with self._tx() as conn:
                conn.execute(cases_t.update().where(cases_t.c.engine_id == engine_id, cases_t.c.id == case_id)
                             .values(**values))
        self._changed()

    def delete_case(self, engine_id: str, case_id: str) -> None:
        if case_id not in self._need_engine(engine_id)["cases"]:
            raise NotFound(f"engine {engine_id!r} has no case {case_id!r}")
        with self._tx() as conn:
            conn.execute(case_prompts_t.delete().where(case_prompts_t.c.engine_id == engine_id,
                                                       case_prompts_t.c.case_id == case_id))
            conn.execute(cases_t.delete().where(cases_t.c.engine_id == engine_id, cases_t.c.id == case_id))
        self._changed()

    def set_case_prompt(self, engine_id: str, case_id: str, kind_id: str, change: str) -> None:
        """The case's ONE change for frames of kind_id; an empty text removes it (the case no longer takes them)."""
        catalogue = self.catalogue()
        if case_id not in self._need_engine(engine_id)["cases"]:
            raise NotFound(f"engine {engine_id!r} has no case {case_id!r}")
        kind = catalogue["input_kinds"].get(kind_id)
        if kind is None or kind["engine"] != engine_id:
            raise StoreError(f"{kind_id!r} is not an input kind of {engine_id!r}")
        change = text(change, "change", 20000, required=False)
        check_named(change, self.catalogue()["engines"][engine_id]["cases"][case_id]["refs"], "the prompt")
        key = (case_prompts_t.c.engine_id == engine_id, case_prompts_t.c.case_id == case_id,
               case_prompts_t.c.input_kind_id == kind_id)
        with self._tx() as conn:
            conn.execute(case_prompts_t.delete().where(*key))
            if change:
                conn.execute(case_prompts_t.insert().values(engine_id=engine_id, case_id=case_id,
                                                            input_kind_id=kind_id, change_text=change))
        self._changed()


_current: Store | None = None


def use(store: Store | None) -> None:
    global _current
    _current = store


def current() -> Store:
    if _current is None:
        raise RuntimeError("the database is not open yet")
    return _current
