"""Object references: the images a case gives Astra after the frame (an LSP sheet, a skid, an open floor hatch), each
with the text that tells Astra what they show. Managed on the Object references page instead of hard-coded: ONE
image per reference (user, 2026-10-09: one reference sheet showing the object from several angles), kept in its own
folder REFERENCES_DIR/<id>/ (references/ at the repository root, its file name in the database). Uploading another
replaces it in use, and the earlier one is kept in REFERENCES_DIR/<id>/previous/; a removed image or a deleted
reference is kept there too, so no uploaded sheet is ever lost.

SKID and CARGO are sheets like LSP (synth.reference_sheet: six skids / four cargo kinds of the catalogue, three
angles each). A reference with `render`
and no image (the old SKID) is the frame's own render from the 3D catalogue instead; an upload ends that. Prompt texts name references as {ID} (e.g. "one skid (like {SKID})"); each output gets them numbered from
image 2 by how many images each reference has (synth.try_astra_edit_batch.reference_labels).

On first start the references are filled from the code (synth.try_astra_edit_batch: REFERENCE_TEXT and the fixed
sheets in work/), and the "image 2" / "image 3" of every stored case prompt become the {ID} of the reference at that
place in the case's refs.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import time
import uuid
from pathlib import Path

import sqlalchemy as sa
from PIL import Image, UnidentifiedImageError

from service.store import Conflict, NotFound, Store, StoreError, case_prompts_t, cases_t, references_t, text
from synth.try_astra_edit_batch import CATALOGUE_REFS, FIXED_REFERENCES, REFERENCE_TEXT, ROOT, reference_labels

REFERENCES_DIR = ROOT / "references"
LEGACY_DIR = ROOT / "work/service/references"     # where the sheets were kept before they had their own folder
MAX_IMAGES = 1
MAX_BYTES = 20 * 1024 * 1024
FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
REF_ID = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
SEED_TITLES = {"LSP": "LSP (load spreader plate)", "SKID": "Skid", "FLOOR_OPENING": "Open floor hatch",
               "CARGO": "Cargo"}
SEED_IMAGES = dict(FIXED_REFERENCES)       # id -> the image file the code used
SKID_TEXT = ("A skid reference sheet: SIX different wooden skids of this warehouse (top deck boards on 3 runners, "
             "no bottom deck; closed or open deck), each shown from three angles side by side in its own block. Copy "
             "only how a skid of this warehouse looks (colour, material, boards and runners), taking ONE of them for "
             "each skid and never mixing them; they are renders, not to scale, so take no size or viewpoint from "
             "them, and never copy the grid.")
CARGO_TEXT = ("A cargo reference sheet: FOUR different kinds of cargo load of this warehouse (film-wrapped, "
              "strapped, a wooden crate, black net), each shown from three angles side by side in its own "
              "block. Use it only for how a load of a kind looks, taking ONE of them for each load and never mixing "
              "them, and prefer the real loads in image 1 when there are any. They are renders, sharper and cleaner "
              "than the CCTV frame: a new load takes the frame's blur, noise and lighting; they are not to scale, so "
              "take no size or viewpoint from them, and never copy the grid.")
# references built from the catalogue on a new database: id -> its text
CATALOGUE_SHEETS = {"SKID": SKID_TEXT, "CARGO": CARGO_TEXT}


# ---------- first start ----------
def move_legacy() -> None:
    """Sheets kept under work/service/references/ move to their own folder (references/), once."""
    if not LEGACY_DIR.is_dir():
        return
    for folder in LEGACY_DIR.iterdir():
        target = REFERENCES_DIR / folder.name
        target.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            if not (target / item.name).exists():
                shutil.move(str(item), str(target / item.name))
    shutil.rmtree(LEGACY_DIR, ignore_errors=True)


def catalogue_sheet(ref: str) -> bytes | None:
    """The sheet of the catalogue models of ref (SKID, CARGO), or None when the catalogue renders are not there."""
    from synth.reference_sheet import PRESETS, build_sheet
    out = REFERENCES_DIR / ref / "catalogue-sheet.jpg"
    try:
        return build_sheet(PRESETS[ref], out).read_bytes()
    except (SystemExit, OSError):
        return None
    finally:
        out.unlink(missing_ok=True)


def seed(store: Store) -> None:
    """Fill an empty references table from the code and turn the case prompts' image numbers into {ID} names."""
    move_legacy()
    with store.db.begin() as conn:
        if conn.execute(sa.select(sa.func.count()).select_from(references_t)).scalar():
            return
        for position, (ref, words) in enumerate((REFERENCE_TEXT | {"CARGO": CARGO_TEXT}).items()):
            images = []
            source = SEED_IMAGES.get(ref)
            data = Path(source).read_bytes() if source is not None and Path(source).is_file() else None
            if ref in CATALOGUE_SHEETS and data is None and (data := catalogue_sheet(ref)) is not None:
                words = CATALOGUE_SHEETS[ref]
            if data is not None:
                images.append(_store_file(ref, data))
            conn.execute(references_t.insert().values(id=ref, title=SEED_TITLES.get(ref, ref), text=words,
                                                      images=json.dumps(images),
                                                      render=ref in CATALOGUE_REFS and not images, position=position))
        refs = {(c.engine_id, c.id): json.loads(c.refs) for c in conn.execute(sa.select(cases_t))}
        for row in conn.execute(sa.select(case_prompts_t)).mappings().all():
            named = name_images(row["change_text"], refs.get((row["engine_id"], row["case_id"]), []))
            if named != row["change_text"]:
                conn.execute(case_prompts_t.update().where(
                    case_prompts_t.c.engine_id == row["engine_id"], case_prompts_t.c.case_id == row["case_id"],
                    case_prompts_t.c.input_kind_id == row["input_kind_id"]).values(change_text=named))
    store._changed()


def name_images(change: str, refs: list[str]) -> str:
    """"image 3" -> "{SKID}" when SKID is the case's second reference (image 1 is the frame, then one image each)."""
    def name(match: re.Match) -> str:
        index = int(match[1]) - 2
        return "{" + refs[index] + "}" if 0 <= index < len(refs) else match[0]
    return re.sub(r"\bimage (\d+)\b", name, change)


# ---------- files ----------
def _store_file(ref: str, data: bytes) -> str:
    """Check data is an image and keep it in REFERENCES_DIR/<ref>/ under a new name; returns the file name."""
    if len(data) > MAX_BYTES:
        raise StoreError(f"the image is over {MAX_BYTES // (1024 * 1024)} MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            kind = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise StoreError("the file is not an image")
    if kind not in FORMATS:
        raise StoreError(f"use a JPEG, PNG or WebP image, not {kind}")
    folder = REFERENCES_DIR / ref
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex[:10]}{FORMATS[kind]}"
    (folder / name).write_bytes(data)
    return name


def image_path(ref: str, name: str) -> Path:
    """The file of one of a reference's images; 404 unless the database lists it (so no other path is served)."""
    found = _need(ref)
    if name not in found["images"]:
        raise NotFound(f"reference {ref} has no image {name}")
    return REFERENCES_DIR / ref / name


# ---------- what the pages and the generator read ----------
def _need(ref: str) -> dict:
    found = current_references().get(ref)
    if found is None:
        raise NotFound(f"no reference {ref!r}")
    return found


def current_references() -> dict:
    from service import store as db
    return db.current().catalogue()["references"]


def used_by(catalogue: dict) -> dict[str, list[dict]]:
    """reference id -> the cases that give it, as {engine, engine_name, case, title}."""
    uses: dict[str, list[dict]] = {}
    for engine_id, engine in catalogue["engines"].items():
        for case_id, case in engine["cases"].items():
            for ref in case["refs"]:
                uses.setdefault(ref, []).append({"engine": engine_id, "engine_name": engine["name"],
                                                 "case": case_id, "title": case["title"]})
    return uses


def listing(catalogue: dict) -> list[dict]:
    """Every reference for the pages: its images' URLs, what it is when no image is given, and its cases."""
    uses = used_by(catalogue)
    return [{"id": ref, "title": r["title"], "text": r["text"], "render": r["render"],
             "images": [{"name": name, "url": f"/api/references/{ref}/images/{name}"} for name in r["images"]],
             "source": ("uploaded image" if r["images"]
                        else "rendered per frame from the 3D catalogue" if r["render"] else "no image yet"),
             "used_by": uses.get(ref, [])}
            for ref, r in catalogue["references"].items()]


def for_case(catalogue: dict, refs: list[str]) -> tuple[list[dict], dict[str, str]]:
    """The references of a case as edit_frame takes them, and what the prompt calls each ({ID} -> "image 3").
    StoreError names a reference that is gone or has no image."""
    resolved = []
    for ref in refs:
        found = catalogue["references"].get(ref)
        if found is None:
            raise StoreError(f"the reference {ref} no longer exists (Object references page)")
        if not found["images"] and not found["render"]:
            raise StoreError(f"the reference {ref} has no image: upload one on the Object references page")
        resolved.append({"id": ref, "text": found["text"], "render": found["render"],
                         "images": [REFERENCES_DIR / ref / name for name in found["images"]]})
    labels = reference_labels([(r["id"], len(r["images"]) or 1) for r in resolved])
    return resolved, labels


# ---------- changes ----------
def create(store: Store, data: dict) -> None:
    ref = str(data.get("id") or "").strip().upper()
    if not REF_ID.match(ref):
        raise StoreError("the id must be 1-32 characters of A-Z, 0-9 and _, starting with a letter (e.g. PALLET)")
    if ref in store.catalogue()["references"]:
        raise Conflict(f"a reference {ref} exists")
    with store._tx() as conn:
        position = conn.execute(sa.select(sa.func.coalesce(sa.func.max(references_t.c.position), -1))).scalar() + 1
        conn.execute(references_t.insert().values(id=ref, title=text(data.get("title"), "title", 160),
                                                  text=text(data.get("text"), "text", 20000),
                                                  images="[]", render=False, position=position))
    store._changed()


def update(store: Store, ref: str, data: dict) -> None:
    _need(ref)
    values = {}
    if "title" in data:
        values["title"] = text(data["title"], "title", 160)
    if "text" in data:
        values["text"] = text(data["text"], "text", 20000)
    if values:
        with store._tx() as conn:
            conn.execute(references_t.update().where(references_t.c.id == ref).values(**values))
    store._changed()


def delete(store: Store, ref: str) -> None:
    _need(ref)
    uses = used_by(store.catalogue()).get(ref)
    if uses:
        raise Conflict(f"{ref} is given by {len(uses)} case(s) ({', '.join(u['title'] for u in uses)}); "
                       "take it off them on the Engines page first")
    with store._tx() as conn:
        conn.execute(references_t.delete().where(references_t.c.id == ref))
    store._changed()
    folder = REFERENCES_DIR / ref                 # kept, out of the way: references/.deleted/<ref>-<when>/
    if folder.is_dir():
        (REFERENCES_DIR / ".deleted").mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(REFERENCES_DIR / ".deleted" / f"{ref}-{time.strftime('%Y%m%d-%H%M%S')}"))


def keep(ref: str, name: str) -> None:
    """Move an image out of use into REFERENCES_DIR/<ref>/previous/, its name prefixed with when it left."""
    source = REFERENCES_DIR / ref / name
    if source.is_file():
        folder = REFERENCES_DIR / ref / "previous"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"))


def set_image(store: Store, ref: str, data: bytes) -> None:
    """The reference's image: the new file takes the place of the one it had, which is kept in previous/. An
    upload also ends a catalogue render (the old SKID): the sheet is the reference from now on."""
    old = _need(ref)["images"]
    name = _store_file(ref, data)
    with store._tx() as conn:
        conn.execute(references_t.update().where(references_t.c.id == ref)
                     .values(images=json.dumps([name]), render=False))
    store._changed()
    for previous in old:
        keep(ref, previous)


def remove_image(store: Store, ref: str, name: str) -> None:
    path = image_path(ref, name)
    images = [n for n in _need(ref)["images"] if n != name]
    with store._tx() as conn:
        conn.execute(references_t.update().where(references_t.c.id == ref).values(images=json.dumps(images)))
    store._changed()
    keep(ref, path.name)
