"""Routes of the management pages: projects (and the engines each may use), the engine catalogue (engines, their
input kinds and cases, the outputs they make) and every prompt text. Bodies are JSON; errors are 400 (invalid), 404
(unknown) or 409 (in use / exists), mapped from service.store's errors by the app."""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

from service import reference_store as refs_db
from service import store as db
from synth.try_astra_edit_batch import compose_prompt, with_labels
from synth.violation_cases import fill

router = APIRouter(prefix="/api")
JsonBody = Body(default_factory=dict)


def as_dict(body) -> dict:
    if not isinstance(body, dict):
        raise HTTPException(400, "the body must be a JSON object")
    return body


# ---------- projects ----------
@router.get("/projects")
def list_projects(all: bool = False) -> list[dict]:
    """The projects the pages show; all=1 includes the hidden ones (Default, the jobs made before projects)."""
    return db.current().projects(hidden=all)


@router.post("/projects")
def create_project(body: dict = JsonBody) -> dict:
    return db.current().create_project(as_dict(body))


@router.get("/projects/{project_id}")
def get_project(project_id: int) -> dict:
    return db.current().project(project_id)


@router.patch("/projects/{project_id}")
def update_project(project_id: int, body: dict = JsonBody) -> dict:
    """Any of name, description, engines (the full list of engine ids the project may use)."""
    return db.current().update_project(project_id, as_dict(body))


@router.delete("/projects/{project_id}")
def delete_project(project_id: int) -> dict:
    db.current().delete_project(project_id)
    return {"deleted": project_id}


# ---------- catalogue ----------
@router.get("/catalogue")
def full_catalogue() -> dict:
    """Every engine with its prompt context, input kinds and cases with their change prompts, and the references."""
    catalogue = db.current().catalogue()
    return catalogue | {"references": refs_db.listing(catalogue), "statuses": list(db.STATUSES),
                                       "prompt_fields": list(db.PROMPT_FIELDS)}


@router.post("/engines")
def create_engine(body: dict = JsonBody) -> dict:
    """id, name, color, setting, scene_prefix, objects, keep (the warehouse text when left out), projects (ids that
    may use it)."""
    body = as_dict(body)
    db.current().create_engine(body)
    return full_catalogue()


@router.patch("/engines/{engine_id}")
def update_engine(engine_id: str, body: dict = JsonBody) -> dict:
    db.current().update_engine(engine_id, as_dict(body))
    return full_catalogue()


@router.delete("/engines/{engine_id}")
def delete_engine(engine_id: str) -> dict:
    db.current().delete_engine(engine_id)
    return full_catalogue()


@router.post("/engines/{engine_id}/input-kinds")
def create_input_kind(engine_id: str, body: dict = JsonBody) -> dict:
    db.current().create_input_kind(engine_id, as_dict(body))
    return full_catalogue()


@router.patch("/input-kinds/{kind_id}")
def update_input_kind(kind_id: str, body: dict = JsonBody) -> dict:
    db.current().update_input_kind(kind_id, as_dict(body))
    return full_catalogue()


@router.delete("/input-kinds/{kind_id}")
def delete_input_kind(kind_id: str) -> dict:
    db.current().delete_input_kind(kind_id)
    return full_catalogue()


@router.post("/engines/{engine_id}/cases")
def create_case(engine_id: str, body: dict = JsonBody) -> dict:
    db.current().create_case(engine_id, as_dict(body))
    return full_catalogue()


@router.patch("/engines/{engine_id}/cases/{case_id}")
def update_case(engine_id: str, case_id: str, body: dict = JsonBody) -> dict:
    db.current().update_case(engine_id, case_id, as_dict(body))
    return full_catalogue()


@router.delete("/engines/{engine_id}/cases/{case_id}")
def delete_case(engine_id: str, case_id: str) -> dict:
    db.current().delete_case(engine_id, case_id)
    return full_catalogue()


@router.put("/engines/{engine_id}/cases/{case_id}/prompts/{kind_id}")
def set_case_prompt(engine_id: str, case_id: str, kind_id: str, body: dict = JsonBody) -> dict:
    """change: the case's ONE change for frames of that kind; empty removes it."""
    db.current().set_case_prompt(engine_id, case_id, kind_id, as_dict(body).get("change", ""))
    return full_catalogue()


@router.get("/prompt-preview")
def prompt_preview(engine: str, case: str, kind: str) -> dict:
    """The whole prompt one output of this case would get for a frame of this kind: its references numbered as
    they are now, the skid described as the catalogue's usual skid, and {lid_pose} filled as for seed 0."""
    catalogue = db.current().catalogue()
    found = catalogue["engines"].get(engine, {}).get("cases", {}).get(case)
    if found is None or kind not in found["changes"]:
        raise HTTPException(404, f"case {case!r} of {engine!r} has no prompt for {kind!r}")
    references, labels = refs_db.for_case(catalogue, found["refs"])
    change, _ = fill(with_labels(found["changes"][kind], labels), 0)
    listed = [(r["id"], r["text"], len(r["images"]) or 1) for r in references]
    return {"prompt": compose_prompt(change, catalogue["input_kinds"][kind]["scene"], listed,
                                     catalogue["engines"][engine]["prompt"])}


# ---------- object references ----------
@router.get("/references")
def list_references() -> list[dict]:
    return refs_db.listing(db.current().catalogue())


@router.post("/references")
def create_reference(body: dict = JsonBody) -> list[dict]:
    """id (A-Z, 0-9, _), title, text (how the prompt describes its images; may hold {skid})."""
    refs_db.create(db.current(), as_dict(body))
    return list_references()


@router.patch("/references/{ref}")
def update_reference(ref: str, body: dict = JsonBody) -> list[dict]:
    refs_db.update(db.current(), ref, as_dict(body))
    return list_references()


@router.delete("/references/{ref}")
def delete_reference(ref: str) -> list[dict]:
    refs_db.delete(db.current(), ref)
    return list_references()


@router.post("/references/{ref}/images")
async def add_reference_image(ref: str, request: Request) -> list[dict]:
    """Form field image: one JPEG, PNG or WebP file, the reference's one image (it replaces the one it had)."""
    upload = (await request.form()).get("image")
    if not isinstance(upload, UploadFile) or not upload.filename:
        raise HTTPException(400, "choose an image file")
    refs_db.set_image(db.current(), ref, await upload.read())
    return list_references()


@router.delete("/references/{ref}/images/{name}")
def remove_reference_image(ref: str, name: str) -> list[dict]:
    refs_db.remove_image(db.current(), ref, name)
    return list_references()


@router.get("/references/{ref}/images/{name}")
def reference_image(ref: str, name: str) -> FileResponse:
    return FileResponse(refs_db.image_path(ref, name))
