"""Projects, the engines each may use, and the engine catalogue with its prompts, kept in the database."""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

from service import app as service
from service.codex_status import CodexUsage
from synth import violation_cases
from synth.try_astra_edit_batch import with_labels

PUSH, SKIDS, FLOOR = "forklift-pushing-multiple-lsps", "forklift-charging-multiple-skids-horizontally", "floor-opening"


class CatalogueApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.inputs = self.root / "frames"
        for kind, names in {"forklift-with-lsp-cargo": ["a.jpg", "b.jpg"], "floor-closed": ["h.jpg"]}.items():
            (self.inputs / kind).mkdir(parents=True)
            for name in names:
                Image.new("RGB", (64, 36), "gray").save(self.inputs / kind / name)
        self.calls = []
        patches = [mock.patch.dict(os.environ, {"DATABASE_URL": "sqlite://"}),
                   mock.patch.object(service, "JOBS_DIR", self.root / "jobs"),
                   # object references: their files in the test folder, seeded from small stand-in sheets
                   mock.patch.object(service.reference_store, "REFERENCES_DIR", self.root / "references"),
                   mock.patch.object(service.reference_store, "LEGACY_DIR", self.root / "legacy-references"),
                   mock.patch.object(service.reference_store, "catalogue_sheet", return_value=None),   # SKID renders
                   mock.patch.dict(service.reference_store.SEED_IMAGES, self.seed_images(), clear=True),
                   mock.patch.object(service, "load_token_history"),
                   mock.patch.object(service, "_token_samples", []),
                   mock.patch.object(service, "edit_frame", side_effect=self.fake_edit),
                   mock.patch.dict(service.jobs, clear=True),
                   mock.patch.object(service, "USAGE", CodexUsage(self.root / "usage.json", service.service_totals,
                                                                    reader=self.fake_limits))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.folder.cleanup)
        self.client = self.enterContext(TestClient(service.app))
        self.client.patch("/api/projects/1", json={"hidden": False})     # Default is hidden; these tests use it

    def seed_images(self) -> dict:
        sheets = {}
        for ref in ("LSP", "FLOOR_OPENING"):
            sheets[ref] = self.root / f"seed-{ref}.png"
            Image.new("RGB", (32, 18), "blue").save(sheets[ref])
        return sheets

    def fake_limits(self) -> dict:
        return {"plan": "prolite", "ordinary_usage_allowed": True, "limit_reached": False, "used_percent": 10,
                "window_minutes": 10080, "resets_at": 1_800_000_000, "secondary_used_percent": None,
                "secondary_resets_at": None, "read_at": time.time()}

    def fake_edit(self, image, run, output, seed, change, scene, refs=None, context=None, references=None):
        self.calls.append({"image": image.name, "change": change, "scene": scene, "refs": refs, "context": context,
                           "references": references})
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 36), "red").save(output)
        return {"output": str(output), "seconds": 1, "tokens": 1_000}

    def job(self, project_id, engine=PUSH, cases=("push_2_lsp_cargo",)):
        return self.client.post("/api/jobs", data={"project_id": str(project_id), "engine": engine,
                                                   "cases": list(cases), "folder": str(self.inputs)})

    def wait_finished(self, job_id: str) -> dict:
        with self.client.websocket_connect(f"/ws/jobs/{job_id}") as socket:
            while True:
                message = socket.receive_json()
                if message["type"] == "job" and message["job"]["finished"]:
                    return message["job"]

    def test_the_catalogue_is_seeded_from_the_code_with_a_default_project_of_every_engine(self) -> None:
        catalogue = self.client.get("/api/catalogue").json()
        self.assertEqual(list(catalogue["engines"]), list(violation_cases.ENGINES))
        self.assertEqual(list(catalogue["input_kinds"]), list(violation_cases.INPUT_KINDS))
        case = catalogue["engines"][PUSH]["cases"]["push_3_lsp_cargo"]
        labels = {"LSP": "image 2", "SKID": "image 3"}          # stored with {ID} names; numbered they are the code's
        self.assertEqual({k: with_labels(t, labels) for k, t in case["changes"].items()},
                         violation_cases.ENGINES[PUSH]["cases"]["push_3_lsp_cargo"]["changes"])
        self.assertEqual(catalogue["engines"][FLOOR]["prompt"]["setting"], violation_cases.FLOOR_SETTING)
        self.assertIn("warehouse", catalogue["engines"][PUSH]["prompt"]["setting"])
        self.assertEqual({r["id"] for r in catalogue["references"]}, {"LSP", "SKID", "FLOOR_OPENING", "CARGO"})
        [default] = self.client.get("/api/projects").json()
        self.assertEqual(default["name"], "Default")
        self.assertEqual(default["engines"], list(violation_cases.ENGINES))

    def test_a_project_may_only_use_the_engines_it_was_given(self) -> None:
        made = self.client.post("/api/projects", json={"name": "Hatch audit", "description": "floor only",
                                                       "engines": [FLOOR]})
        self.assertEqual(made.status_code, 200, made.text)
        project = made.json()
        self.assertEqual(project["engines"], [FLOOR])
        shown = self.client.get(f"/api/engines?project_id={project['id']}").json()
        self.assertEqual([e["id"] for e in shown["engines"]], [FLOOR])
        self.assertEqual([k["id"] for k in shown["input_kinds"]], ["floor-closed"])
        self.assertEqual(self.job(project["id"]).status_code, 403)          # PUSH not given to it
        granted = self.client.patch(f"/api/projects/{project['id']}", json={"engines": [FLOOR, PUSH]}).json()
        self.assertEqual(granted["engines"], [PUSH, FLOOR])                  # in catalogue order
        job = self.job(project["id"]).json()
        self.assertEqual(job["project_id"], project["id"])
        self.assertEqual(job["total"], 2)
        self.wait_finished(job["id"])
        self.assertEqual([j["id"] for j in self.client.get(f"/api/jobs?project_id={project['id']}").json()], [job["id"]])
        self.assertEqual(self.client.get("/api/jobs?project_id=1").json(), [])
        self.assertEqual(self.client.delete(f"/api/projects/{project['id']}").status_code, 409)   # it has a job
        self.assertEqual(self.client.delete(f"/api/engines/{PUSH}").status_code, 409)             # in use

    def test_the_default_project_is_hidden_from_the_pages_and_takes_no_new_job(self) -> None:
        with mock.patch.dict(os.environ, {"DATABASE_URL": "sqlite://"}):
            fresh = service.db.Store("sqlite://")
            fresh.setup()
        self.assertEqual(fresh.projects(hidden=False), [])
        self.assertTrue(fresh.projects()[0]["hidden"])
        self.client.patch("/api/projects/1", json={"hidden": True})
        self.assertEqual(self.client.get("/api/projects").json(), [])
        self.assertEqual([p["name"] for p in self.client.get("/api/projects?all=1").json()], ["Default"])
        refused = self.job(1)
        self.assertEqual(refused.status_code, 403)
        self.assertIn("hidden", refused.text)

    def test_bad_project_requests_are_refused(self) -> None:
        self.assertEqual(self.client.post("/api/projects", json={"name": " "}).status_code, 400)
        self.assertEqual(self.client.post("/api/projects", json={"name": "Default"}).status_code, 409)
        self.assertEqual(self.client.post("/api/projects", json={"name": "x", "engines": ["nope"]}).status_code, 400)
        self.assertEqual(self.client.patch("/api/projects/99", json={"name": "y"}).status_code, 404)
        self.assertEqual(self.job(99).status_code, 404)
        empty = self.client.post("/api/projects", json={"name": "Empty"}).json()
        self.assertEqual(self.client.delete(f"/api/projects/{empty['id']}").status_code, 200)

    def test_an_engine_with_its_kinds_cases_and_prompts_is_made_on_the_engines_page(self) -> None:
        made = self.client.post("/api/engines", json={"id": "helmet", "name": "No Helmet", "color": "#1f62b5",
                                                      "setting": "a real CCTV camera on a building site",
                                                      "projects": [1]})
        self.assertEqual(made.status_code, 200, made.text)
        self.assertIn("helmet", self.client.get("/api/projects/1").json()["engines"])
        self.assertEqual(self.client.post("/api/engines/helmet/input-kinds", json={
            "id": "worker-with-helmet", "title": "Worker with helmet", "scene": "one worker wearing a helmet"}
        ).status_code, 200)
        self.assertEqual(self.client.post("/api/engines/helmet/cases", json={
            "id": "remove_helmet", "title": "Helmet removed", "refs": []}).status_code, 200)
        put = self.client.put("/api/engines/helmet/cases/remove_helmet/prompts/worker-with-helmet",
                              json={"change": "the worker no longer wears a helmet."})
        catalogue = put.json()
        self.assertEqual(catalogue["engines"]["helmet"]["cases"]["remove_helmet"]["changes"],
                         {"worker-with-helmet": "the worker no longer wears a helmet."})
        preview = self.client.get("/api/prompt-preview", params={"engine": "helmet", "case": "remove_helmet",
                                                                 "kind": "worker-with-helmet"}).json()["prompt"]
        self.assertIn("building site", preview)
        self.assertIn("one worker wearing a helmet", preview)
        self.assertIn("Make this ONE change: the worker no longer wears a helmet.", preview)

        (self.inputs / "worker-with-helmet").mkdir()
        Image.new("RGB", (64, 36)).save(self.inputs / "worker-with-helmet" / "w.jpg")
        job = self.job(1, "helmet", ["remove_helmet"]).json()
        self.assertEqual(job["engine_name"], "No Helmet")
        self.wait_finished(job["id"])
        [call] = self.calls
        self.assertEqual((call["image"], call["change"], call["scene"], call["refs"]),
                         ("w.jpg", "the worker no longer wears a helmet.", "one worker wearing a helmet", []))
        self.assertEqual(call["context"]["setting"], "a real CCTV camera on a building site")

    def test_an_edited_prompt_is_used_by_the_next_output(self) -> None:
        self.client.patch(f"/api/engines/{PUSH}", json={"keep": "Keep everything."})
        self.client.patch("/api/input-kinds/forklift-with-lsp-cargo", json={"scene": "a forklift, edited"})
        self.client.put(f"/api/engines/{PUSH}/cases/push_2_lsp_cargo/prompts/forklift-with-lsp-cargo",
                        json={"change": "an edited change."})
        self.wait_finished(self.job(1).json()["id"])
        self.assertEqual({c["change"] for c in self.calls}, {"an edited change."})
        self.assertEqual({c["scene"] for c in self.calls}, {"a forklift, edited"})
        self.assertEqual({c["context"]["keep"] for c in self.calls}, {"Keep everything."})

    def test_an_emptied_prompt_means_the_case_no_longer_takes_that_kind(self) -> None:
        self.client.put(f"/api/engines/{PUSH}/cases/push_2_lsp_cargo/prompts/forklift-with-lsp-cargo",
                        json={"change": ""})
        refused = self.job(1)
        self.assertEqual(refused.status_code, 400)
        self.assertIn("none of the chosen cases", refused.text)

    def test_bad_catalogue_requests_are_refused(self) -> None:
        bad = [("post", "/api/engines", {"id": "Bad Id", "name": "x"}),
               ("post", f"/api/engines/{PUSH}/cases", {"id": "a__b", "title": "t"}),     # "__" splits output names
               ("post", "/api/engines", {"id": PUSH, "name": "again"}),
               ("post", "/api/engines", {"id": "ok", "name": "x", "color": "red"}),
               ("post", f"/api/engines/{PUSH}/cases", {"id": "c", "title": "t", "refs": ["NOPE"]}),
               ("post", f"/api/engines/{PUSH}/cases", {"id": "c", "title": "t", "status": "odd"}),
               ("post", f"/api/engines/{PUSH}/input-kinds", {"id": "floor-closed", "title": "t", "scene": "s"}),
               ("put", f"/api/engines/{PUSH}/cases/push_2_lsp_cargo/prompts/floor-closed", {"change": "x"}),
               ("patch", "/api/engines/nope", {"name": "x"})]
        for method, url, body in bad:
            with self.subTest(url=url, body=body):
                self.assertIn(getattr(self.client, method)(url, json=body).status_code, (400, 404, 409))
        self.assertEqual(self.client.get("/api/prompt-preview", params={
            "engine": PUSH, "case": "push_2_lsp_cargo", "kind": "floor-closed"}).status_code, 404)

    def test_a_job_on_disk_unknown_to_the_database_joins_the_default_project(self) -> None:
        folder = self.root / "jobs" / "20260101-000000-abcdef"
        (folder / "inputs" / "forklift-with-lsp-cargo").mkdir(parents=True)
        (folder / "job.json").write_text(json.dumps({
            "id": folder.name, "created": 1.0, "engine": PUSH, "cases": ["push_2_lsp_cargo"], "variants": 2,
            "rates": {"push_2_lsp_cargo": 1.0}, "inputs": {"forklift-with-lsp-cargo": []}, "tasks": []}))
        service.load_jobs(service.db.current())
        self.assertEqual(service.jobs[folder.name].project_id, 1)
        self.assertEqual(self.client.get("/api/projects/1").json()["jobs"], 1)
        stored = json.loads((folder / "job.json").read_text())
        self.assertEqual(stored["project_id"], 1)
        self.assertNotIn("variants", stored)

    def test_the_database_holds_a_jobs_full_state_and_a_restart_loads_it_from_there(self) -> None:
        job = self.job(1).json()
        final = self.wait_finished(job["id"])
        task = final["tasks"][0]
        self.client.post(f"/api/jobs/{job['id']}/tasks/{task['id']}/review", data={"verdict": "good"})
        self.client.post(f"/api/jobs/{job['id']}/inputs/forklift-with-lsp-cargo/b.jpg/exclude", data={"excluded": "1"})
        self.client.post(f"/api/jobs/{job['id']}/delete", data={"task": [final["tasks"][1]["id"]]})
        (self.root / "jobs" / job["id"] / "job.json").unlink()      # the file copy is gone: the database is enough
        service.jobs.clear()
        service.load_jobs(service.db.current())
        state = self.client.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual((state["project_id"], state["engine"], state["cases"]), (1, PUSH, ["push_2_lsp_cargo"]))
        self.assertEqual([i["name"] for i in state["inputs"]], ["a.jpg", "b.jpg"])
        self.assertEqual(state["tasks"][0]["review"], "good")
        self.assertEqual(state["tasks"][0]["token_attempts"], final["tasks"][0]["token_attempts"])
        self.assertEqual(state["excluded_inputs"], ["forklift-with-lsp-cargo/b.jpg"])
        self.assertEqual((state["total"], state["deleted"]), (1, 1))
        self.assertEqual(state["tokens"]["used"], final["tokens"]["used"])
        rows = service.job_records.job_states(service.db.current().db)[job["id"]]
        self.assertEqual(len(rows["tasks"]) + len(rows["deleted_tasks"]), 2)

    def png(self, color: str = "green") -> bytes:
        data = io.BytesIO()
        Image.new("RGB", (40, 20), color).save(data, format="PNG")
        return data.getvalue()

    def test_object_references_are_seeded_from_the_code_and_prompts_name_them(self) -> None:
        refs = {r["id"]: r for r in self.client.get("/api/references").json()}
        self.assertEqual(list(refs), ["LSP", "SKID", "FLOOR_OPENING", "CARGO"])
        self.assertEqual(len(refs["LSP"]["images"]), 1)                    # the code's sheet, copied in
        self.assertEqual(refs["SKID"]["images"], [])
        self.assertTrue(refs["SKID"]["render"])                             # rendered per frame instead
        self.assertEqual(self.client.get(refs["LSP"]["images"][0]["url"]).status_code, 200)
        change = self.client.get("/api/catalogue").json()["engines"][PUSH]["cases"]["push_2_lsp_cargo"]["changes"]
        self.assertIn("one skid (like {SKID})", change["forklift-with-lsp-cargo"])
        self.assertIn("use {LSP} only if image 1 showed no LSP", change["forklift-with-lsp-cargo"])
        self.assertNotIn("image 3", change["forklift-with-lsp-cargo"])
        self.assertEqual({u["case"] for u in refs["SKID"]["used_by"]},
                         {"push_2_lsp_cargo", "push_3_lsp_cargo", "push_2_lsp_empty_extra",
                          "carry_2_skids_side_by_side", "carry_2_cargo_stacked"})

    def test_an_object_has_one_reference_sheet_and_a_new_upload_replaces_it(self) -> None:
        old = self.client.get("/api/references").json()[0]["images"][0]
        answer = self.client.post("/api/references/LSP/images", files={"image": ("b.png", self.png(), "image/png")})
        self.assertEqual(answer.status_code, 200, answer.text)
        lsp = answer.json()[0]
        self.assertEqual(len(lsp["images"]), 1)
        self.assertNotEqual(lsp["images"][0]["name"], old["name"])
        self.assertFalse((self.root / "references" / "LSP" / old["name"]).exists())       # out of use
        kept = list((self.root / "references" / "LSP" / "previous").iterdir())
        self.assertEqual([k.name.split("-", 2)[-1] for k in kept], [old["name"]])          # but kept
        self.assertEqual(self.client.get(old["url"]).status_code, 404)
        self.assertEqual(self.client.post("/api/references/SKID/images",
                                          files={"image": ("x.txt", b"not an image", "text/plain")}).status_code, 400)
        skid = self.client.post("/api/references/SKID/images", files={"image": ("s.png", self.png(), "image/png")})
        self.assertEqual(len(skid.json()[1]["images"]), 1)                                  # replaces the render
        self.assertFalse(skid.json()[1]["render"])                                          # a sheet like LSP now
        preview = self.client.get("/api/prompt-preview", params={"engine": PUSH, "case": "push_2_lsp_cargo",
                                                                 "kind": "forklift-with-lsp-cargo"}).json()["prompt"]
        self.assertIn("You are given 3 reference images", preview)
        self.assertIn("use image 2 only if image 1 showed no LSP", preview)
        self.assertIn("one skid (like image 3)", preview)
        self.wait_finished(self.job(1).json()["id"])
        lsp_ref, skid_ref = self.calls[0]["references"]
        self.assertEqual((len(lsp_ref["images"]), len(skid_ref["images"])), (1, 1))
        self.assertIn("one skid (like image 3)", self.calls[0]["change"])

    def test_a_new_database_takes_the_skid_and_cargo_sheets_and_the_old_folder(self) -> None:
        legacy = self.root / "legacy-references" / "LSP"
        legacy.mkdir(parents=True)
        (legacy / "old.jpg").write_bytes(self.png())
        sheet = self.png("brown")
        with mock.patch.dict(os.environ, {"DATABASE_URL": "sqlite://"}), \
                mock.patch.object(service.reference_store, "catalogue_sheet", return_value=sheet):
            fresh = service.db.Store("sqlite://")
            fresh.setup()
        refs = fresh.catalogue()["references"]
        self.assertFalse(refs["SKID"]["render"])
        self.assertEqual(len(refs["SKID"]["images"]), 1)
        self.assertIn("SIX different wooden skids", refs["SKID"]["text"])
        self.assertIn("FOUR different kinds of cargo", refs["CARGO"]["text"])
        self.assertEqual(len(refs["CARGO"]["images"]), 1)
        stored = self.root / "references" / "SKID" / refs["SKID"]["images"][0]
        self.assertEqual(stored.read_bytes(), sheet)
        self.assertTrue((self.root / "references" / "LSP" / "old.jpg").is_file())     # moved to its own folder
        self.assertFalse((self.root / "legacy-references").exists())

    def test_a_prompt_only_names_the_references_its_output_gives(self) -> None:
        url = f"/api/engines/{PUSH}/cases/push_2_lsp_empty_extra"
        refused = self.client.put(f"{url}/prompts/forklift-with-lsp-cargo", json={"change": "like {FLOOR_OPENING}."})
        self.assertEqual(refused.status_code, 400)
        self.assertIn("{FLOOR_OPENING}", refused.text)
        self.assertEqual(self.client.patch(url, json={"refs": ["SKID"]}).status_code, 400)    # its prompts name {LSP}
        self.assertEqual(self.client.patch(url, json={"refs": ["LSP"]}).status_code, 200)     # SKID is not named
        case = self.client.get("/api/catalogue").json()["engines"][PUSH]["cases"]["push_2_lsp_empty_extra"]
        self.assertEqual(case["refs"], ["LSP"])

    def test_object_references_are_made_edited_and_only_deleted_when_unused(self) -> None:
        made = self.client.post("/api/references", json={"id": "pallet", "title": "Pallet", "text": "A pallet."})
        self.assertEqual(made.status_code, 200, made.text)
        self.assertIn("PALLET", [r["id"] for r in made.json()])             # ids are upper case
        name = self.client.post("/api/references/PALLET/images",
                                files={"image": ("p.png", self.png(), "image/png")}).json()[-1]["images"][0]["name"]
        self.assertEqual(self.client.get("/api/references/PALLET/images/..%2F..%2Fjob.json").status_code, 404)
        self.client.patch("/api/references/PALLET", json={"text": "A wooden pallet."})
        self.assertEqual(self.client.post(f"/api/engines/{PUSH}/cases", json={
            "id": "pallet_case", "title": "Pallet", "refs": ["PALLET"]}).status_code, 200)
        self.assertEqual(self.client.delete("/api/references/PALLET").status_code, 409)     # a case gives it
        self.client.delete(f"/api/engines/{PUSH}/cases/pallet_case")
        self.assertEqual(self.client.delete(f"/api/references/PALLET/images/{name}").status_code, 200)
        self.assertEqual(self.client.delete("/api/references/PALLET").status_code, 200)
        self.assertFalse((self.root / "references" / "PALLET").exists())
        self.assertEqual(len(list((self.root / "references" / ".deleted").glob("PALLET-*/previous/*"))), 1)
        for bad in ({"id": "LSP", "title": "x", "text": "x"}, {"id": "1X", "title": "x", "text": "x"},
                    {"id": "NEW", "title": "x", "text": " "}):
            with self.subTest(body=bad):
                self.assertIn(self.client.post("/api/references", json=bad).status_code, (400, 409))

    def test_an_output_whose_reference_has_no_image_fails_with_a_clear_error(self) -> None:
        lsp = self.client.get("/api/references").json()[0]
        self.client.delete(f"/api/references/LSP/images/{lsp['images'][0]['name']}")
        final = self.wait_finished(self.job(1).json()["id"])
        self.assertEqual(final["counts"]["failed"], 2)
        self.assertIn("LSP has no image", final["tasks"][0]["error"])
        self.assertEqual(self.calls, [])

    def test_regenerate_makes_a_new_job_of_the_same_frames_and_cases_with_the_prompts_of_now(self) -> None:
        first = self.job(1).json()
        self.wait_finished(first["id"])
        self.client.post(f"/api/jobs/{first['id']}/inputs/forklift-with-lsp-cargo/b.jpg/exclude", data={"excluded": "1"})
        self.client.put(f"/api/engines/{PUSH}/cases/push_2_lsp_cargo/prompts/forklift-with-lsp-cargo",
                        json={"change": "the prompt of now."})
        answer = self.client.post(f"/api/jobs/{first['id']}/regenerate")
        self.assertEqual(answer.status_code, 200, answer.text)
        again = answer.json()
        self.assertNotEqual(again["id"], first["id"])
        self.assertEqual((again["project_id"], again["engine"], again["cases"]), (1, PUSH, ["push_2_lsp_cargo"]))
        self.assertEqual([i["name"] for i in again["inputs"]], ["a.jpg"])          # the excluded frame is left out
        self.assertIn(f"re-generated from job {first['id']}", again["notes"])
        self.wait_finished(again["id"])
        self.assertEqual(self.calls[-1]["change"], "the prompt of now.")
        self.assertEqual(self.client.get(f"/api/jobs/{first['id']}").json()["total"], 2)   # the old job is untouched
        summary = next(j for j in self.client.get("/api/jobs").json() if j["id"] == again["id"])
        self.assertEqual(summary["review"]["outputs"], 1)
        self.client.patch("/api/projects/1", json={"hidden": True})
        self.assertEqual(self.client.post(f"/api/jobs/{first['id']}/regenerate").status_code, 403)
        shown = self.client.post("/api/projects", json={"name": "Shown", "engines": [PUSH]}).json()
        into = self.client.post(f"/api/jobs/{first['id']}/regenerate", data={"project_id": str(shown["id"])})
        self.assertEqual(into.json()["project_id"], shown["id"])
        self.wait_finished(into.json()["id"])

    def test_the_pages_and_their_menu_are_served(self) -> None:
        for page in ("/", "/jobs", "/projects", "/engines", "/prompts", "/references"):
            with self.subTest(page=page):
                response = self.client.get(page)
                self.assertEqual(response.status_code, 200)
                self.assertIn("Synthetic Data Generation", response.text)
                self.assertIn("/static/nav.js", response.text)
        self.assertEqual(self.client.get("/nope").status_code, 404)
        self.assertEqual(self.client.get("/static/nav.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
