"""Behavioral checks for the generation service: input kinds, jobs, the 3-thread pool, live updates, downloads."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

from service import app as service
from service.codex_status import CodexUsage
from synth import violation_cases

PUSH, SKIDS = "forklift-pushing-multiple-lsps", "forklift-charging-multiple-skids-horizontally"
FLOOR = "floor-opening"


def png(color: str = "gray") -> bytes:
    data = io.BytesIO()
    Image.new("RGB", (64, 36), color).save(data, format="PNG")
    return data.getvalue()


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.inputs = self.root / "frames"
        for kind, names in {"forklift-with-lsp-cargo": ["a.jpg", "b.png", "e.jpg", "f.jpg"],
                            "forklift-with-lsp-empty": ["c.jpg", "g.jpg"],
                            "forklift-empty": ["d.jpg"]}.items():
            (self.inputs / kind).mkdir(parents=True)
            for name in names:
                Image.new("RGB", (64, 36), "gray").save(self.inputs / kind / name)
        (self.inputs / "unsorted").mkdir()
        Image.new("RGB", (64, 36), "gray").save(self.inputs / "unsorted" / "x.jpg")
        self.running, self.peak, self.calls = 0, 0, []
        self.gate = threading.Event()        # fake edits wait on it; a test clears it to hold outputs in the queue
        self.gate.set()
        self.lock = threading.Lock()
        patches = [mock.patch.dict(os.environ, {"DATABASE_URL": "sqlite://"}),      # a fresh database per test
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
                   mock.patch.object(service, "ENFORCE_CODEX_LIMIT", True),     # the limit tests below refuse
                   mock.patch.object(service, "USAGE", CodexUsage(self.root / "usage.json", service.service_totals,
                                                                    reader=self.fake_limits))]
        self.used_percent, self.limit_reached = 40, False
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.folder.cleanup)
        # one event loop for the whole test, as under uvicorn: the pool threads post their updates to it
        self.client = self.enterContext(TestClient(service.app))
        self.client.patch("/api/projects/1", json={"hidden": False})     # Default is hidden; these tests use it

    def seed_images(self) -> dict:
        sheets = {}
        for ref in ("LSP", "FLOOR_OPENING"):
            sheets[ref] = self.root / f"seed-{ref}.png"
            Image.new("RGB", (32, 18), "blue").save(sheets[ref])
        return sheets

    def fake_limits(self) -> dict:
        return {"plan": "prolite", "ordinary_usage_allowed": not self.limit_reached, "limit_reached": self.limit_reached,
                "used_percent": self.used_percent, "window_minutes": 10080, "resets_at": 1_800_000_000,
                "secondary_used_percent": None, "secondary_resets_at": None, "read_at": time.time()}

    def fake_edit(self, image, run, output, seed, change, scene, refs=None, context=None, references=None):
        self.gate.wait(5)
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
            self.calls.append({"image": image.name, "output": output, "seed": seed, "change": change,
                               "scene": scene, "refs": refs, "references": references})
        time.sleep(0.05)
        with self.lock:
            self.running -= 1
        if image.name == "b.png":
            return {"error": "Astra produced no image", "seconds": 1, "tokens": 5_000}
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 36), "red").save(output)
        return {"output": str(output), "seconds": 1, "tokens": 20_000}

    def create(self, **fields):
        data = {"project_id": "1", "engine": PUSH, "cases": ["push_2_lsp_cargo", "push_3_lsp_cargo"],
                "folder": str(self.inputs)}
        data.update(fields)
        return self.client.post("/api/jobs", data=data)

    def restart_after(self, job_id: str, change) -> None:
        """What the database holds of a job, changed as a stopped service would have left it; then a restart."""
        store = service.db.current()
        state = service.job_records.job_states(store.db)[job_id]
        change(state["tasks"])
        service.job_records.save_job(store.db, state, [(False, task) for task in state["tasks"]])
        service.jobs.clear()
        service.load_jobs(store)

    def wait_running(self, job_id: str, count: int) -> None:
        deadline = time.time() + 5
        while self.client.get(f"/api/jobs/{job_id}").json()["counts"]["running"] < count and time.time() < deadline:
            time.sleep(0.01)

    def wait_finished(self, job_id: str) -> dict:
        """The job state once the WebSocket reports it finished."""
        with self.client.websocket_connect(f"/ws/jobs/{job_id}") as socket:
            while True:
                message = socket.receive_json()
                if message["type"] == "job" and message["job"]["finished"]:
                    return message["job"]

    def test_catalogue_lists_kinds_engines_and_cases_without_prompts(self) -> None:
        data = self.client.get("/api/engines").json()
        self.assertEqual([k["id"] for k in data["input_kinds"]], list(violation_cases.INPUT_KINDS))
        engines = {e["id"]: e for e in data["engines"]}
        self.assertEqual([c["id"] for c in engines[PUSH]["cases"]],
                         ["push_2_lsp_cargo", "push_3_lsp_cargo", "push_2_lsp_empty_extra"])
        self.assertEqual(engines[PUSH]["input_kinds"], ["forklift-with-lsp-cargo", "forklift-with-lsp-empty"])
        self.assertEqual(engines[SKIDS]["input_kinds"], ["forklift-with-cargo-no-lsp", "forklift-empty"])
        self.assertNotIn("changes", engines[PUSH]["cases"][0])

    def test_a_folder_job_uses_the_engines_kinds_one_output_per_task_three_at_a_time(self) -> None:
        response = self.create()
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        self.assertEqual({(i["kind"], i["name"]) for i in job["inputs"]},
                         {("forklift-with-lsp-cargo", n) for n in ("a.jpg", "b.png", "e.jpg", "f.jpg")}
                         | {("forklift-with-lsp-empty", n) for n in ("c.jpg", "g.jpg")})
        self.assertEqual(job["total"], 6 * 2)                 # at most 4 per frame, but only 2 cases, each once
        per_frame = {}
        for task in job["tasks"]:
            per_frame.setdefault(task["input"], []).append(task["case"])
        self.assertEqual(per_frame["a.jpg"], ["push_2_lsp_cargo", "push_3_lsp_cargo"])
        notes = " ".join(job["notes"])
        self.assertIn("unsorted", notes)                                  # not a kind folder
        self.assertIn("1 forklift-empty image(s) left out", notes)        # feeds the other engine
        final = self.wait_finished(job["id"])
        self.assertEqual(final["counts"], {"queued": 0, "running": 0, "done": 10, "failed": 2, "interrupted": 0})
        self.assertEqual(len(self.calls), 12)
        self.assertLessEqual(self.peak, service.WORKERS)
        self.assertEqual(len({str(call["output"]) for call in self.calls}), 12)            # one output each
        self.assertEqual(len({(call["image"], call["seed"]) for call in self.calls}), 12)  # each its own seed
        self.assertEqual(final["tokens"]["used"], 10 * 20_000 + 2 * 5_000)
        self.assertEqual(final["tokens"]["outputs_counted"], 12)
        folder = self.root / "jobs" / job["id"]
        self.assertTrue((folder / "inputs" / "forklift-with-lsp-cargo" / "a.jpg").is_file())   # inputs kept
        self.assertTrue(all(c["image"] == Path(c["image"]).name for c in self.calls))
        stored = json.loads((folder / "job.json").read_text())
        self.assertEqual(sum(t["status"] == "done" for t in stored["tasks"]), 10)
        frame = job["inputs"][0]
        self.assertEqual(self.client.get(frame["url"]).status_code, 200)
        self.assertEqual(self.client.get(frame["thumb"]).headers["content-type"], "image/jpeg")
        self.assertEqual([j["id"] for j in self.client.get("/api/jobs").json()], [job["id"]])
        empty = next(c for c in self.calls if c["image"] == "c.jpg" and "push_3" in c["output"].name)
        self.assertEqual(empty["change"], violation_cases.change(PUSH, "push_3_lsp_cargo", "forklift-with-lsp-empty"))
        self.assertIn("EMPTY LSP", empty["scene"])
        self.assertEqual(empty["output"].parent.name, "forklift-with-lsp-empty")
        self.assertEqual(empty["refs"], ["LSP", "SKID"])

        done = next(t for t in final["tasks"] if t["status"] == "done")
        self.assertEqual(self.client.get(done["url"]).headers["content-type"], "image/png")
        self.assertEqual(self.client.get(done["thumb"]).headers["content-type"], "image/jpeg")
        archive = zipfile.ZipFile(io.BytesIO(self.client.get(f"/api/jobs/{job['id']}/download.zip").content))
        self.assertEqual(len(archive.namelist()), 10)
        self.assertTrue(all("/" not in name for name in archive.namelist()))       # flat: one folder

    def test_images_uploaded_per_kind_feed_their_engine(self) -> None:
        response = self.client.post(
            "/api/jobs", data={"engine": SKIDS, "cases": ["carry_2_skids_side_by_side"], "project_id": "1"},
            files=[("files:forklift-empty", ("my frame.png", png(), "image/png")),
                   ("files:forklift-with-cargo-no-lsp", ("e.png", png(), "image/png")),
                   ("files:forklift-with-lsp-cargo", ("f.png", png(), "image/png")),
                   ("files:forklift-empty", ("notes.txt", b"x", "text/plain"))])
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        self.assertEqual(job["total"], 2)
        self.assertEqual({t["input"] for t in job["tasks"]}, {"my_frame.png", "e.png"})
        self.assertTrue(any("forklift-with-lsp-cargo" in note for note in job["notes"]))
        self.assertEqual(self.wait_finished(job["id"])["counts"]["done"], 2)
        self.assertEqual({tuple(c["refs"]) for c in self.calls}, {("SKID",)})

    def test_a_zip_of_kind_folders_is_an_input(self) -> None:
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("batch/forklift-with-lsp-empty/g.png", png())
            archive.writestr("batch/forklift-with-lsp-empty/readme.txt", "x")
            archive.writestr("../evil.png", png())
        response = self.client.post("/api/jobs", data={"engine": PUSH, "cases": ["push_2_lsp_empty_extra"],
                                                       "project_id": "1"},
                                    files=[("zip", ("inputs.zip", data.getvalue(), "application/zip"))])
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        self.assertEqual([(i["kind"], i["name"]) for i in job["inputs"]], [("forklift-with-lsp-empty", "g.png")])
        self.assertFalse((self.root / "jobs" / "evil.png").exists())
        self.assertEqual(self.wait_finished(job["id"])["counts"]["done"], 1)

    def test_a_dry_run_counts_outputs_and_tokens_and_starts_nothing(self) -> None:
        response = self.create(dry_run="1")
        self.assertEqual(response.status_code, 200, response.text)
        plan = response.json()
        self.assertEqual(plan["outputs"], 12)
        self.assertEqual(plan["frames"], {"forklift-with-lsp-cargo": 4, "forklift-with-lsp-empty": 2})
        self.assertEqual(plan["tokens"]["median"], 12 * plan["tokens"]["per_output"]["median"])
        time.sleep(0.2)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.client.get("/api/jobs").json(), [])
        self.assertFalse(any((self.root / "jobs").glob("2*")) if (self.root / "jobs").exists() else False)

    def test_a_stored_job_reloads_and_its_unfinished_tasks_are_not_rerun(self) -> None:
        job = self.create(cases=["push_2_lsp_cargo"]).json()
        self.wait_finished(job["id"])
        calls = len(self.calls)

        def stopped(tasks):                                   # as if the service had stopped during these
            tasks[0].update(status="running")
            tasks[1].update(status="queued")
        self.restart_after(job["id"], stopped)
        reloaded = self.client.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(reloaded["counts"]["interrupted"], 2)
        self.assertTrue(reloaded["finished"])
        time.sleep(0.2)
        self.assertEqual(len(self.calls), calls)               # nothing ran again
        done = next(t for t in reloaded["tasks"] if t["status"] == "done")
        self.assertEqual(self.client.get(done["url"]).status_code, 200)
        self.assertEqual(len(reloaded["inputs"]), 6)

    def test_codex_room_is_calibrated_and_a_job_over_it_is_refused_unless_forced(self) -> None:
        status = self.client.get("/api/codex").json()
        self.assertEqual(status["reading"]["used_percent"], 40)
        self.assertIsNone(status["outputs_left"])                     # not calibrated yet
        job = self.create().json()                                     # 12 outputs; fits while the room is unknown
        self.used_percent = 46                                         # what its 10 outputs take while it runs
        self.wait_finished(job["id"])
        time.sleep(0.2)                                                # the reading taken when the job finished
        status = self.client.get("/api/codex").json()
        self.assertTrue(status["calibration"]["calibrated"])
        self.assertAlmostEqual(status["calibration"]["percent_per_output"], 0.6)
        self.assertEqual(status["outputs_left"], 90)                   # (100 - 46) / 0.6
        plan = self.create(dry_run="1").json()
        self.assertTrue(plan["codex"]["fits"])
        self.used_percent = 96                                         # room for 6 outputs, the job makes 12
        self.client.get("/api/codex?refresh=true")
        refused = self.create()
        self.assertEqual(refused.status_code, 409)
        self.assertIn("room for about 6 more", refused.json()["detail"]["message"])
        self.assertFalse(self.create(dry_run="1").json()["codex"]["fits"])
        forced = self.create(force="1")
        self.assertEqual(forced.status_code, 200)
        self.wait_finished(forced.json()["id"])

    def test_an_unenforced_codex_limit_only_warns(self) -> None:
        self.limit_reached = True
        with mock.patch.object(service, "ENFORCE_CODEX_LIMIT", False):
            plan = self.create(dry_run="1").json()
            self.assertTrue(plan["codex"]["fits"])
            self.assertIn("Not enforced", plan["codex"]["reason"])
            response = self.create(cases=["push_2_lsp_cargo"])
            self.assertEqual(response.status_code, 200, response.text)
            self.assertFalse(self.client.get("/api/codex").json()["enforced"])
        self.wait_finished(response.json()["id"])

    def test_a_reached_codex_limit_blocks_new_jobs(self) -> None:
        self.limit_reached = True
        response = self.create()
        self.assertEqual(response.status_code, 409)
        self.assertIn("limit is reached", response.json()["detail"]["message"])
        self.assertTrue(self.client.get("/api/codex").json()["blocked"])

    def test_a_rebuild_adds_one_output_with_the_users_request_and_keeps_the_first(self) -> None:
        job = self.create(cases=["push_2_lsp_cargo"]).json()
        final = self.wait_finished(job["id"])
        first = next(t for t in final["tasks"] if t["status"] == "done")
        self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/tasks/{first['id']}/rebuild",
                                          data={"note": "  "}).status_code, 400)
        answer = self.client.post(f"/api/jobs/{job['id']}/tasks/{first['id']}/rebuild",
                                  data={"note": "make the second LSP touch the first"})
        self.assertEqual(answer.status_code, 200, answer.text)
        task = answer.json()["task"]
        self.assertEqual((task["rebuild_of"], task["rebuild_round"]), (first["id"], 1))
        self.assertTrue(task["output_name"].endswith("__v1__r1.png"))
        state = self.wait_finished(job["id"])
        self.assertEqual(state["total"], final["total"] + 1)
        call = next(c for c in self.calls if c["output"].name == task["output_name"])
        # the rejected output goes in after the references (LSP image 2, SKID image 3) with the feedback on it
        self.assertEqual([r["id"] for r in call["references"]], ["LSP", "SKID", "REJECTED"])
        rejected = call["references"][-1]["images"][0]
        self.assertEqual(rejected.read_bytes(), (self.root / "jobs" / job["id"] / "outputs" / first["path"]).read_bytes())
        self.assertIn("image 4 is an earlier result of this same edit that the reviewer rejected", call["change"])
        self.assertIn("feedback on it is: make the second LSP touch the first", call["change"])
        self.assertEqual(call["image"], first["input"])
        self.client.post(f"/api/jobs/{job['id']}/delete", data={"task": [first["id"]]})
        self.assertTrue(rejected.is_file())                          # kept for the rebuild though the output is deleted
        again = self.client.post(f"/api/jobs/{job['id']}/tasks/{task['id']}/rebuild", data={"note": "wider"}).json()
        self.assertEqual((again["task"]["rebuild_of"], again["task"]["rebuild_round"]), (first["id"], 2))
        self.wait_finished(job["id"])
        call = next(c for c in self.calls if c["output"].name == again["task"]["output_name"])
        redo = call["references"][-1]["images"][0]                    # a redo of the redo: the r1 output goes in
        self.assertEqual(redo.read_bytes(), (self.root / "jobs" / job["id"] / "outputs" / task["path"]).read_bytes())
        self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/tasks/t9999/rebuild",
                                          data={"note": "x"}).status_code, 404)

    def test_a_batch_rebuild_adds_one_output_per_chosen_output_with_one_request(self) -> None:
        job = self.create().json()                                    # 12 outputs, 10 done, 2 failed
        final = self.wait_finished(job["id"])
        done = [t for t in final["tasks"] if t["status"] == "done"]
        failed = next(t for t in final["tasks"] if t["status"] == "failed")
        chosen = [done[0]["id"], done[3]["id"], failed["id"], done[0]["id"]]       # the repeat is made once
        answer = self.client.post(f"/api/jobs/{job['id']}/rebuild",
                                  data={"task": chosen, "note": "  close the gap   between the LSPs "})
        self.assertEqual(answer.status_code, 200, answer.text)
        added = answer.json()["tasks"]
        self.assertEqual([t["rebuilt_from"] for t in added], [done[0]["id"], done[3]["id"], failed["id"]])
        self.assertEqual({t["note"] for t in added}, {"close the gap between the LSPs"})
        self.assertTrue(all(t["output_name"].endswith("__r1.png") for t in added))
        state = self.wait_finished(job["id"])
        self.assertEqual(state["total"], final["total"] + 3)
        for task in added:
            call = next(c for c in self.calls if c["output"].name == task["output_name"])
            self.assertIn("close the gap between the LSPs", call["change"])
        made = [t for t in state["tasks"] if t["id"] in {a["id"] for a in added}]
        self.assertEqual([t["status"] for t in made], ["done", "done", "failed"])   # fake_edit fails b.png v2 again
        self.assertFalse(any("review" in t for t in made))
        self.assertEqual(state["review"]["unreviewed"], final["review"]["unreviewed"] + 2)   # new ones await review
        again = self.client.post(f"/api/jobs/{job['id']}/rebuild",
                                 data={"task": [done[0]["id"], added[0]["id"]], "note": "wider"}).json()["tasks"]
        self.assertEqual([t["rebuild_round"] for t in again], [2, 3])             # rounds count from the first output
        self.assertEqual(len({t["output_name"] for t in again}), 2)
        self.wait_finished(job["id"])

    def test_a_batch_rebuild_refuses_bad_choices_and_adds_nothing(self) -> None:
        job = self.create().json()
        busy = job["tasks"][-1]["id"]                                  # still waiting for a free thread
        url = f"/api/jobs/{job['id']}/rebuild"
        self.assertEqual(self.client.post(url, data={"task": [busy], "note": "x"}).status_code, 409)
        final = self.wait_finished(job["id"])
        done = next(t for t in final["tasks"] if t["status"] == "done")["id"]
        self.assertEqual(self.client.post(url, data={"task": [done], "note": "  "}).status_code, 400)
        self.assertEqual(self.client.post(url, data={"note": "x"}).status_code, 400)
        self.assertEqual(self.client.post(url, data={"task": [done], "note": "x" * 1001}).status_code, 400)
        self.assertEqual(self.client.post(url, data={"task": [done, "t9999"], "note": "x"}).status_code, 404)
        self.assertEqual(self.client.post("/api/jobs/nojob/rebuild", data={"task": [done], "note": "x"}).status_code, 404)
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}").json()["total"], final["total"])
        self.limit_reached = True
        self.client.get("/api/codex?refresh=true")
        refused = self.client.post(url, data={"task": [done], "note": "x"})
        self.assertEqual(refused.status_code, 409)
        self.assertIn("codex", refused.json()["detail"])
        self.assertEqual(self.client.post(url, data={"task": [done], "note": "x", "force": "1"}).status_code, 200)
        self.wait_finished(job["id"])

    def test_resume_runs_the_interrupted_outputs_and_the_failed_ones_on_request(self) -> None:
        job = self.create().json()                                    # 12 outputs, 2 of them fail (b.png v2)
        self.wait_finished(job["id"])

        def stopped(tasks):                                           # as if the service stopped during these
            for task in tasks[:4]:
                if task["status"] == "done":
                    task.update(status="running")
        self.restart_after(job["id"], stopped)
        state = self.client.get(f"/api/jobs/{job['id']}").json()
        interrupted = state["counts"]["interrupted"]
        self.assertGreater(interrupted, 0)
        calls = len(self.calls)
        answer = self.client.post(f"/api/jobs/{job['id']}/resume")
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()["resumed"], interrupted)
        final = self.wait_finished(job["id"])
        self.assertEqual(final["counts"]["interrupted"], 0)
        self.assertEqual(len(self.calls) - calls, interrupted)      # only those ran again
        retry = self.client.post(f"/api/jobs/{job['id']}/resume", data={"failed": "1"}).json()
        self.assertEqual(retry["resumed"], 2)
        self.wait_finished(job["id"])
        self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/resume").status_code, 400)   # nothing left

    def test_delete_removes_outputs_but_keeps_their_tokens_and_names(self) -> None:
        job = self.create().json()                                    # 12 outputs, 10 done, 2 failed
        final = self.wait_finished(job["id"])
        done = [t for t in final["tasks"] if t["status"] == "done"]
        failed = next(t for t in final["tasks"] if t["status"] == "failed")
        gone = [done[0], failed]
        folder = self.root / "jobs" / job["id"]
        self.client.get(done[0]["thumb"])                             # make its thumbnail
        answer = self.client.post(f"/api/jobs/{job['id']}/delete", data={"task": [t["id"] for t in gone]})
        self.assertEqual(answer.status_code, 200, answer.text)
        state = answer.json()["job"]
        self.assertEqual(state["total"], 10)
        self.assertEqual(state["deleted"], 2)
        self.assertEqual(state["review"]["outputs"], 9)
        self.assertEqual(state["tokens"]["used"], final["tokens"]["used"])          # their tokens still count
        self.assertFalse((folder / "outputs" / done[0]["path"]).exists())
        self.assertFalse(any((folder / "thumbs").rglob(Path(done[0]["output_name"]).stem + ".jpg")))
        self.assertEqual(self.client.get(done[0]["url"]).status_code, 404)
        stored = json.loads((folder / "job.json").read_text())
        self.assertEqual([t["id"] for t in stored["deleted_tasks"]], [t["id"] for t in gone])
        service.jobs.clear()
        service.load_jobs(service.db.current())                                           # survives a restart
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}").json()["deleted"], 2)
        # a rebuild of a sibling gets a new id and name, never one of a deleted output
        source = done[1]
        first = self.client.post(f"/api/jobs/{job['id']}/tasks/{source['id']}/rebuild", data={"note": "x"}).json()
        self.assertEqual(first["task"]["id"], "t0013")
        self.wait_finished(job["id"])
        self.client.post(f"/api/jobs/{job['id']}/delete", data={"task": first["task"]["id"]})
        second = self.client.post(f"/api/jobs/{job['id']}/tasks/{source['id']}/rebuild", data={"note": "y"}).json()
        self.assertEqual(second["task"]["id"], "t0014")
        self.assertNotEqual(second["task"]["output_name"], first["task"]["output_name"])
        self.wait_finished(job["id"])
        # refused: unknown, nothing chosen, or still being made; nothing is deleted then
        self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/delete", data={"task": ["t9999", done[2]["id"]]}).status_code, 404)
        self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/delete").status_code, 400)
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}").json()["total"], 11)

    def test_every_chosen_case_makes_one_output_of_every_frame_it_takes(self) -> None:
        plan = self.create(dry_run="1").json()
        job = self.create().json()                                    # 6 frames, 2 cases: 12 outputs
        self.assertEqual(plan["outputs"], job["total"])
        self.assertEqual(job["total"], 12)
        pairs = [(t["input"], t["case"]) for t in job["tasks"]]
        self.assertEqual(len(set(pairs)), 12)                         # each frame and case once
        self.assertEqual(len({t["seed"] for t in job["tasks"]}), 12)
        self.assertNotIn("variants", job)
        self.assertNotIn("rates", job)
        self.wait_finished(job["id"])

    def test_cancel_stops_the_waiting_outputs_and_resume_runs_them_once(self) -> None:
        self.gate.clear()                                             # the first 3 outputs hold their threads
        job = self.create().json()                                    # 12 outputs
        self.wait_running(job["id"], service.WORKERS)
        answer = self.client.post(f"/api/jobs/{job['id']}/cancel")
        self.assertEqual(answer.status_code, 200, answer.text)
        cancelled = answer.json()["cancelled"]
        self.assertEqual(cancelled, 12 - service.WORKERS)
        self.assertEqual(answer.json()["job"]["counts"]["queued"], 0)
        self.gate.set()
        final = self.wait_finished(job["id"])
        self.assertEqual(final["counts"]["interrupted"], cancelled)
        self.assertEqual(len(self.calls), service.WORKERS)             # the cancelled ones never ran
        self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/cancel").status_code, 400)   # nothing waits
        self.gate.clear()                                             # resume, then cancel again while they wait
        self.client.post(f"/api/jobs/{job['id']}/resume")
        self.client.post(f"/api/jobs/{job['id']}/cancel")
        self.client.post(f"/api/jobs/{job['id']}/resume")             # the stale submits must stay idle
        self.gate.set()
        final = self.wait_finished(job["id"])
        self.assertEqual(final["counts"]["interrupted"], 0)
        self.assertEqual(len(self.calls), 12)                          # every output made exactly once
        self.assertEqual(len({str(call["output"]) for call in self.calls}), 12)

    def test_retry_accumulates_structured_usage_and_survives_reload(self):
        job = self.create().json()
        first = self.wait_finished(job["id"])
        with mock.patch.object(service, "edit_frame", return_value={"error": "failed again", "seconds": 1,
                "tokens": 120, "token_usage": {"input_tokens": 100, "cached_input_tokens": 80,
                "output_tokens": 20, "reasoning_output_tokens": 10, "source": "codex_json"}}):
            self.client.post(f"/api/jobs/{job['id']}/resume", data={"failed": "1"})
            final = self.wait_finished(job["id"])
        self.assertEqual(final["tokens"]["used"], first["tokens"]["used"] + 240)
        self.assertEqual(final["tokens"]["output_tokens"], 40)
        self.assertEqual(final["tokens"]["cached_input_tokens"], 160)
        failed = next(t for t in final["tasks"] if t["status"] == "failed")
        self.assertEqual(len(failed["token_attempts"]), 2)
        self.assertNotEqual(failed["token_attempts"][0]["log"], failed["token_attempts"][1]["log"])
        loaded = service.Job.load(self.root / "jobs" / job["id"])
        self.assertEqual(loaded.public()["tokens"]["used"], final["tokens"]["used"])

    def test_legacy_job_migration_is_persisted(self):
        job = self.create(cases=["push_2_lsp_cargo"]).json()
        self.wait_finished(job["id"])
        folder = self.root / "jobs" / job["id"]
        stored = json.loads((folder / "job.json").read_text())
        for task in stored["tasks"]:
            task.pop("token_attempts", None)
            task.pop("token_usage", None)
        (folder / "job.json").write_text(json.dumps(stored))
        loaded = service.Job.load(folder)
        loaded.save()                                                 # as load_jobs does for a job new to the database
        saved = json.loads((folder / "job.json").read_text())
        self.assertEqual(saved["tasks"][0]["token_usage"]["input_tokens"], 20000)
        self.assertIsNone(saved["tasks"][0]["token_usage"]["output_tokens"])
        self.assertEqual(service.Job.load(folder).public()["tokens"], loaded.public()["tokens"])

    def test_review_marks_exclusion_and_filtered_downloads(self) -> None:
        job = self.create().json()                                    # 12 outputs, 10 done
        final = self.wait_finished(job["id"])
        done = [t for t in final["tasks"] if t["status"] == "done"]
        base = f"/api/jobs/{job['id']}"
        self.assertEqual(final["review"], {"good": 0, "bad": 0, "unreviewed": 10, "outputs": 10})
        failed = next(t for t in final["tasks"] if t["status"] == "failed")
        self.assertEqual(self.client.post(f"{base}/tasks/{failed['id']}/review", data={"verdict": "good"}).status_code, 409)
        self.assertEqual(self.client.post(f"{base}/tasks/{done[0]['id']}/review", data={"verdict": "maybe"}).status_code, 400)
        for task in done[:3]:
            self.client.post(f"{base}/tasks/{task['id']}/review", data={"verdict": "good"})
        answer = self.client.post(f"{base}/tasks/{done[3]['id']}/review", data={"verdict": "bad"}).json()
        self.assertEqual(answer["review"], {"good": 3, "bad": 1, "unreviewed": 6, "outputs": 10})
        cleared = self.client.post(f"{base}/tasks/{done[2]['id']}/review", data={"verdict": "clear"}).json()
        self.assertEqual(cleared["review"]["good"], 2)

        def names(**params):
            response = self.client.get(f"{base}/download.zip", params=params)
            return response.status_code, sorted(zipfile.ZipFile(io.BytesIO(response.content)).namelist()) if response.status_code == 200 else []
        self.assertEqual(names(filter="good")[1], sorted(t["output_name"] for t in done[:2]))
        self.assertEqual(names(filter="bad")[1], [done[3]["output_name"]])
        by_case = names(filter="all", layout="by_case")[1]
        self.assertEqual(len(by_case), 10)
        self.assertTrue(all(name.split("/")[0] in ("push_2_lsp_cargo", "push_3_lsp_cargo") for name in by_case))
        self.assertEqual(self.client.get(f"{base}/download.zip", params={"filter": "nope"}).status_code, 400)

        # X: the input of the first good output leaves review counts and every download; undo puts it back
        kind, name = done[0]["input_kind"], done[0]["input"]
        out = self.client.post(f"{base}/inputs/{kind}/{name}/exclude", data={"excluded": "1"}).json()
        self.assertEqual(out["excluded_inputs"], [f"{kind}/{name}"])
        left = [t for t in done if t["input"] != name]
        self.assertEqual(out["review"]["outputs"], len(left))
        self.assertTrue(all(name not in n for n in names(filter="all")[1]))
        stored = json.loads((self.root / "jobs" / job["id"] / "job.json").read_text())
        self.assertEqual(stored["excluded_inputs"], [f"{kind}/{name}"])
        self.assertEqual(stored["tasks"][int(done[3]["id"][1:]) - 1]["review"], "bad")
        back = self.client.post(f"{base}/inputs/{kind}/{name}/exclude", data={"excluded": "0"}).json()
        self.assertEqual(back["excluded_inputs"], [])
        self.assertEqual(self.client.post(f"{base}/inputs/{kind}/missing.png/exclude").status_code, 404)

    def test_bad_requests_are_refused(self) -> None:
        flat = self.root / "flat"
        flat.mkdir()
        Image.new("RGB", (8, 8)).save(flat / "x.jpg")
        cases = [dict(folder=str(self.root / "missing")), dict(folder=str(flat)), dict(cases=["nope"]),
                 dict(engine=SKIDS), dict(project_id="x"), dict(folder="")]
        for fields in cases:
            with self.subTest(fields=fields):
                self.assertEqual(self.create(**fields).status_code, 400)
        only_skid_kinds = self.root / "skid_only"
        (only_skid_kinds / "forklift-empty").mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(only_skid_kinds / "forklift-empty" / "x.jpg")
        refused = self.create(folder=str(only_skid_kinds))
        self.assertEqual(refused.status_code, 400)
        self.assertIn("it takes forklift-with-lsp-cargo", refused.json()["detail"])
        self.assertEqual(self.client.get("/api/jobs/none").status_code, 404)

    def test_outputs_outside_the_job_are_not_served(self) -> None:
        job = self.create(cases=["push_2_lsp_cargo"]).json()
        self.wait_finished(job["id"])
        base = f"/api/jobs/{job['id']}/outputs"
        self.assertEqual(self.client.get(f"{base}/forklift-with-lsp-cargo/..%2F..%2Fsecret.png").status_code, 404)
        self.assertEqual(self.client.get(f"{base}/forklift-with-lsp-cargo/x.png").status_code, 404)


class CaseTests(unittest.TestCase):
    def test_an_empty_input_lsp_stays_empty_and_only_new_lsps_get_cargo(self) -> None:
        for case_id in ("push_2_lsp_cargo", "push_3_lsp_cargo", "push_2_lsp_empty_extra"):
            text = violation_cases.change(PUSH, case_id, "forklift-with-lsp-empty")
            self.assertIn("original LSP the forklift pushes stays EMPTY", text)
            self.assertNotIn("On the original", text)

    def test_the_skids_engine_has_a_side_by_side_and_a_stacked_case(self) -> None:
        cases = violation_cases.ENGINES[SKIDS]["cases"]
        self.assertEqual(list(cases), ["carry_2_skids_side_by_side", "carry_2_cargo_stacked"])
        for kind in ("forklift-empty", "forklift-with-cargo-no-lsp"):
            side = violation_cases.change(SKIDS, "carry_2_skids_side_by_side", kind)
            self.assertIn("one behind the other" if kind == "forklift-empty" else "side by side", side)
            stacked = violation_cases.change(SKIDS, "carry_2_cargo_stacked", kind)
            self.assertIn("stacked one on top of the other", stacked)
            self.assertIn("No LSP", stacked)

    def test_the_lsp_reference_is_the_fixed_crop_sheet_and_only_a_fallback(self) -> None:
        from synth import try_astra_edit_batch as batch
        self.assertEqual(batch.reference_image("LSP", Path("/nowhere")), batch.LSP_REFERENCE)
        self.assertEqual(batch.reference_image("SKID", Path("/run")), Path("/run/refs/SKID.png"))
        self.assertTrue(batch.LSP_REFERENCE.is_file())
        with tempfile.TemporaryDirectory() as folder:
            reference_map = Path(folder) / "reference_map.json"
            reference_map.write_text(json.dumps({"SKID": {"variant": "skid_SK046-1"}}))
            with mock.patch.object(batch, "ROOT", batch.ROOT):
                prompt = batch.reference_prompt(reference_map, "add one LSP")
        self.assertIn("to use ONLY when image 1 shows no LSP", prompt)
        self.assertIn("never copy the grid, the captions under the tiles or any of its text", prompt)
        self.assertEqual(batch.LSP_REFERENCE.name, "ref-lsp.jpg")
        self.assertNotIn("3D renders of real objects", prompt)

    def test_floor_opening_takes_closed_frames_and_has_four_open_cases(self) -> None:
        engine = violation_cases.ENGINES[FLOOR]
        self.assertEqual(list(engine["cases"]), ["open_with_person", "open_no_person", "open_main_door_closed",
                                                 "open_main_door_open"])
        for case_id, case in engine["cases"].items():
            self.assertEqual(list(case["changes"]), ["floor-closed"])
            self.assertEqual(case["refs"], ["FLOOR_OPENING"])
            self.assertIn("is now OPEN", violation_cases.change(FLOOR, case_id, "floor-closed"))
        self.assertIn("Nobody is in the scene", violation_cases.change(FLOOR, "open_no_person", "floor-closed"))
        self.assertIn("main door is CLOSED", violation_cases.change(FLOOR, "open_main_door_closed", "floor-closed"))
        self.assertIn("main door is OPEN", violation_cases.change(FLOOR, "open_main_door_open", "floor-closed"))

    def test_the_floor_prompt_has_no_warehouse_or_catalogue_and_runs_no_reference_map(self) -> None:
        from synth import try_astra_edit_batch as batch
        prompt = batch.reference_prompt(None, "open it", violation_cases.INPUT_KINDS["floor-closed"]["scene"],
                                        ["FLOOR_OPENING"], violation_cases.prompt_context(FLOOR))
        self.assertIn("floor hatch", prompt)
        self.assertIn("on-screen timestamp", prompt)
        self.assertNotIn("warehouse", prompt)
        self.assertNotIn("LSP", prompt)
        self.assertEqual(batch.reference_image("FLOOR_OPENING", Path("/run")), batch.FLOOR_OPENING_REFERENCE)
        with tempfile.TemporaryDirectory() as folder:
            frame = Path(folder) / "f.png"
            Image.new("RGB", (64, 36), "gray").save(frame)
            with mock.patch.object(batch.subprocess, "run", return_value=mock.Mock(returncode=1)) as run:
                row = batch.edit_frame(frame, Path(folder) / "run", Path(folder) / "out.png", 0,
                                       violation_cases.change(FLOOR, "open_no_person", "floor-closed"),
                                       refs=["FLOOR_OPENING"], context=violation_cases.prompt_context(FLOOR))
            commands = [call.args[0] for call in run.call_args_list]
            self.assertFalse(any("synth.catalogue_reference_map" in c for c in commands))
            self.assertIn("no image from Astra", row["error"])
            prompt_text = (Path(folder) / "run" / "prompt.txt").read_text()
            self.assertIn("floor hatch", prompt_text)
            self.assertNotIn("{lid_pose}", prompt_text)

    def test_the_hatch_cover_gets_a_random_angle_per_output_fixed_by_its_seed(self) -> None:
        text = violation_cases.change(FLOOR, "open_no_person", "floor-closed")
        self.assertIn("{lid_pose}", text)
        filled, choices = violation_cases.fill(text, 3)
        self.assertNotIn("{lid_pose}", filled)
        self.assertIn(f"about {choices['lid_angle']} degrees", filled)
        self.assertEqual(violation_cases.fill(text, 3), (filled, choices))          # same task, same angle
        angles = {violation_cases.fill(text, seed)[1]["lid_angle"] for seed in range(60)}
        self.assertGreater(len(angles), 8)
        self.assertTrue(min(angles) <= 35 and max(angles) >= 100)                   # from barely open to past upright
        other = violation_cases.change(FLOOR, "open_with_person", "floor-closed")
        self.assertNotEqual({violation_cases.fill(text, s)[1]["lid_angle"] for s in range(5)},
                            {violation_cases.fill(other, s)[1]["lid_angle"] for s in range(5)})
        lsp = violation_cases.change(PUSH, "push_2_lsp_cargo", "forklift-with-lsp-cargo")
        self.assertEqual(violation_cases.fill(lsp, 3), (lsp, {}))

    def test_the_warehouse_prompt_is_unchanged_by_engine_contexts(self) -> None:
        from synth import try_astra_edit_batch as batch
        with tempfile.TemporaryDirectory() as folder:
            reference_map = Path(folder) / "reference_map.json"
            reference_map.write_text(json.dumps({"SKID": {"variant": "skid_SK046-1"}}))
            prompt = batch.reference_prompt(reference_map, "add one LSP")
        self.assertIn("from a real CCTV camera in an air-cargo terminal warehouse:\n1. Full scene: the warehouse floor "
                      "with", prompt)
        self.assertIn("Do not change the camera angle, the floor, the red laser lines", prompt)

    def test_the_token_estimate_counts_only_fully_counted_runs(self) -> None:
        from synth.token_usage import legacy_usage
        for _ in range(20):
            service.remember_tokens(legacy_usage(20_000))                   # old partial readings: left out
        self.assertEqual(service.token_stats()["basis"], "first measurements")
        for n in range(5):
            service.remember_tokens({"source": "codex_json", "input_tokens": 120_000 + n * 1000,
                                     "cached_input_tokens": 100_000, "output_tokens": 1_200})
        stats = service.token_stats()
        self.assertEqual((stats["median"], stats["cached_median"], stats["output_median"], stats["samples"]),
                         (122_000, 100_000, 1_200, 5))

    def test_cases_only_take_their_engines_kinds(self) -> None:
        for engine_id, engine in violation_cases.ENGINES.items():
            for case in engine["cases"].values():
                for kind in case["changes"]:
                    self.assertEqual(violation_cases.INPUT_KINDS[kind]["engine"], engine_id)
        with self.assertRaises(KeyError):
            violation_cases.change(PUSH, "push_2_lsp_cargo", "forklift-empty")


if __name__ == "__main__":
    unittest.main()
