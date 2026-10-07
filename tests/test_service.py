"""Behavioral checks for the generation service: input kinds, jobs, the 3-thread pool, live updates, downloads."""

from __future__ import annotations

import io
import json
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


def png(color: str = "gray") -> bytes:
    data = io.BytesIO()
    Image.new("RGB", (64, 36), color).save(data, format="PNG")
    return data.getvalue()


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.inputs = self.root / "frames"
        for kind, names in {"forklift-with-lsp-cargo": ["a.jpg", "b.png"], "forklift-with-lsp-empty": ["c.jpg"],
                            "forklift-empty": ["d.jpg"]}.items():
            (self.inputs / kind).mkdir(parents=True)
            for name in names:
                Image.new("RGB", (64, 36), "gray").save(self.inputs / kind / name)
        (self.inputs / "unsorted").mkdir()
        Image.new("RGB", (64, 36), "gray").save(self.inputs / "unsorted" / "x.jpg")
        self.running, self.peak, self.calls = 0, 0, []
        self.lock = threading.Lock()
        patches = [mock.patch.object(service, "JOBS_DIR", self.root / "jobs"),
                   mock.patch.object(service, "edit_frame", side_effect=self.fake_edit),
                   mock.patch.dict(service.jobs, clear=True),
                   mock.patch.object(service, "USAGE", CodexUsage(self.root / "usage.json", service.service_totals,
                                                                    reader=self.fake_limits))]
        self.used_percent, self.limit_reached = 40, False
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.folder.cleanup)
        # one event loop for the whole test, as under uvicorn: the pool threads post their updates to it
        self.client = self.enterContext(TestClient(service.app))

    def fake_limits(self) -> dict:
        return {"plan": "prolite", "ordinary_usage_allowed": not self.limit_reached, "limit_reached": self.limit_reached,
                "used_percent": self.used_percent, "window_minutes": 10080, "resets_at": 1_800_000_000,
                "secondary_used_percent": None, "secondary_resets_at": None, "read_at": time.time()}

    def fake_edit(self, image, run, output, seed, change, scene, refs=None):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
            self.calls.append({"image": image.name, "output": output, "seed": seed, "change": change,
                               "scene": scene, "refs": refs})
        time.sleep(0.05)
        with self.lock:
            self.running -= 1
        if image.name == "b.png" and "__v2" in output.name:
            return {"error": "Astra produced no image", "seconds": 1, "tokens": 5_000}
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 36), "red").save(output)
        return {"output": str(output), "seconds": 1, "tokens": 20_000}

    def create(self, **fields):
        data = {"engine": PUSH, "cases": ["push_2_lsp_cargo", "push_3_lsp_cargo"], "variants": "2",
                "folder": str(self.inputs)}
        data.update(fields)
        return self.client.post("/api/jobs", data=data)

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
                         {("forklift-with-lsp-cargo", "a.jpg"), ("forklift-with-lsp-cargo", "b.png"),
                          ("forklift-with-lsp-empty", "c.jpg")})
        self.assertEqual(job["total"], 3 * 2 * 2)
        notes = " ".join(job["notes"])
        self.assertIn("unsorted", notes)                                  # not a kind folder
        self.assertIn("1 forklift-empty image(s) left out", notes)        # feeds the other engine
        final = self.wait_finished(job["id"])
        self.assertEqual(final["counts"], {"queued": 0, "running": 0, "done": 10, "failed": 2, "interrupted": 0})
        self.assertEqual(len(self.calls), 12)
        self.assertLessEqual(self.peak, service.WORKERS)
        self.assertEqual(len({str(call["output"]) for call in self.calls}), 12)            # one output each
        self.assertEqual(len({(call["image"], call["seed"]) for call in self.calls}), 6)  # variants differ in seed
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
        self.assertTrue(all(name.split("/")[0] in violation_cases.INPUT_KINDS for name in archive.namelist()))

    def test_images_uploaded_per_kind_feed_their_engine(self) -> None:
        response = self.client.post(
            "/api/jobs", data={"engine": SKIDS, "cases": ["carry_2_skids_side_by_side"], "variants": "1"},
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
                                                       "variants": "1"},
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
        self.assertEqual(plan["frames"], {"forklift-with-lsp-cargo": 2, "forklift-with-lsp-empty": 1})
        self.assertEqual(plan["tokens"]["median"], 12 * plan["tokens"]["per_output"]["median"])
        time.sleep(0.2)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.client.get("/api/jobs").json(), [])
        self.assertFalse(any((self.root / "jobs").glob("2*")) if (self.root / "jobs").exists() else False)

    def test_a_stored_job_reloads_and_its_unfinished_tasks_are_not_rerun(self) -> None:
        job = self.create(variants="1").json()
        self.wait_finished(job["id"])
        folder = self.root / "jobs" / job["id"]
        stored = json.loads((folder / "job.json").read_text())
        stored["tasks"][0].update(status="running")          # as if the service had stopped during it
        stored["tasks"][1].update(status="queued")
        (folder / "job.json").write_text(json.dumps(stored))
        calls = len(self.calls)
        service.jobs.clear()
        service.load_jobs()                                   # what a restart does
        reloaded = self.client.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(reloaded["counts"]["interrupted"], 2)
        self.assertTrue(reloaded["finished"])
        time.sleep(0.2)
        self.assertEqual(len(self.calls), calls)               # nothing ran again
        done = next(t for t in reloaded["tasks"] if t["status"] == "done")
        self.assertEqual(self.client.get(done["url"]).status_code, 200)
        self.assertEqual(len(reloaded["inputs"]), 3)

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

    def test_a_reached_codex_limit_blocks_new_jobs(self) -> None:
        self.limit_reached = True
        response = self.create()
        self.assertEqual(response.status_code, 409)
        self.assertIn("limit is reached", response.json()["detail"]["message"])
        self.assertTrue(self.client.get("/api/codex").json()["blocked"])

    def test_a_rebuild_adds_one_output_with_the_users_request_and_keeps_the_first(self) -> None:
        job = self.create(variants="1", cases=["push_2_lsp_cargo"]).json()
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
        self.assertIn("Additional request for this image", call["change"])
        self.assertIn("make the second LSP touch the first", call["change"])
        self.assertEqual(call["image"], first["input"])
        again = self.client.post(f"/api/jobs/{job['id']}/tasks/{task['id']}/rebuild", data={"note": "wider"}).json()
        self.assertEqual((again["task"]["rebuild_of"], again["task"]["rebuild_round"]), (first["id"], 2))
        self.wait_finished(job["id"])
        self.assertEqual(self.client.get(first["url"]).status_code, 200)       # the first output stays
        self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/tasks/t9999/rebuild",
                                          data={"note": "x"}).status_code, 404)

    def test_resume_runs_the_interrupted_outputs_and_the_failed_ones_on_request(self) -> None:
        job = self.create().json()                                    # 12 outputs, 2 of them fail (b.png v2)
        self.wait_finished(job["id"])
        folder = self.root / "jobs" / job["id"]
        stored = json.loads((folder / "job.json").read_text())
        for task in stored["tasks"][:4]:                              # as if the service stopped during these
            if task["status"] == "done":
                task.update(status="running")
        (folder / "job.json").write_text(json.dumps(stored))
        service.jobs.clear()
        service.load_jobs()
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

    def test_bad_requests_are_refused(self) -> None:
        flat = self.root / "flat"
        flat.mkdir()
        Image.new("RGB", (8, 8)).save(flat / "x.jpg")
        cases = [dict(folder=str(self.root / "missing")), dict(folder=str(flat)), dict(cases=["nope"]),
                 dict(engine=SKIDS), dict(variants="0"), dict(folder="")]
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
        job = self.create(variants="1").json()
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

    def test_cases_only_take_their_engines_kinds(self) -> None:
        for engine_id, engine in violation_cases.ENGINES.items():
            for case in engine["cases"].values():
                for kind in case["changes"]:
                    self.assertEqual(violation_cases.INPUT_KINDS[kind]["engine"], engine_id)
        with self.assertRaises(KeyError):
            violation_cases.change(PUSH, "push_2_lsp_cargo", "forklift-empty")


if __name__ == "__main__":
    unittest.main()
