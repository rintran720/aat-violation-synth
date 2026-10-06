"""Behavioral checks for the generation service: input kinds, jobs, the 3-thread pool, live updates, downloads."""

from __future__ import annotations

import io
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
                   mock.patch.dict(service.jobs, clear=True)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.folder.cleanup)
        # one event loop for the whole test, as under uvicorn: the pool threads post their updates to it
        self.client = self.enterContext(TestClient(service.app))

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
            return {"error": "Astra produced no image", "seconds": 1}
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 36), "red").save(output)
        return {"output": str(output), "seconds": 1}

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
        self.assertEqual(job["inputs"], {"forklift-with-lsp-cargo": 2, "forklift-with-lsp-empty": 1})
        self.assertEqual(job["total"], 3 * 2 * 2)
        notes = " ".join(job["notes"])
        self.assertIn("unsorted", notes)                                  # not a kind folder
        self.assertIn("1 forklift-empty image(s) left out", notes)        # feeds the other engine
        final = self.wait_finished(job["id"])
        self.assertEqual(final["counts"], {"queued": 0, "running": 0, "done": 10, "failed": 2})
        self.assertEqual(len(self.calls), 12)
        self.assertLessEqual(self.peak, service.WORKERS)
        self.assertEqual(len({str(call["output"]) for call in self.calls}), 12)            # one output each
        self.assertEqual(len({(call["image"], call["seed"]) for call in self.calls}), 6)  # variants differ in seed
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
        self.assertEqual(job["inputs"], {"forklift-with-lsp-empty": 1})
        self.assertFalse((self.root / "jobs" / "evil.png").exists())
        self.assertEqual(self.wait_finished(job["id"])["counts"]["done"], 1)

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

    def test_cases_only_take_their_engines_kinds(self) -> None:
        for engine_id, engine in violation_cases.ENGINES.items():
            for case in engine["cases"].values():
                for kind in case["changes"]:
                    self.assertEqual(violation_cases.INPUT_KINDS[kind]["engine"], engine_id)
        with self.assertRaises(KeyError):
            violation_cases.change(PUSH, "push_2_lsp_cargo", "forklift-empty")


if __name__ == "__main__":
    unittest.main()
