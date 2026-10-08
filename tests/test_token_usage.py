import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from synth.token_usage import migrate_task, read_usage, summarize


class TokenUsageTests(unittest.TestCase):
    def test_failed_edit_still_records_json_usage(self):
        from synth import try_astra_edit_batch as batch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            usage = {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 80}

            def run(command, **kwargs):
                if "--json" in command:
                    kwargs["stdout"].write(json.dumps({"type": "turn.completed", "usage": usage}) + "\n")
                    return mock.Mock(returncode=1)
                return mock.Mock(returncode=0)

            with mock.patch.object(batch, "camera_sized", return_value=root / "input.png"), \
                    mock.patch.object(batch, "reference_prompt", return_value="edit"), \
                    mock.patch.object(batch.subprocess, "run", side_effect=run) as called:
                result = batch.edit_frame(root / "input.png", root / "run", root / "out.png", 0, "edit", refs=())
            self.assertIn("--json", called.call_args.args[0])
            self.assertIn("error", result)
            self.assertEqual(result["token_usage"]["input_tokens"], 100)
            self.assertEqual(result["token_usage"]["output_tokens"], 20)
            self.assertEqual(result["tokens"], 120)

    def test_json_turns_sum_without_double_counting_cache_or_reasoning(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "astra.log"
            event = {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 80,
                     "output_tokens": 20, "reasoning_output_tokens": 10}}
            log.write_text("diagnostic\n" + json.dumps(event) + "\n{broken\n" + json.dumps(event))
            usage = read_usage(log)
            self.assertEqual(usage["input_tokens"], 200)
            self.assertEqual(usage["output_tokens"], 40)
            self.assertEqual(usage["cached_input_tokens"], 160)
            self.assertEqual(summarize([usage])["known_total_tokens"], 240)

    def test_legacy_migration_is_input_only_and_idempotent(self):
        task = {"status": "done", "tokens": 21301}
        self.assertTrue(migrate_task(task))
        self.assertFalse(migrate_task(task))
        self.assertEqual(task["token_usage"]["input_tokens"], 21301)
        self.assertIsNone(task["token_usage"]["output_tokens"])
        self.assertFalse(task["token_usage"]["complete"])
        self.assertEqual(len(task["token_attempts"]), 1)

    def test_missing_usage_is_unknown_and_zero_is_known(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "astra.log"
            self.assertIsNone(read_usage(log)["input_tokens"])
            log.write_text('tokens used\n0\n')
            self.assertEqual(read_usage(log)["input_tokens"], 0)
            self.assertIsNone(read_usage(log)["output_tokens"])
            log.write_text('{"type":"turn.completed","usage":{"input_tokens":0,"output_tokens":0}}')
            self.assertTrue(summarize([read_usage(log)])["complete"])
