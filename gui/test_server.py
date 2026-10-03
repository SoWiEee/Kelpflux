import unittest

from gui.server import _aggregate_job_usage, _parse_pmon, _parse_sacct_history


class SacctHistoryTest(unittest.TestCase):
    def test_computes_jct_and_keeps_command_pipes(self):
        text = (
            "42|bert-train|COMPLETED|2026-10-04T10:00:00|2026-10-04T10:00:10|"
            "2026-10-04T10:02:00|110|slurm-worker-gpu-rtx4070-0|gres/mps=50|"
            "gres/mps=50|gpu|python train.py --name a|b|\n"
        )

        job = _parse_sacct_history(text)[0]

        self.assertEqual(job["jct_seconds"], 120)
        self.assertEqual(job["wait_seconds"], 10)
        self.assertEqual(job["runtime_seconds"], 110)
        self.assertEqual(job["command"], "python train.py --name a|b")

    def test_preserves_trailing_pipe_and_terminal_states(self):
        rows = "\n".join(
            f"{index}|job-{state}|{state}|2026-10-04T10:00:00|2026-10-04T10:00:01|"
            f"2026-10-04T10:00:02|1|node||gpu|gpu|printf foo ||"
            for index, state in enumerate(("BOOT_FAIL", "DEADLINE", "REVOKED"), 1)
        )

        jobs = _parse_sacct_history(rows)

        self.assertEqual({job["state"] for job in jobs}, {"BOOT_FAIL", "DEADLINE", "REVOKED"})
        self.assertTrue(all(job["command"] == "printf foo |" for job in jobs))

    def test_attributes_process_metrics_only_to_mapped_jobs(self):
        pmon = """
# gpu pid type sm mem enc dec jpg ofa fb ccpm command
0 123 M+C 37 12 - - - - 512 0 python
0 999 G - - - - - - 128 0 Xorg
"""
        processes = _parse_pmon(pmon)
        usage = _aggregate_job_usage(
            processes,
            {123: "42"},
            [{"gpu_index": 0, "gpu_uuid": "GPU-abc", "memory_total_mib": 10240}],
        )

        self.assertEqual(set(usage), {"42"})
        self.assertEqual(usage["42"]["sm_percent"], 37)
        self.assertEqual(usage["42"]["vram_used_mib"], 512)
        self.assertEqual(usage["42"]["vram_percent"], 5)

    def test_labels_mps_server_sm_as_shared_not_per_job(self):
        pmon = """
0 123 M+C - - - - - - 512 0 python
0 456 C 98 79 - - - - 30 0 nvidia-cuda-mps
"""
        usage = _aggregate_job_usage(
            _parse_pmon(pmon),
            {123: "42"},
            [{"gpu_index": 0, "gpu_uuid": "GPU-abc", "memory_total_mib": 10240}],
        )["42"]

        self.assertIsNone(usage["sm_percent"])
        self.assertEqual(usage["shared_gpu_sm_percent"], 98)


if __name__ == "__main__":
    unittest.main()
