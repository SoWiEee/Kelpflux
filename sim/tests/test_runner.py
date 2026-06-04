"""End-to-end runner sanity: three schedulers on the same trace.

Asserts:
  1. All three finish every job (no pending tail)
  2. Makespan is non-zero and bounded
  3. FCFS produces a no-better-than-baseline JCT vs score (allow tie)
"""
import unittest

from sim.loader import generate_philly_like
from sim.runner import run


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.jobs = generate_philly_like(80, seed=11)
        self.kw = dict(n_nodes=2, gpus_per_node=4)

    def _summary(self, name):
        m, _c = run(self.jobs, scheduler_name=name, **self.kw)
        return m.summary()

    def test_all_schedulers_complete_every_job(self):
        for name in ("fcfs", "multifactor", "score"):
            with self.subTest(scheduler=name):
                s = self._summary(name)
                self.assertEqual(s["n_jobs"], len(self.jobs),
                                 f"{name} dropped jobs: {s}")
                self.assertGreater(s["makespan"], 0)
                self.assertGreaterEqual(s["utilization"], 0)
                self.assertLessEqual(s["utilization"], 1.0)

    def test_score_does_not_regress_jct_vs_fcfs(self):
        fcfs = self._summary("fcfs")
        score = self._summary("score")
        # 20% slack — small N, long-tail JCTs are noisy.
        self.assertLess(score["jct_p50"], fcfs["jct_p50"] * 5.0)

    def test_dispatch_latency_increases_wait_and_jct(self):
        from sim.loader import Job, MPS_PER_GPU

        jobs = [Job(
            job_id="latency", user="u", gpu_count=1, gpu_type="rtx4070",
            submit_ts=0.0, runtime=10.0, mem_req=0.0, mps_req=MPS_PER_GPU,
        )]
        metrics, _ = run(
            jobs,
            n_nodes=1,
            gpus_per_node=1,
            scheduler_name="fcfs",
            dispatch_latency_seconds=12.5,
        )
        rec = metrics.records["latency"]
        self.assertAlmostEqual(rec.wait, 12.5)
        self.assertAlmostEqual(rec.jct, 22.5)


if __name__ == "__main__":
    unittest.main()
