"""Write synthetic parquet fixtures for CPU dry runs: python scripts/make_fixture.py [--n 240]"""
import argparse
import os

from codedet.fixtures import make_fixture

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=240)
ap.add_argument("--out", default="fixtures")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
make_fixture(a.n, seed=0).to_parquet(f"{a.out}/task_a.parquet")
make_fixture(a.n, seed=1, human_share=0.5).to_parquet(f"{a.out}/task_b.parquet")
print("wrote", a.out)
