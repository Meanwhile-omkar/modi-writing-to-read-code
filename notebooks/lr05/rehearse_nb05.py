"""Execute every code cell of notebooks/05_line_reader_round2_GPU.ipynb on this PC's CPU.

Only the ENVIRONMENT is swapped: Kaggle paths -> a local folder, the HF download -> the local parquet, nvidia-smi ->
two pretend GPUs (= two CPU job slots), pip -> nothing. LR05_LOCAL=1 makes the notebook use a handful of patches and
tiny budgets. Everything else - renderer, benchmark, planner, scheduler, every job type, the combination, LM decoding,
the test score and the report - runs as written.
usage: python rehearse_nb05.py <scratch dir>
"""
import io, json, os, pathlib, shutil, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class LocalDS:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, k):
        if k == "filename":
            return list(self.df["filename"])
        raise KeyError(k)

    def __iter__(self):
        from PIL import Image
        for r in self.df.itertuples():
            yield {"image": Image.open(io.BytesIO(r.image["bytes"])), "filename": r.filename}

    def __len__(self):
        return len(self.df)


def main():
    run = pathlib.Path(sys.argv[1])
    shutil.rmtree(run, ignore_errors=True)
    (run / "out").mkdir(parents=True); (run / "work").mkdir()
    OUT, WORK = (run / "out").as_posix(), (run / "work").as_posix()
    META = (ROOT / "FYP-modi-script-code/assets/metadata").as_posix()
    os.environ["LR05_LOCAL"] = "1"; os.environ["PYTHONIOENCODING"] = "utf-8"
    import subprocess
    subprocess.run([sys.executable, str(HERE / "build_nb05.py")], check=True)
    nb = json.loads((HERE.parent / "05_line_reader_round2_GPU.ipynb").read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    SUB = [
        ("WORK = ('/kaggle/temp' if os.path.isdir('/kaggle/temp') else '/tmp') + '/lr05'", f"WORK = r'{WORK}'"),
        ("/kaggle/working", OUT),
        ("smi = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip()",
         "smi = 'GPU 0: pretend T4\\nGPU 1: pretend T4'"),
        ("!pip -q install datasets editdistance sacrebleu rapidfuzz 2>&1 | tail -1", "pass"),
        ("META = find_metadata()", f"META = r'{META}'"),
        ("from datasets import load_dataset", "pass"),
        ("ds = load_dataset('historyHulk/MoDeTrans', split='train')     # 2,043 rows", "ds = LOCAL_DS"),
    ]
    for a, _ in SUB:
        if not any(a in "".join(c["source"]) for c in cells):
            raise SystemExit(f"substitution target not found in the notebook: {a}")
    import pandas as pd, matplotlib
    matplotlib.use("Agg")
    pq = pd.read_parquet(ROOT / "FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet")
    G = {"LOCAL_DS": LocalDS(pq), "__name__": "__nb05__"}
    t0 = time.time()
    for i, c in enumerate(cells, 1):
        src = "".join(c["source"])
        first = src.split("\n", 1)[0]
        if first.startswith("%%writefile"):
            path = first.split(None, 1)[1].replace("/kaggle/working", OUT)
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(path).write_text(src.split("\n", 1)[1], encoding="utf-8")
            print(f"\n===== CELL {i}: wrote {path}", flush=True)
            continue
        for a, b in SUB:
            src = src.replace(a, b)
        print(f"\n===== CELL {i} [{time.time() - t0:.0f}s]: {first[:90]}", flush=True)
        try:
            exec(compile(src, f"cell{i}", "exec"), G)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"\n*** FAILED at cell {i}", flush=True)
            sys.exit(1)
    print(f"\n*** ALL CELLS RAN in {time.time() - t0:.0f}s ***", flush=True)


if __name__ == "__main__":
    main()
