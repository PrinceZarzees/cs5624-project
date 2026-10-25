"""Convert GraphArena problems into our common JSONL format.

GraphArena's processed data is a set of pickle files (``Distance_easy.pkl``,
``MCP_hard.pkl``, ...) inside ``dataset.zip``, which lives in a Google Drive
folder linked from https://github.com/squareRoot3/GraphArena. There are three
ways to get it:

    # 1. download + unzip + convert in one step (needs `pip install gdown`)
    python -m graphharness.grapharena_adapter --download

    # 2. you downloaded dataset.zip by hand in a browser
    python -m graphharness.grapharena_adapter --zip ~/Downloads/dataset.zip

    # 3. you already unzipped it
    python -m graphharness.grapharena_adapter --dataset-dir path/to/dataset

Raw files go to ``data/grapharena/`` (ignored by git) and the converted
problems to ``data/ga_easy.jsonl`` and ``data/ga_hard.jsonl``.

Only the GraphArena tasks that match our tasks are converted:

    Distance  -> shortest_path
    Connected -> connected_components
    Diameter  -> diameter
    MCP       -> max_clique

Nodes are named by GraphArena's ``name`` attribute (e.g. real author names).
Ground truth is recomputed with our own solvers so it is checked exactly the
same way as synthetic data; GraphArena's own answer is kept in ``meta``.

Note: GraphArena's "hard" graphs have at most 50 nodes, so they are still
"small" for this project. Larger test graphs come from ``build_data``.

Security note: pickle files can run code when loaded. Only load the official
GraphArena files.
"""

from __future__ import annotations

import argparse
import pickle
import zipfile
from pathlib import Path

import networkx as nx

from .data_loader import DEFAULT_DATA_DIR, save_jsonl
from .generators import graph_to_json
from .tasks import as_json_value, get_task

GA_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1mvJSUTrfOX13wgpkyb3w8s_SJqipnb1c"
GA_ZIP_NAME = "dataset.zip"
GA_ZIP_ID = "1NjULkBbM3kS15Q_1YVcPmTQ-DWiNJM9r"  # dataset.zip in that folder, ~1.4 GB

GA_TO_OURS = {
    "Distance": "shortest_path",
    "Connected": "connected_components",
    "Diameter": "diameter",
    "MCP": "max_clique",
}


def convert_problem(problem: dict, ga_task: str, difficulty: str, index: int) -> dict | None:
    """Convert one GraphArena problem dict. Returns None if it cannot be used."""
    task_name = GA_TO_OURS[ga_task]
    task = get_task(task_name)
    G0: nx.Graph = problem["graph"]

    names = {v: str(d.get("name", v)) for v, d in G0.nodes(data=True)}
    if len(set(names.values())) != len(names):
        return None  # two nodes share a name; answers would be ambiguous

    G = nx.Graph()
    G.add_nodes_from(names.values())
    G.add_edges_from((names[u], names[v]) for u, v in G0.edges())

    if task_name == "shortest_path":
        params = {"source": names[problem["source"]], "target": names[problem["target"]]}
    else:
        params = {}
    if task_name == "diameter" and not nx.is_connected(G):
        return None

    answer = task.solve(G, params)
    split = f"ga_{difficulty}"
    return {
        "id": f"{task_name}/{split}/{index:05d}",
        "source": "grapharena",
        "task": task_name,
        "split": split,
        "graph": graph_to_json(G),
        "params": as_json_value(params),
        "question": task.question(params),
        "answer": as_json_value(answer),
        "meta": {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "task_kind": task.kind,
            "grapharena_task": ga_task,
            "grapharena_id": problem.get("id", index),
            "grapharena_exact_answer": as_json_value(problem.get("exact_answer")),
            "original_prompt": problem.get("problem_text"),
        },
    }


def convert_dataset(dataset_dir: Path, difficulty: str, limit: int | None = None) -> list[dict]:
    records = []
    for ga_task in GA_TO_OURS:
        path = dataset_dir / f"{ga_task}_{difficulty}.pkl"
        if not path.exists():
            print(f"  skip {path.name} (not found)")
            continue
        with open(path, "rb") as f:
            problems = pickle.load(f)
        kept = 0
        for i, p in enumerate(problems):
            if limit is not None and kept >= limit:
                break
            rec = convert_problem(p, ga_task, difficulty, i)
            if rec is not None:
                records.append(rec)
                kept += 1
        print(f"  {ga_task:<10} -> {GA_TO_OURS[ga_task]:<22} {kept} problems")
    return records


def download_zip(raw_dir: Path) -> Path:
    """Download GraphArena's dataset.zip from Google Drive into raw_dir.

    Only dataset.zip (~1.4 GB) is fetched by its file id; the same folder
    also has source.zip and results.zip, which we do not need.
    """
    zip_path = raw_dir / GA_ZIP_NAME
    if zip_path.exists():
        print(f"using existing {zip_path}")
        return zip_path
    try:
        import gdown
    except ImportError:
        raise SystemExit("--download needs gdown: pip install gdown") from None

    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(".zip.part")
    print(f"downloading {GA_ZIP_NAME} (~1.4 GB)")
    try:
        gdown.download(id=GA_ZIP_ID, output=str(tmp), quiet=False, resume=True)
    except Exception as e:  # network / proxy / Google rate limit
        raise SystemExit(
            f"could not download from Google Drive ({e}).\n"
            f"Download {GA_ZIP_NAME} in a browser from\n"
            f"  https://drive.google.com/file/d/{GA_ZIP_ID}/view\n"
            f"then run: python -m graphharness.grapharena_adapter --zip path/to/{GA_ZIP_NAME}"
        ) from None
    tmp.rename(zip_path)
    return zip_path


def unzip(zip_path: Path, raw_dir: Path) -> Path:
    """Unzip zip_path into raw_dir (skipped if already done) and return raw_dir."""
    if find_dataset_dir(raw_dir, missing_ok=True) is None:
        print(f"unzipping {zip_path} -> {raw_dir}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
    return raw_dir


def find_dataset_dir(root: Path, missing_ok: bool = False) -> Path | None:
    """Return the folder under root that holds the GraphArena *.pkl files.

    The zip may unpack to ``dataset/`` or straight into root, so look for it.
    """
    root = Path(root)
    for path in sorted(root.rglob("*_easy.pkl")):
        if "__MACOSX" not in path.parts and path.name.split("_")[0] in GA_TO_OURS:
            return path.parent
    if missing_ok:
        return None
    raise SystemExit(f"no GraphArena *_easy.pkl / *_hard.pkl files found under {root}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--download", action="store_true", help=f"download {GA_ZIP_NAME} from Google Drive")
    src.add_argument("--zip", help=f"path to a {GA_ZIP_NAME} you downloaded yourself")
    src.add_argument("--dataset-dir", help="folder with already unzipped GraphArena *.pkl files")
    ap.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--raw-dir", help="where to put the zip and unzipped files (default: <out>/grapharena)")
    ap.add_argument("--limit", type=int, help="max problems per task (default: all)")
    args = ap.parse_args(argv)

    raw_dir = Path(args.raw_dir or Path(args.out) / "grapharena")
    if args.download:
        dataset_dir = find_dataset_dir(unzip(download_zip(raw_dir), raw_dir))
    elif args.zip:
        dataset_dir = find_dataset_dir(unzip(Path(args.zip), raw_dir))
    else:
        dataset_dir = find_dataset_dir(Path(args.dataset_dir))
    print(f"GraphArena files: {dataset_dir}")

    for difficulty in ("easy", "hard"):
        print(f"GraphArena {difficulty}:")
        records = convert_dataset(dataset_dir, difficulty, args.limit)
        if records:
            path = Path(args.out) / f"ga_{difficulty}.jsonl"
            save_jsonl(records, path)
            print(f"  wrote {len(records)} problems -> {path}")


if __name__ == "__main__":
    main()
