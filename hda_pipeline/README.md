# HDA pipeline (phase 1: polynomial-time tasks)

A Harness Developing Agent (HDA) writes a C++17, standard-library-only solver for a graph task
from k I/O examples (with or without the problem statement), refines it for n iterations against
hidden small-graph validation, and is finally tested on held-out graphs up to 100,000 nodes.

    pip install -r requirements.txt
    python -m pytest -q tests                      # 18 tests, ~25 s, needs g++
    python -m hda.generator --out data/hda         # all splits (~70 s)
    python -m hda.human_baseline --out runs/hda/human
    python -m hda.identifiability --data data/hda --out data/hda/identifiability.json
    python -m hda.agent --task diameter --condition hidden --out runs/hda/smoke --base-url http://HOST:PORT/v1
    python -m hda.grid --out runs/hda/main --base-url http://HOST:PORT/v1
    python -m hda.aggregate --runs runs/hda/main --out report/tables/hda

See HDA_RUNBOOK.md (project files, runbook/) for the experiment plan.
