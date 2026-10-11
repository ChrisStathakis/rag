"""A tiny sample corpus used to smoke-test extract + chunk + embed + query.

One file per supported format lives in `sample/`. Run:

    python pipeline.py ingest  --src sample --table sample --dry-run
    python pipeline.py ingest  --src sample --table sample --replace
    python pipeline.py embed   --table sample
    python pipeline.py query   "how does attention work" --table sample
"""
