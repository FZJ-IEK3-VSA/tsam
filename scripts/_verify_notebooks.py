"""Quick verification of notebook structure after edits."""
import json
import pathlib

nb_dir = pathlib.Path(r"C:\Programming\tsam\docs\notebooks\how_it_works")

for nb_file in sorted(nb_dir.glob("*.ipynb")):
    raw = nb_file.read_text(encoding="utf-8")
    nb = json.loads(raw)
    cell_ids = [(c["cell_type"], c.get("id", "?")) for c in nb["cells"]]
    print(f"\n=== {nb_file.name} ({len(nb['cells'])} cells) ===")
    for ct, cid in cell_ids:
        print(f"  {ct[:4]:4s} | {cid}")
