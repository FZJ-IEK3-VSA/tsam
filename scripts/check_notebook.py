"""Check notebook cell counts and outputs."""
import sys
import nbformat

nb = nbformat.read(r"C:\Programming\tsam\docs\notebooks\how_algorithms_work.ipynb", as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == "code"]
cells_with_output = [c for c in code_cells if c.get("outputs")]
plotly_cells = [
    c for c in code_cells
    if any("plotly" in str(o) or "application/vnd" in str(o) for o in c.get("outputs", []))
]
print(f"Total cells: {len(nb.cells)}")
print(f"Code cells: {len(code_cells)}")
print(f"Code cells with output: {len(cells_with_output)}")
print(f"Cells with plotly output: {len(plotly_cells)}")

# Print any cells without outputs (that should have them)
print("\nCode cells WITHOUT outputs:")
for i, c in enumerate(code_cells):
    if not c.get("outputs"):
        src_preview = c.source[:80].replace("\n", " ")
        print(f"  cell {i}: {src_preview}")
