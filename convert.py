import json

with open("第3章程式_ALL_IN_ONE (1).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("hw3_baseline.py", "w", encoding="utf-8") as f:
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            f.write("".join(cell["source"]) + "\n\n")
