import json
from pathlib import Path

outputs = Path("outputs")
for d in sorted(outputs.iterdir()):
    p = d / "progress.json"
    r = d / "results.json"
    if r.exists():
        with open(r) as f:
            res = json.load(f)
        epochs = res.get("epochs", 0)
        best_acc = res.get("best_test_accuracy", "?")
        print(f"{d.name}: epochs={epochs}, best_acc={best_acc}")
    else:
        print(f"{d.name}: NO results.json")
