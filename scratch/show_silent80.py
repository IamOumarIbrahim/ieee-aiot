import json

with open("scratch/all_models_silent80.json") as f:
    data = json.load(f)

for img, res in data.items():
    models_0 = res["0"]
    # Check what classes and boxes are present
    dets_str = []
    for m, dlist in models_0.items():
        dets_str.append(f"{m}: {[(l, round(c, 2), [int(x) for x in b]) for l, c, b in dlist]}")
    print(f"Frame: {img}")
    for s in dets_str:
        print(f"   {s}")
