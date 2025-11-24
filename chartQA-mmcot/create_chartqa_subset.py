import json, os

# Create output folder
os.makedirs("data/chartqa_small", exist_ok=True)

# Load ChartQA training data
with open("ChartQA/train.json") as f:
    data = json.load(f)

# Take only the first 20 samples
mini = data[:20]

# Adjust image paths (if needed)
for x in mini:
    if "imgname" in x:
        x["image"] = os.path.join("ChartQA/train/png", x["imgname"])

# Save the mini subset
with open("data/chartqa_small/train.json", "w") as f:
    json.dump(mini, f, indent=2)

print("✅ Created small ChartQA subset at data/chartqa_small/train.json")
