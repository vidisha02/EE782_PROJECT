import os
import json
import pandas as pd

# paths
base_dir = "/Users/parvmaheshwari/Desktop/AdvML/mm-cot/ChartQA/ChartQA Dataset/train"
csv_dir = os.path.join(base_dir, "tables")
img_dir = os.path.join(base_dir, "png")
output_path = "data/chartqa_manual/train.json"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

dataset = []
skipped = 0

for csv_file in os.listdir(csv_dir):
    if not csv_file.endswith(".csv"):
        continue

    chart_id = os.path.splitext(csv_file)[0]
    csv_path = os.path.join(csv_dir, csv_file)
    img_path = os.path.join(img_dir, f"{chart_id}.png")

    try:
        df = pd.read_csv(csv_path)

        # Skip if file is empty or malformed
        if df.empty or len(df.columns) < 2:
            skipped += 1
            continue

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        x_col = df.columns[0]
        y_cols = df.columns[1:]

        # Drop rows with NaN in x_col
        df = df.dropna(subset=[x_col])

        for y_col in y_cols:
            # Skip if column is all NaN or non-numeric
            if df[y_col].dropna().empty:
                skipped += 1
                continue

            # Try converting to numeric (ignore non-numeric junk)
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            if df[y_col].dropna().empty:
                skipped += 1
                continue

            # Find the max value safely
            valid_rows = df.dropna(subset=[y_col])
            if valid_rows.empty:
                skipped += 1
                continue

            max_idx = valid_rows[y_col].idxmax()
            answer = str(valid_rows.loc[max_idx, x_col])

            # Build question
            question = f"In which {x_col.lower()} did '{y_col}' have the highest value?"
            choices = valid_rows[x_col].astype(str).tolist()

            sample = {
                "id": f"chartqa_train_{chart_id}_{y_col}",
                "question": question,
                "choices": choices,
                "answer": answer,
                "image": img_path,
                "hint": ""
            }
            dataset.append(sample)

    except Exception as e:
        print(f"⚠️ Skipped {csv_file}: {e}")
        skipped += 1
        continue

print(f"\n✅ Created {len(dataset)} QA samples from {len(os.listdir(csv_dir))} charts.")
print(f"⚠️ Skipped {skipped} problematic files/columns.")

with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"📁 Saved to {output_path}")
