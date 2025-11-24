import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Path to your experiment log file
LOG_PATH = "experiments/chartqa_manual_cpu/log.txt"

# --- 1️⃣ Extract predicted and GT answers from log ---
def extract_answers(path):
    preds, gts = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("✅ Predicted answer:"):
                preds.append(line.split("✅ Predicted answer:")[-1].strip())
            elif line.startswith("🎯 Ground truth answer:"):
                gts.append(line.split("🎯 Ground truth answer:")[-1].strip())
            elif line.startswith("🎯 GT Answer:"):   # 👈 ADD this line for your log format
                gts.append(line.split("🎯 GT Answer:")[-1].strip())
    return preds, gts


# --- 2️⃣ Numeric comparison helper ---
def extract_number(s):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return float(nums[0]) if nums else None

def numeric_match(pred, gt, tol=1e-2):
    p, g = extract_number(pred), extract_number(gt)
    if p is None or g is None:
        return False
    return abs(p - g) < tol

# --- 3️⃣ Semantic similarity model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(pred, gt):
    emb_pred = model.encode(pred, convert_to_tensor=True)
    emb_gt = model.encode(gt, convert_to_tensor=True)
    score = util.cos_sim(emb_pred, emb_gt).item()
    return score

# --- 4️⃣ Evaluation ---
def evaluate(preds, gts):
    exact, numeric, sim_scores = [], [], []
    reasons = []

    for i, (p, g) in enumerate(zip(preds, gts)):
        em = int(p.strip().lower() == g.strip().lower())
        exact.append(em)

        nm = int(numeric_match(p, g))
        numeric.append(nm)

        sim = semantic_similarity(p, g)
        sim_scores.append(sim)

        reason = None
        if em == 0 and nm == 0 and sim < 0.5:
            if extract_number(p) and not extract_number(g):
                reason = "Predicted numeric value but ground truth is textual"
            elif not extract_number(p) and extract_number(g):
                reason = "Ground truth numeric but prediction is textual"
            elif len(p.split()) > 6:
                reason = "Prediction is too verbose or repetitive"
            else:
                reason = "Semantic drift — model guessed unrelated concept"
        reasons.append(reason)

    return {
        "exact_acc": np.mean(exact),
        "numeric_acc": np.mean(numeric),
        "mean_similarity": np.mean(sim_scores),
        "reasons": [r for r in reasons if r],
    }

# --- 5️⃣ Run evaluation ---
if __name__ == "__main__":
    preds, gts = extract_answers(LOG_PATH)
    print(f"Loaded {len(preds)} predictions and {len(gts)} ground truths")

    results = evaluate(preds, gts)

    print("\n📊 Evaluation Results:")
    print(f"  ✅ Exact Match Accuracy: {results['exact_acc']*100:.2f}%")
    print(f"  🔢 Numeric Accuracy:     {results['numeric_acc']*100:.2f}%")
    print(f"  💬 Mean Semantic Sim:    {results['mean_similarity']:.3f}")

    print("\n⚠️  Common Failure Reasons:")
    for r in set(results["reasons"]):
        print(f"  - {r}")
