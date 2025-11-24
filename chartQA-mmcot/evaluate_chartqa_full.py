import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# ================= CONFIG =================
base_model = "declare-lab/flan-alpaca-base"
trained_model_dir = "experiments/chartqa_manual_cpu/rationale_declare-lab-flan-alpaca-base_vit_QCM-E_lr5e-05_bs0_op128_ep1"
test_file = "ChartQA/ChartQA Dataset/test/test_human.json"
output_file = "experiments/chartqa_manual_eval/chartqa_test_predictions_full.json"
max_length = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

print(f"🔹 Loading base model and tokenizer: {base_model}")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

# Load your trained weights manually if present
safetensors_path = os.path.join(trained_model_dir, "model.safetensors")
bin_path = os.path.join(trained_model_dir, "pytorch_model.bin")

if os.path.exists(safetensors_path):
    print(f"✅ Loading trained weights from {safetensors_path}")
    from safetensors.torch import load_file as safe_load
    model.load_state_dict(safe_load(safetensors_path), strict=False)
elif os.path.exists(bin_path):
    print(f"✅ Loading trained weights from {bin_path}")
    model.load_state_dict(torch.load(bin_path, map_location="cpu"), strict=False)
else:
    print("⚠️ WARNING: No model weights found — using base model only!")

model.to(device)
model.eval()

print(f"🔹 Loading test data from {test_file}")
with open(test_file, "r") as f:
    test_data = json.load(f)

results = []
print(f"🧠 Running inference on {len(test_data)} samples...\n")

for i, item in enumerate(test_data):
    img = item.get("imgname", "")
    question = item.get("query", "")
    gt_answer = item.get("label", "")
    gt_rationale = item.get("rationale", "N/A")  # ChartQA usually lacks rationale

    # Prompt designed for CoT reasoning
    prompt = f"Question: {question}\nAnswer with reasoning step-by-step, then give final answer clearly."

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            temperature=0.7,
            do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract reasoning and final answer from generated text
    if "Answer:" in decoded:
        pred_rationale = decoded.split("Answer:")[0].strip()
        pred_answer = decoded.split("Answer:")[-1].strip()
    else:
        pred_rationale = decoded
        pred_answer = decoded.strip()

    print(f"🧾 [{i+1}/{len(test_data)}]")
    print(f"📊 Image: {img}")
    print(f"❓ Question: {question}")
    print(f"🤔 Predicted rationale: {pred_rationale}")
    print(f"✅ Predicted answer: {pred_answer}")
    print(f"💡 Ground truth rationale: {gt_rationale}")
    print(f"🎯 Ground truth answer: {gt_answer}")
    print("-" * 90)

    results.append({
        "index": i + 1,
        "imgname": img,
        "question": question,
        "predicted_rationale": pred_rationale,
        "predicted_answer": pred_answer,
        "ground_truth_rationale": gt_rationale,
        "ground_truth_answer": gt_answer
    })

# Save everything neatly
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ All predictions saved → {output_file}")
