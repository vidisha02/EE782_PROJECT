import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# ================= CONFIG =================
base_model = "declare-lab/flan-alpaca-base"
trained_model_dir = "experiments/chartqa_manual_cpu/rationale_declare-lab-flan-alpaca-base_vit_QCM-E_lr5e-05_bs0_op128_ep1"
test_file = "ChartQA/ChartQA Dataset/test/test_human.json"
output_file = "experiments/chartqa_manual_eval/chartqa_test_predictions_detailed.json"
max_length = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

print(f"🔹 Loading tokenizer/model from {base_model}")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

# Try loading fine-tuned weights
for fname in ["model.safetensors", "pytorch_model.bin"]:
    fpath = os.path.join(trained_model_dir, fname)
    if os.path.exists(fpath):
        print(f"✅ Loading fine-tuned weights from {fpath}")
        if fpath.endswith(".safetensors"):
            from safetensors.torch import load_file
            model.load_state_dict(load_file(fpath), strict=False)
        else:
            model.load_state_dict(torch.load(fpath, map_location="cpu"), strict=False)
        break
else:
    print("⚠️ No fine-tuned weights found, using base model.")

model.to(device)
model.eval()

print(f"🔹 Loading test data from {test_file}")
test_data = json.load(open(test_file))

results = []

for i, item in enumerate(test_data):
    img = item.get("imgname", "")
    question = item.get("query", item.get("question", ""))
    gt_lecture = item.get("lecture", "N/A")
    gt_solution = item.get("solution", "N/A")
    gt_rationale = item.get("rationale", "N/A")
    gt_answer = item.get("answer", item.get("label", "N/A"))

    prompt = (
        f"Question: {question}\n"
        f"Explain your reasoning step-by-step, then clearly give the final answer."
    )

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
    if "Answer:" in decoded:
        pred_rationale = decoded.split("Answer:")[0].strip()
        pred_answer = decoded.split("Answer:")[-1].strip()
    else:
        pred_rationale = decoded
        pred_answer = decoded.strip()

    print(f"\n🧾 [{i+1}/{len(test_data)}]")
    print(f"📊 Image: {img}")
    print(f"❓ Question: {question}")
    print(f"🤔 Predicted rationale: {pred_rationale}")
    print(f"✅ Predicted answer: {pred_answer}")
    print(f"💡 GT Lecture: {gt_lecture}")
    print(f"💡 GT Solution: {gt_solution}")
    print(f"💡 GT Rationale: {gt_rationale}")
    print(f"🎯 GT Answer: {gt_answer}")
    print("-" * 100)

    results.append({
        "index": i + 1,
        "imgname": img,
        "question": question,
        "predicted_rationale": pred_rationale,
        "predicted_answer": pred_answer,
        "gt_lecture": gt_lecture,
        "gt_solution": gt_solution,
        "gt_rationale": gt_rationale,
        "gt_answer": gt_answer
    })

os.makedirs(os.path.dirname(output_file), exist_ok=True)
json.dump(results, open(output_file, "w"), indent=2)
print(f"\n✅ Results saved → {output_file}")

