import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import *

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (145, 1024),
}

def load_data_std(args):
    problems = json.load(open(os.path.join(args.data_root, 'train.json')))


    # 🧩 Handle dataset splits
    pid_splits = None
    split_file = None

# ScienceQA compatibility
    scienceqa_split = os.path.join(args.data_root, 'scienceqa/pid_splits.json')
    if os.path.exists(scienceqa_split):
        pid_splits = json.load(open(scienceqa_split))
        print("🔹 Loaded ScienceQA pid_splits.json")

# ChartQA fallback
    else:
        print("🔹 Using ChartQA-style split structure (train/val/test JSON files).")
        pid_splits = {
        "train": [i for i in range(0, len(problems))],
        "val": [],
        "test": []
        }

    
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    if pid_splits:
        train_qids = pid_splits.get(args.train_split, [])
        val_qids = pid_splits.get(args.val_split, [])
        test_qids = pid_splits.get(args.test_split, [])
    else:
        train_qids = list(range(len(problems)))
        val_qids, test_qids = [], []

    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    name_maps = None
    image_features = None

    return problems, train_qids, name_maps, image_features


def load_data_img(args):
    import glob
    import os, json

# ✅ Try to load ChartQA file depending on split
    if os.path.exists(os.path.join(args.data_root, 'train', 'train_human.json')):
        split_file = os.path.join(args.data_root, args.train_split, f"{args.train_split}_human.json")
    elif os.path.exists(os.path.join(args.data_root, 'train_human.json')):
        split_file = os.path.join(args.data_root, f"{args.train_split}_human.json")
    else:
        split_file = os.path.join(args.data_root, 'train.json')

    print(f"🔹 Loading dataset from {split_file}")
    problems = json.load(open(split_file))

    
    """
    Modified version for custom datasets like ChartQA.
    Loads a single train.json instead of ScienceQA's pid_splits.json.
    """
    # 1️⃣ ChartQA-style dataset
    manual_train_path = os.path.join(args.data_root, "train.json")
    if os.path.exists(manual_train_path):
        print(f"🔹 Detected custom dataset at: {manual_train_path}")
        with open(manual_train_path, "r") as f:
            problems = json.load(f)

        qids = list(range(len(problems)))
        name_maps = {}
        image_features = None  # skip multimodal features for now

        print(f"✅ Loaded {len(problems)} problems from custom dataset.")
        return problems, qids, name_maps, image_features

    # 2️⃣ ScienceQA fallback
    sciqa_path = os.path.join(args.data_root, "scienceqa/pid_splits.json")
    if os.path.exists(sciqa_path):
        pid_splits = json.load(open(sciqa_path))
        print("Loaded ScienceQA pid_splits.json")
        # ... keep the rest of the original code below
    else:
        raise FileNotFoundError(f"❌ No dataset found at {manual_train_path} or {sciqa_path}")


class ScienceQADatasetStd(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, args, test_le=None
    ):
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, idx):
        source_text = str(self.source_text[idx])
        target_text = str(self.target_text[idx])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [src_text],
            max_length=self.args.input_len,
            padding="max_length",       # <-- changed here
            truncation=True,
            return_tensors="pt"
        )

        target = self.tokenizer.batch_encode_plus(
            [tgt_text],
            max_length=self.args.output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }


class ScienceQADatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, name_maps, tokenizer, source_len, target_len, args, image_features, test_le=None, le_data=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.image_ids = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)
            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))
        super().__init__()
        self.problems = problems      # ✅ ADD THIS LINE
        self.qids = qids
        self.args = args
        self.le_data = le_data
    
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, idx):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[idx])
        target_text = str(self.target_text[idx])
        image_ids = self.image_ids[idx]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        # --- ADD THIS BEFORE THE tokenization step ---
        problem = self.problems[self.qids[idx]]

# For ScienceQA (original dataset)
        if "lecture" in problem:
            src_text = f"Question: {problem['question']} Hint: {problem.get('hint', '')} Lecture: {problem.get('lecture', '')}"
        else:
    # For ChartQA — simpler prompt
            src_text = f"Question: {problem.get('question', problem.get('query', ''))}"
            tgt_text = problem.get('answer', problem.get('label', ''))  

            if 'choices' in problem:
                src_text += " Choices: " + ", ".join(problem['choices'])

        if "solution" in problem:
            tgt_text = problem["solution"]
        else:
            tgt_text = problem.get("answer", problem.get("label", ""))


        source = self.tokenizer.batch_encode_plus(
            [src_text],
            max_length=self.args.input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target = self.tokenizer.batch_encode_plus(
            [tgt_text],
            max_length=self.args.output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        image_ids = torch.tensor(image_ids).squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
        }
