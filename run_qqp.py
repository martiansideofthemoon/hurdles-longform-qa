import argparse
import csv
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--input_file", default="generations/final_guess_eli5_0.6_similar_questions.jsonl", type=str,
                    help="Path to dataset containing retrieved question pairs.")
parser.add_argument("--model_name_or_path", default="roberta-large-finetuned-qqp", type=str,
                    help="Path to pre-trained model or shortcut name.")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Batch size.")
args = parser.parse_args()

with open(args.input_file, "r") as f:
    guess_predicted = [json.loads(x) for x in f.read().strip().split("\n")]

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
model.cuda()
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

all_preds = []
no_overlap = []
for i, guess in tqdm(enumerate(guess_predicted)):

    input_sent = guess["input"]
    train_set_questions = [x['snippet'] for x in guess['output'][0]['provenance']]

    batch = [(input_sent, x) for x in train_set_questions]

    inputs = tokenizer.batch_encode_plus(
        batch,
        add_special_tokens=True,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    inputs["input_ids"] = inputs["input_ids"].cuda()
    inputs["attention_mask"] = inputs["attention_mask"].cuda()
    logits = model(**inputs)[0]
    preds = torch.argmax(logits, dim=1).tolist()
    probs = torch.softmax(logits, dim=1).tolist()
    all_preds.append(preds)
    if max(preds) < 0.5:
        no_overlap.append(guess)

    if i % 100 == 0:
        print("Matching questions = {:d} / {:d}".format(sum([max(x) for x in all_preds]), len(all_preds)))

print("Matching questions = {:d} / {:d}".format(sum([max(x) for x in all_preds]), len(all_preds)))

# with open("no_overlap_set.jsonl", "w") as f:
#     f.write("\n".join([json.dumps(x) for x in no_overlap]) + "\n")
