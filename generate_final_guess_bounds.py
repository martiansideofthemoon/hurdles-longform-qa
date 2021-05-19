import argparse
import json
import subprocess
import tqdm
import random
import numpy as np

from kilt import eval_downstream

parser = argparse.ArgumentParser()
parser.add_argument('--gold_file', default="valid.jsonl")
parser.add_argument('--bound_type', default="longest_gold")
args = parser.parse_args()

with open(args.gold_file, "r") as f:
    gold_data = [json.loads(x) for x in f.read().strip().split("\n")]

with open("train.jsonl", "r") as f:
    train_data = [json.loads(x) for x in f.read().strip().split("\n")]
    all_train_answers = [ans["answer"] for x in train_data for ans in x["output"] if "answer" in ans]
    train_ans_len_sorted = [
        sorted([ans["answer"] for ans in x["output"] if "answer" in ans], key=lambda x: len(x.split()), reverse=True)
        for x in train_data
    ]
    longest_train_answers = [x[0] for x in train_ans_len_sorted]
    print(len(all_train_answers))

all_ids_in_guess = {}

num_overwrites = 0

final_gold = []
final_guess = []

all_scores = []

for i, inp_instance in tqdm.tqdm(enumerate(gold_data), total=len(gold_data)):
    if "output" in inp_instance:
        answers = [x for x in inp_instance["output"] if "answer" in x]
        if len(answers) <= 1:
            continue
    else:
        answers = []


    if args.bound_type == "longest_gold":
        answers.sort(key=lambda x: len(x["answer"].split()), reverse=True)
        best_answer = answers[0]
        other_answers = [x for x in answers if x["answer"] != best_answer["answer"]]
    elif args.bound_type == "best_gold":
        curr_scores = []
        for ans in answers:
            other_answers = [x["answer"] for x in answers if x["answer"] != ans["answer"]]
            assert len(other_answers) == len(answers) - 1
            rouge_score = eval_downstream._metric_max_over_ground_truths(eval_downstream._rougel_score, ans["answer"], other_answers)
            curr_scores.append(rouge_score)
        best_answer = answers[np.argmax(curr_scores)]
        other_answers = [x for x in answers if x["answer"] != best_answer["answer"]]

    elif args.bound_type =="copy_input":
        best_answer = {
            "answer": " ".join([inp_instance["input"] for _ in range(5)])
        }
        other_answers = answers

    elif args.bound_type == "random_train_ans":
        best_answer = {
            "answer": random.choice(all_train_answers)
        }
        other_answers = answers

    elif args.bound_type == "random_train_ans_longest":
        best_answer = {
            "answer": random.choice(longest_train_answers)
        }
        other_answers = answers

    final_gold.append({
        "id": inp_instance["id"],
        "input": inp_instance["input"],
        "output": other_answers
    })
    final_guess.append({
        "id": inp_instance["id"],
        "input": inp_instance["input"],
        "output": [best_answer]
    })

print("Final aligned pairs = {:d}".format(len(final_gold)))

with open("final_gold.jsonl", "w") as f:
    f.write("\n".join([json.dumps(x) for x in final_gold]) + "\n")

with open("final_guess.jsonl", "w") as f:
    f.write("\n".join([json.dumps(x) for x in final_guess]) + "\n")

output = subprocess.check_output("python kilt/eval_downstream.py final_guess.jsonl final_gold.jsonl", shell=True)
print(output.decode("utf-8"))
