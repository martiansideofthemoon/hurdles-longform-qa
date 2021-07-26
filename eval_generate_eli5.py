import argparse
import glob
import numpy as np
import random
import time
import pickle
import re
import string
import json
import collections as cll
from routing_tf_api_generation_eli5 import SparseTransformerWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--attention', default="clustering", type=str)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--total', default=1, type=int)
parser.add_argument('--max_seq_length', default=2816, type=int)
args = parser.parse_args()

random.seed(args.local_rank + 42)

valid_files = glob.glob("generations/final_guess_eli5_0.6_predicted_retrieval.jsonl")
valid_files.sort()

print(args)
all_data = []
for file in valid_files:
    with open(file, "r") as f:
        all_data.extend([json.loads(x) for x in f.read().strip().split("\n")])

sptf_model = SparseTransformerWrapper(
    local_attention=(args.attention == "local"),
    max_seq_length=args.max_seq_length
)

for qa_pair in all_data:
    question = qa_pair["input"]
    retrievals = [x["snippet"] for x in qa_pair["output"][0]["provenance"]]
    start = time.time()
    gen_outputs = sptf_model.forward([question], [retrievals])
    answer = sptf_model.encoder.decode(gen_outputs['generation'])
    answer = " ".join(answer[:answer.index("<EOS>")].split())
    total_time = time.time() - start
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Time Taken: {total_time}")

sptf_model.close()
