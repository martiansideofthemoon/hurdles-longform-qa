import argparse
import glob
import numpy as np
import random
import tqdm
import pickle
import re
import string
import json
import collections as cll
import tensorflow as tf
from tensor2tensor.data_generators import text_encoder
import tensorflow_hub as hub
from transformers import AutoTokenizer

VOCAB_PATH = "models/vocab.pg19_length8k.32768.subwords"

parser = argparse.ArgumentParser()
parser.add_argument('--retrieval_corpus', default="eli5_train", type=str, choices=["eli5_train", "kilt_wiki"])
parser.add_argument('--retriever_path', default="models/retriever", type=str)
parser.add_argument('--total', default=1, type=int)
parser.add_argument('--max_seq_length', default=2816, type=int)
args = parser.parse_args()

print("Loading retrieval index...")
if args.retrieval_corpus == "eli5_train":
    blocks_file = "models/eli5_retrieval_train/blocks_and_titles_pg19_dash_sep.tfr"
    encoded_path = "models/retriever/encoded_eli5_train_pg19_vocab_dash_sep/encoded.ckpt"
    encoded_weights = tf.train.load_variable(encoded_path, "block_emb")
else:
    blocks_file = "models/retrieval_train/blocks_and_titles_pg19_dash_sep.tfr"
    encoded_path = "models/retriever/encoded_kilt_wiki_pg19_vocab_dash_sep/encoded.ckpt"
    encoded_weights = tf.train.load_variable(encoded_path, "block_emb")

print("Loading retrieval corpus...")
pg19_vocab_encoder = text_encoder.SubwordTextEncoder(VOCAB_PATH)
dataset = tf.data.TFRecordDataset(blocks_file)
all_processed = []

for dd in tqdm.tqdm(dataset.as_numpy_iterator()):
    dd2 = tf.train.Example.FromString(dd)
    processed_list = dd2.features.feature['block_and_title'].int64_list.value
    decoded_str = pg19_vocab_encoder.decode(processed_list).split("---")[:2]
    all_processed.append(decoded_str)

print("Loading retriever...")
# encode_queries and encode_candidates are the same since the encoders are shared
retriever = hub.KerasLayer(args.retriever_path, signature="encode_candidates", signature_outputs_as_dict=True)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("Processing inputs...")
for index, question in enumerate(all_processed):
    # The following lines can be minibatched as well
    str_tokens = bert_tokenizer(question[0], truncation=True, padding="max_length",
                                max_length=288, return_tensors="tf")
    input_map = {
        "input_ids": str_tokens["input_ids"],
        "segment_ids": str_tokens["token_type_ids"],
        "input_mask": str_tokens["attention_mask"]
    }
    retrieved_emb = retriever(input_map)["default"]
    # Alternatively, use MIPS libraries like ScaNN / FAISS for faster top-k searching
    retrieval_scores = tf.matmul(retrieved_emb, tf.transpose(encoded_weights))
    top_retrievals = tf.math.top_k(retrieval_scores, k=8).indices.numpy()
    print(f"\nInput question = {question[0]}\n")
    for retr_num, retr_id in enumerate(top_retrievals[0]):
        retrieval = all_processed[retr_id]
        if retrieval[0].strip() == retrieval[1].strip():
            # title and text is the same, just print one of them
            print(f"Retrieval #{retr_num} = {retrieval[0]}")
        else:
            print(f"Retrieval #{retr_num}\nTitle = {retrieval[0]}\nText = {retrieval[1]}\n")
