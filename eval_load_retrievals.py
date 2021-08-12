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


parser = argparse.ArgumentParser()
parser.add_argument('--blocks_file', default="models/eli5_retrieval_train/blocks_and_titles_pg19_dash_sep.tfr", type=str)
parser.add_argument('--total', default=1, type=int)
parser.add_argument('--max_seq_length', default=2816, type=int)
args = parser.parse_args()

VOCAB_PATH = "/mnt/nfs/work1/miyyer/simengsun/in-book-retrieval/RT-data/vocab.pg19_length8k.32768.subwords"
encoder = text_encoder.SubwordTextEncoder(VOCAB_PATH)
dataset = tf.data.TFRecordDataset(args.blocks_file)
all_processed = []

for dd in tqdm.tqdm(dataset.as_numpy_iterator()):
    dd2 = tf.train.Example.FromString(dd)
    processed_list = dd2.features.feature['block_and_title'].int64_list.value
    all_processed.append(encoder.decode(processed_list))
