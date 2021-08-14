import os
import pdb

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')
from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
from tensor2tensor.data_generators import text_encoder

from tensor2tensor.data_generators import problem

from routing_transformer.problems import pg19
from tqdm import tqdm

from routing_transformer.sparse_transformer import SparseTransformer
import routing_transformer.utils as utils

import numpy as np
import time
import random


VOCAB_PATH = "models/vocab.pg19_length8k.32768.subwords"
HPARAMS_PATH = "models/hparams.json"
CKPT_PATH = "models/eli5_checkpoint/model.ckpt-100000"
MAX_SEQUENCE_LENGTH = 8192


class SparseTransformerWrapper(object):
    def __init__(self, chunk_length=288, num_prefix=2304, max_seq_length=8192, top_p=0.9, local_attention=False, load_model=True):
        # Load hyperparameters
        self.num_prefix = num_prefix
        self.chunk_length = chunk_length
        self.max_seq_length = max_seq_length or MAX_SEQUENCE_LENGTH
        # Needed since RT uses blocks of size 256
        assert self.max_seq_length % 256 == 0

        hparams = hparams_lib.create_hparams_from_json(HPARAMS_PATH)
        hparams.use_tpu = False
        hparams = zero_dropout(hparams)
        # Build TF1 graph of model
        sptf_model = SparseTransformer(hparams, tf.estimator.ModeKeys.PREDICT)
        sptf_model.hparams.sampling_keep_top_k = 0
        sptf_model.hparams.nucleus_sampling = top_p
        sptf_model.hparams.fast_decode = False
        sptf_model._decode_hparams.batch_size = 1
        sptf_model.hparams.num_decode_cores = 1
        sptf_model.hparams.batch_size = 1
        sptf_model.hparams.max_target_length = max_seq_length
        if local_attention:
            print("Using local attention...")
            # Fast decoding works with local attention, but skipping it for fair comparison with routing attention
            # Uncomment it for faster decoding
            # sptf_model.hparams.fast_decode = True
            sptf_model.hparams.sparsity_skip_first = 22
        else:
            print("Using clustering attention...")

        self.encoder = text_encoder.SubwordTextEncoder(VOCAB_PATH)
        if load_model:
            self.input_nodes = {
                "targets": tf.placeholder(tf.int32, [1, num_prefix])
            }
            # self.output_nodes = sptf_model.infer(self.input_nodes)
            self.output_nodes = infer_with_prefix(sptf_model, self.input_nodes)
            # Map the checkpoint variables to the graph
            init_from_checkpoint(CKPT_PATH, checkpoint_strip_str="/body/")
            # create a session object, and actually initialize the graph
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())


    def tokenize_and_pad(self, seq):
        encoded = self.encoder.encode(seq.strip())
        encoded = encoded[:self.chunk_length]
        # pad shorter sequences to the full length
        encoded = encoded + [text_encoder.PAD_ID for _ in range(self.chunk_length - len(encoded))]
        assert len(encoded) == self.chunk_length
        return encoded

    def forward(self, questions, retrievals):
        encoded_prefixes = []

        for ques, rets in zip(questions, retrievals):
            if "---" not in ques:
                ques += "---"
            encoded_ques = self.tokenize_and_pad(ques)
            encoded_rets = [self.tokenize_and_pad(x) for x in rets]
            encoded_rets = encoded_rets[::-1]
            prefix = [x for y in encoded_rets for x in y] + encoded_ques
            encoded_prefixes.append(prefix)

        feed_dict = {
            self.input_nodes["targets"]: np.array(encoded_prefixes)
        }
        outputs = self.sess.run(self.output_nodes, feed_dict=feed_dict)
        outputs = outputs["outputs"].squeeze()
        return_outputs = {
            "prefix": encoded_prefixes[0],
            "model_output_prefix": outputs[:self.num_prefix],
            "generation": outputs[self.num_prefix:],
        }

        return return_outputs

    def close(self):
        self.sess.close()


def infer_with_prefix(sparse_transformer, features):
    """Modified from SparseTransformer.infer to support prefixes."""

    sptf = sparse_transformer

    with tf.variable_scope("sparse_transformer", reuse=tf.AUTO_REUSE):
        features = sptf.bottom(features)
    batch_size, seq_len = features["targets"].shape
    assert sptf.batch_size == batch_size
    decode_length = sptf.hparams.max_target_length
    cache = {}
    decoding_stats = {}
    targets_old = features.get("targets")
    start_step = int(seq_len)

    reshaped_targets = tf.reshape(features["targets"],
                                  [batch_size, seq_len, 1, 1])
    initial_output = tf.zeros((batch_size, decode_length - seq_len, 1, 1),
                              dtype=tf.int32)
    initial_output = tf.concat([reshaped_targets, initial_output], axis=1)
    initial_logits = tf.zeros((batch_size, decode_length, sptf.vocab_size))

    # call body once to initialize cache with representations of input frames.
    features["targets"] = initial_output
    # Set shape of inputs
    if "inputs" in features:
        features["inputs"].set_shape([batch_size,
                                      sptf.hparams.max_length,
                                      1,
                                      sptf.hparams.hidden_size])

    with tf.variable_scope("sparse_transformer", reuse=tf.AUTO_REUSE):
        sptf.body(
            features,
            decode_step=None,
            cache=cache,
            decoding_stats=decoding_stats
        )

    def infer_step(i, recent_output, recent_logits, cache, decoding_stats):
        """Inference step."""
        features_copy = features.copy()
        features_copy["targets"] = recent_output
        cur_sample, cur_logit = sample_step(
            sparse_transformer,
            features_copy,
            decode_step=i,
            cache=cache,
            decoding_stats=decoding_stats
        )
        pos = i
        samples = recent_output + tf.scatter_nd(
            indices=[[b, pos, 0, 0] for b in range(batch_size)],
            updates=cur_sample,
            shape=utils.shape_list(recent_output)
        )
        logits = recent_logits + tf.scatter_nd(
            indices=[[b, pos] for b in range(batch_size)],
            updates=cur_logit,
            shape=utils.shape_list(recent_logits)
        )
        return i + 1, samples, logits, cache, decoding_stats

    def while_exit_cond(i, result, logits, cache, decoding_stats):  # pylint: disable=unused-argument
        """Exit the loop if it reaches decode_length."""
        not_overflow = i < decode_length
        return not_overflow

    _, final_result, final_logits, _, decoding_stats = tf.while_loop(
        while_exit_cond,
        infer_step,
        [start_step, initial_output, initial_logits, cache, decoding_stats],
        back_prop=False,
        parallel_iterations=1)

    original_shape = [decode_length]

    blocks_per_dim = [
        s // q for s, q in zip(original_shape, sptf.hparams.query_shape)
    ]
    final_result_shape = utils.shape_list(final_result)
    final_result = tf.reshape(
        final_result,
        [final_result_shape[0], -1,
        np.prod(sptf.hparams.query_shape), 1])
    final_logits_shape = utils.shape_list(final_logits)
    final_logits = tf.reshape(final_logits, [
        final_logits_shape[0], -1,
        np.prod(sptf.hparams.query_shape), final_logits_shape[-1]
    ])
    final_result = utils.unflatten_blocks_nd(final_result, blocks_per_dim)
    final_result = utils.put_back_blocks_nd(final_result,
                                            sptf.hparams.query_shape)
    final_logits = utils.unflatten_blocks_nd(final_logits, blocks_per_dim)
    final_logits = utils.put_back_blocks_nd(final_logits,
                                            sptf.hparams.query_shape)
    # Reassign targets back to the previous value.
    if targets_old is not None:
        features["targets"] = targets_old

    return {
        "outputs": final_result,
        "logits": final_logits
    }


def sample_step(sparse_transformer, features, decode_step, cache,
                  decoding_stats):
    """Sample step for infer, adapted from SparseTransformer.sample."""

    sptf = sparse_transformer

    with tf.variable_scope("sparse_transformer", reuse=tf.AUTO_REUSE):
        logits = sptf.body(features, decode_step, cache, decoding_stats)
        if not sptf.hparams.fast_decode:
            logits = tf.gather(logits, decode_step, axis=1)
        logits = tf.reshape(logits, [sptf.batch_size, sptf.vocab_size])

        # Should not use top_k and top_p together
        assert (sptf.hparams.sampling_keep_top_k *
               (1 - sptf.hparams.nucleus_sampling) == 0)
        if sptf.hparams.sampling_keep_top_k:
            tf.logging.info("Top-k sampling top_k = {}".format(sptf.hparams.sampling_keep_top_k))
            values, _ = tf.math.top_k(logits, k=sptf.hparams.sampling_keep_top_k)
            k_largest = tf.reduce_min(values)
            logits = tf.where(tf.less_equal(logits, k_largest),
                              tf.ones_like(logits)*-1e9, logits)
        if sptf.hparams.nucleus_sampling < 1:
            logits = sptf.nucleus_sampling(logits)
        sample = sptf.multinomial_squeeze(logits, sptf.hparams.sampling_temp)
        sample = tf.reshape(sample, [sptf.batch_size])
        return sample, logits



def find_sub_list(sl, l):
    sll=len(sl)
    matches = []
    for ind in (i for i,e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            matches.append(
                (ind, ind + sll)
            )
    if matches:
        return matches


def zero_dropout(hparams):
    hparams.input_dropout = 0.0
    hparams.dropout = 0.0
    hparams.relu_dropout = 0.0
    hparams.attention_dropout = 0.0
    hparams.layer_prepostprocess_dropout = 0.0
    return hparams


def log_variables(name, var_names):
    print_var_names = [
        x for x in var_names
        if "adam" not in x.lower() and "retriever/module" not in x.lower()
    ]
    tf.logging.info("%s (%d total): %s", name, len(print_var_names),
                    random.sample(print_var_names, min(len(print_var_names), 5)))


def init_from_checkpoint(checkpoint_path,
                         checkpoint_prefix=None,
                         variable_prefix=None,
                         target_variables=None,
                         checkpoint_strip_str=None):
    """Initializes all of the variables using `init_checkpoint."""
    tf.logging.info("Loading variables from %s", checkpoint_path)
    checkpoint_variables = {
        name: name for name, _ in tf.train.list_variables(checkpoint_path) if "Adafactor" not in name
    }
    if target_variables is None:
        target_variables = tf.trainable_variables()
    target_variables = {var.name.split(":")[0]: var for var in target_variables}

    if checkpoint_strip_str is not None:
        checkpoint_variables = {
            name.replace(checkpoint_strip_str, "/"): varname
            for name, varname in checkpoint_variables.items()
        }
    elif checkpoint_prefix is not None:
        checkpoint_variables = {
            checkpoint_prefix + "/" + name: varname
            for name, varname in checkpoint_variables.items()
        }
    if variable_prefix is not None:
        target_variables = {
            variable_prefix + "/" + name: var
            for name, var in target_variables.items()
        }

    checkpoint_var_names = set(checkpoint_variables.keys())
    target_var_names = set(target_variables.keys())
    intersected_var_names = target_var_names & checkpoint_var_names

    assignment_map = {
        checkpoint_variables[name]: target_variables[name]
        for name in intersected_var_names
    }
    tf.train.init_from_checkpoint(checkpoint_path, assignment_map)

    log_variables("Loaded variables", intersected_var_names)
    log_variables("Uninitialized variables", target_var_names - checkpoint_var_names)
    log_variables("Unused variables", checkpoint_var_names - target_var_names)
