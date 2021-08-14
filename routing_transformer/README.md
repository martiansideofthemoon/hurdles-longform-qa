# Efficient content-based sparse attention with Routing Transformers

<img src = "image/routing_attention.png" alt="Routing attention">

Code-base accompanying the [paper](https://arxiv.org/abs/2003.05997) (to appear
in [TACL](https://transacl.org/index.php/tacl)).
See also the accompanying
[slides](https://drive.google.com/file/d/1maX-UQbtnVtxQqLmHvWVN6LNYtnBaTd9/view?usp=sharing)
for a quick overview.

## Table of Contents

-   [Updates](#updates)
-   [Pre-trained PG-19 Checkpoint](#pg19)
-   [Explanation of hyperparameters](#hparam)
    *   [Local Attention](#local)
    *   [Routing Attention](#routing)
-   [Samples](#samples)
    *   [PG-19 (sequence length 8k)](#pg19-samples)
    *   [Document Machine Translation (sequence length 4k)](#doc-mt)
-   [Acknowledgments](#ack)
-   [How to Cite](#cite)

## Updates <a name="updates"></a>

* Routing Transformer + [REALM](https://github.com/google-research/language/tree/master/language/realm)
  is now [SOTA](https://eval.ai/web/challenges/challenge-page/689/leaderboard/1908#leaderboardrank-1)
  on long form Question Answering (QA) on the
  [ELI5 data-set](https://github.com/facebookresearch/ELI5) on the Knowledge
  Intensive Language Tasks (KILT) [benchmark](https://github.com/facebookresearch/KILT)
  from [Facebook AI](https://ai.facebook.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/),
  with **significant improvements** in generation quality over [BART](https://arxiv.org/abs/1910.13461),
  [RAG](https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/),
  [T5](https://arxiv.org/abs/1910.10683)/[Mesh TF](https://arxiv.org/abs/1811.02084)
  , e.g. **+4.11, +5.78, +9.14 Rouge-L improvement** over T5/Mesh TF, BART + DPR
  and RAG respectively.

## Pre-trained PG-19 Checkpoint <a name="pg19"></a>

Model     | Hparams  | Context Length | Data-set | Vocab                                                                                     | Download
--------- |  ---------------------- | -------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------- | --------
`Local-base` | `pg19_local8k`         | 8192           | [PG-19](https://github.com/deepmind/pg19) | [vocab98K](https://storage.googleapis.com/rt-checkpoint/vocab.pg19_length8k.32768.subwords) | [checkpoint.zip](https://storage.googleapis.com/rt-checkpoint/pg19_local.zip)
`RT-base`    | `pg19_local_cluster8k` | 8192           | [PG-19](https://github.com/deepmind/pg19) | [vocab98K](https://storage.googleapis.com/rt-checkpoint/vocab.pg19_length8k.32768.subwords) | [checkpoint.zip](https://storage.googleapis.com/rt-checkpoint/checkpoint.zip)
`RT-base`    | `pg19_local_cluster8k` | 8192           | [ELI-5](https://github.com/facebookresearch/ELI5) | [vocab98K](https://storage.googleapis.com/rt-checkpoint/vocab.pg19_length8k.32768.subwords) | [checkpoint.zip](https://storage.googleapis.com/rt-checkpoint/eli5_checkpoint.zip)


## Explanation of hyperparameters <a name="hparam"></a>

### Local Attention <a name="local"></a>

*   `local_num_heads`: Number of local attention heads
*   `query_shape`: This represents the shape of the query block.
    *   For 1-d local attention with block size `b`, this would be `(b,)`
*   `memory_query_shape`: This represents the query shape of memory antecedent
    and is useful for encoder-decoder attention
    * This is usually set the same as `query_shape` by default
    * This is useful when inputs and targets are of different lengths
    * E.g., if inputs are of length `4096` and targets of length `8192`
    * Plausible setting:`query_shape = (256,)`, `memory_flange = (256,)` and
      `memory_query_shape = (128,)`
    * This is because with block size `256`, the targets will have `32` blocks
    * To match this in enc-dec attention, the inputs must have `32` blocks
    * This is why we set `memory_query_shape = (4096/32,) = (128,)`
*   `memory_flange`: This represents the overlap of the memory block
    * Example setting: `query_shape = (b,)` and `memory_flange = (m * b, )`
    * Masked: Each query block attends to `m` previous blocks
    * Unmasked: Each query block attends to `m` previous & `m` subsequent blocks
    * Setting this to `(0,)` means all the blocks are independent of each other
    * Setting to `(0,)` is used for full attention, or for axial attention
    * This must be a multiple of `query_shape` in every dimension
*   Example setting can be found in `sparse_transformer.py` under `pg19_local8k`

### Routing Attention <a name="routing"></a>

*   `sparsity_cluster_num_heads`: Number of routing attention heads
*   `sparsity_cluster_size`: Number of clusters
*   `sparsity_cluster_attention_window`: Average size of each cluster
*   `sparsity_skip_first`: Number of initial layers to skip routing attention
    *   `sparsity_skip_first = 0` would have routing attention in every layer
    *   `sparsity_skip_first` equalling total layers would have no routing
        attention
*   Example setting can be found in `sparse_transformer.py` under
    `pg19_local_cluster8k`

## Samples <a name="samples"></a>

### PG-19 (sequence length 8k) <a name="pg19-samples"></a>

#### Unconditional Samples <a name="unconditional"></a>

-   [sample1](samples/pg19_sample1.txt)
-   [sample2](samples/pg19_sample2.txt)
-   [sample3](samples/pg19_sample3.txt)
-   [sample4](samples/pg19_sample4.txt)
-   [sample5](samples/pg19_sample5.txt)
-   [sample6](samples/pg19_sample6.txt)
-   [sample7](samples/pg19_sample7.txt)
-   [sample8](samples/pg19_sample8.txt)
-   [sample9](samples/pg19_sample9.txt)
-   [sample10](samples/pg19_sample10.txt)

#### Conditional Samples <a name="conditional"></a>

-   [sample](samples/pg19_cond_sample.txt)

### Document Machine Translation (sequence length 4k) <a name="doc-mt"></a>

-   [sample](samples/doc_mt_sample.txt)

## Acknowledgments <a name="ack"></a>
The authors would like to thank Phillip Wang and
Aran Komatsuzaki for a [Pytorch implementation](https://github.com/lucidrains/routing-transformer)
of Routing Transformer. The authors would also like
to thank Yonghui Wu, Weikang Zhou and Dehao
Chen for helpful feedback in improving the implementation of this work.
The authors would also
like to thank anonymous reviewers and the Action
Editor Xavier Carreras of TACL for their constructive comments
which helped improve the exposition of this work.

## How to Cite <a name="cite"></a>

If you extend or use this work, please cite the
[paper](https://arxiv.org/abs/2003.05997) where it was introduced:

```
@article{roy2020efficient,
  title={Efficient content-based sparse attention with routing transformers},
  author={Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  journal={arXiv preprint arXiv:2003.05997},
  year={2020}
}
```
