# Hurdles to Progress in Long-form Question Answering

This repository contains the official scripts and datasets accompanying our NAACL 2021 paper, "[Hurdles to Progress in Long-form Question Answering](https://arxiv.org/abs/2103.06332)". This repository supports **inference from the pretrained retriever / generator** & includes **evaluation scripts**.

Specifically, this codebase contains the model checkpoints, inference scripts for the retriever / generator model, generated outputs from model using c-REALM retrievals and random retrievals, scripts to compute ROUGE-L / R-Prec scores using the generations, scripts for question paraphrase classification, scripts for computing ROUGE-L bounds. You can also find the original Routing Transformer model's codebase [here](https://github.com/google-research/google-research/tree/master/routing_transformer).

## Setup

```
python -m virtualenv lfqa-venv
source lfqa-venv/bin/activate
pip install transformers
pip install tensor2tensor
```

For GPU support, you might need to change the version of your TensorFlow depending on the CUDA / CuDNN installation ([details](https://www.tensorflow.org/install/source#gpu)). *GPU support is strongly recommended for faster inference*.

## Model Checkpoints & Generations

Routing Transformer finetuned on ELI5: [link](https://storage.googleapis.com/rt-checkpoint/eli5_checkpoint.zip)  
c-REALM TF Hub model + encoded retrieval corpora: [link](https://storage.googleapis.com/rt-checkpoint/retriever.zip)  
c-REALM tokenized KILT Wikipedia data: [link](https://storage.googleapis.com/rt-checkpoint/kilt_retrieval_train.zip)   
c-REALM tokenized ELI5 training data: [link](https://storage.googleapis.com/rt-checkpoint/eli5_retrieval_train.zip)  
Pre-computed generations & QQP classifier: [link](https://drive.google.com/drive/folders/1kBIo26SdjHJUKe7wYr2mh87sH0XNnUSJ?usp=sharing)

The original Routing Transformer model (pretrained on PG-19) and a local attention version of it can be found in the main repository ([link](https://github.com/google-research/google-research/tree/master/routing_transformer#pre-trained-pg-19-checkpoint-)).

## Generating from the Routing Transformer

*(We have provided the pre-computed retrievals from c-REALM on ELI5, so no need to run the c-REALM retriever)*

1. Download the "Routing Transformer finetuned on ELI5" model listed above and place it inside `models`.

```
wget https://storage.googleapis.com/rt-checkpoint/eli5_checkpoint.zip
unzip eli5_checkpoint.zip -d models
rm eli5_checkpoint.zip
```

2. Download the `generations` folder from the Google Drive link listed as "Pre-computed generations & QQP classifier" above.

3. Run [`eval_generate_eli5.py`](eval_generate_eli5.py) to generate from the model. We have provided `c-REALM` retrieval outputs in the script for the ELI5 validation / test split. For custom inputs, you will need to load the retriever and wikipedia corpus (see next section). Generation is on the slower side (~4 minutes per ELI5 QA pair on a 1080ti GPU), we hope to switch to the faster decoding mode in the Routing Transformer model in the near future.

## Retrievals from c-REALM

*(This script only tests the retriever, it doesn't depend on the Routing Transformer generator model)*

1. Download the "c-REALM TF Hub model + encoded retrieval corpora" model listed above. Place it inside the `models` folder.

```
wget https://storage.googleapis.com/rt-checkpoint/retriever.zip
unzip retriever.zip -d models
rm retriever.zip
```

2. Download "c-REALM tokenized KILT Wikipedia data" if you are interested in retrieving from the [KILT Wikipedia corpus](https://github.com/facebookresearch/KILT#kilt-knowledge-source) and/or "c-REALM tokenized ELI5 training data" if you are interested in retrieving question paraphrases from the ELI5 training set. Place them inside the `models` folder.

```
wget https://storage.googleapis.com/rt-checkpoint/eli5_retrieval_train.zip
unzip eli5_retrieval_train.zip -d models
rm eli5_retrieval_train.zip
```

3. Run [`eval_retriever_eli5.py`](eval_retriever_eli5.py) to retrieve using `c-REALM`. Modify the `--retrieval_corpus` flag to choose the retrieval corpus.

## Evaluation of Outputs

### Setup

1. Download the `generations` folder from the Google Drive [link](https://drive.google.com/drive/folders/1kBIo26SdjHJUKe7wYr2mh87sH0XNnUSJ?usp=sharing) into this root folder.

2. Clone the [KILT repository](https://github.com/facebookresearch/KILT) in this folder and run the installation in a virtual environment.

```
git clone https://github.com/facebookresearch/KILT
cd KILT
virtualenv -p python3.7 kilt-venv
pip install -r requirements.txt
pip install --editable .
```

3. If you are interested in using the Quora Question Paraphrase classifier (used in Section 3.2 of the paper), download the `roberta-large-finetuned-qqp` folder from "Pre-computed generations & QQP classifier" listed above. This model was built by [Tu Vu](https://people.cs.umass.edu/~tuvu/).

4. Download the ELI5 train, validation and test splits.

```
cd KILT
wget http://dl.fbaipublicfiles.com/KILT/eli5-train-kilt.jsonl -O train.jsonl
wget http://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl -O valid.jsonl
wget http://dl.fbaipublicfiles.com/KILT/eli5-test_without_answers-kilt.jsonl -O test.jsonl
```

### Running evaluations

Enter the `KILT` folder and run the following command for evaluating `p=0.6` with c-REALM retrievals on the validation set:

```
python kilt/eval_downstream.py ../generations/final_guess_eli5_0.6_predicted_retrieval.jsonl ../generations/final_gold_eli5_0.6_predicted_retrieval.jsonl
```

which should give you the output (partly reported in Table 6 of the paper),

```
{   'downstream': {   'accuracy': 0.0,
                      'em': 0.0,
                      'f1': 0.25566078582652935,
                      'rougel': 0.24417152125142375},
    'kilt': {   'KILT-accuracy': 0.0,
                'KILT-em': 0.0,
                'KILT-f1': 0.03414819887348917,
                'KILT-rougel': 0.03205580975169385},
    'retrieval': {'Rprec': 0.13258897418004187, 'recall@5': 0.2122586648057688}}
```

To evaluate other configurations, modify the paths in the command above. You can replace `0.6` with `0.9` for higher entropy generations, and replace `predicted` with `random` for randomly selected retrieval paragraphs (Hurdle #1 or Section 3.1 in the paper). Note that you should make this change for both the `guess` and `gold` files, to ensure correct alignment. We have only provided generations for the validation set since the test set answers / retrievals for ELI5 are hidden behind the [KILT leaderboard](https://eval.ai/web/challenges/challenge-page/689/leaderboard/1908).

### Question paraphrase classification using QQP Classifier

In Section 3.2 of our paper, we used a Quora Question Paraphrase classifier to find question paraphrases amoung similar questions retrieved by c-REALM. To run this, make sure you have downloaded the QQP checkpoint (step 3 in Setup) and run,

```
python run_qqp.py --input_file generations/final_guess_eli5_0.6_similar_questions.jsonl
```

You should get a score of 43.6%. Note that this is a lower-bound --- qualitatively we found this classifier missed several paraphrase pairs with low lexical overlap, or cases where the retrieved training set question will have a super-set of the information needed to answer the validation set question.

### Lower and Upper Bounds on ROUGE-L

Run the following to evaluate bounds on ROUGE-L. Make sure you have completed steps 1, 4 in the setup above. Scripts to evaluate other bounds involving training set retrieval coming soon!

```
cp generate_final_guess_bounds.py KILT/
cd KILT

# Copy input lowerbound, should get 20.0 ROUGE-L
python generate_final_guess_bounds.py --bound_type copy_input

# Random training set answer, should get 15.8-16.2 ROUGE-L depending on randomness
python generate_final_guess_bounds.py --bound_type random_train_ans

# "Performance" can be further boosted by randomly selecting from only longest answers
# for each training set question, up to ~16.7 ROUGE-L. This result is not reported in
# paper, but can be run using:
python generate_final_guess_bounds.py --bound_type random_train_ans_longest

# Longest gold answer upperbound, should get 21.2 ROUGE-L
python generate_final_guess_bounds.py --bound_type longest_gold

# Best gold answer upperbound, should get 26.2 ROUGE-L (takes a while to run, 45 min on single core)
python generate_final_guess_bounds.py --bound_type best_gold
```

## Human Annotations

In the Drive link above, we have also added the human annotations used to generate the results in Table 3 and Table 15, under `human_annotations/ab_testing`. You can see the folder [here](https://drive.google.com/drive/folders/119bkIGWCZnoUaTZlkhnGLluPWr33rwg3?usp=sharing). Each JSONL file corresponds to an experiment conducted in Table 15, which the folders contain the HTML files with answer pairs shown to annotators in a random order. The names of the files should be self-explanatory.

Each JSONL file contains the KILT-formatted ELI5 QA pair for validation instances for which we have annotations. Here are the description of various keys ---

1. `human_annotation_url` - The URL of the HTML file shown to the annotator, a copy of this file is also present in the same folder.
2. `id` - The KILT / ELI5 id of the instance.
3. `input` - The question asked to the model.
4. `Which generation answers the question better?` - (if present) the human annotation for this question, which will be a key in the same dictionary.
5. `Which answer is more coherent?` - (if present) the human annotation for this question, which will be a key in the same dictionary.
6. `Which answer is more factually correct + sensical?` - (if present) the human annotation for this question, which will be a key in the same dictionary.
7. Two keys corresponding to the A/B for the experiment, with the answers. For the `gold_vs_*` experiments, the longest gold answer was shown to users.

## Citation

If you found our paper or this repository useful, please cite:

```
@inproceedings{lfqa21,
author={Kalpesh Krishna and Aurko Roy and Mohit Iyyer},
Booktitle = {North American Association for Computational Linguistics},
Year = "2021",
Title={Hurdles to Progress in Long-form Question Answering},
}
```
