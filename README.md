# Learning to Decompose (WIP)
Code, model and data for [Zhou et al. 2022: Learning to Decompose: Hypothetical Question Decomposition Based on Comparable Texts](https://arxiv.org/pdf/2210.16865.pdf).
We are currently working on making this package easier to use, any advice is welcomed. 

## Data
We provide the following data:
- `comparable_text_pretrain.txt.zip` [Google Drive](https://drive.google.com/file/d/1EI21HDzVl-ajqAUCKOq-PZvJbYRHLxXO/view?usp=share_link): Distant supervision data that we used for pre-train DecompT5 as described in Section 3.
- `data/decomposition_train.txt`: The decomposition supervision we used to train the decomposition model in DecompEntail (on top of DecompT5).
- `data/entailment_train.txt`: The entailment supervision we used to train the entailment model in DecompEntail (On top of T5-3b).
- `data/strategyqa/*`: StrategyQA train/dev/test splits we used for experiments.
- `data/hotpotqa/*`: HotpotQA binary questions we used for experiments.
- `data/overnight/*`: Overnight data used for experiments.
- `data/torque/*`: Torque data used for experiments.

## Models
We provide several trained model weights used in our paper, hosted on Huggingface hub. We randomly released one seed from multi-seed experiments.
- [CogComp/l2d](https://huggingface.co/CogComp/l2d): T5-large trained on `comparable_text_pretrain.txt`.
- [CogComp/l2d-decomp](https://huggingface.co/CogComp/l2d-decomp): DecompT5 trained on `data/decomposition_train.txt`, used in the DecompEntail pipeline.
- [CogComp/l2d-entail](https://huggingface.co/CogComp/l2d-entail): T5-3b trained on `data/entailment_train.txt`, used in the DecompEntail pipeline.

## Code
The code are divided into two separate packages, each using slightly different dependencies, as provided in corresponding `requirements.txt`.
### Sequence-to-sequence models
The `seq2seq` package can be used to reproduce DecompT5, and its related experiments in Section 5 of the paper.
It is also used to train and evaluate the entailment model used in DecompEntail. We provide a few use case examples as shell scripts.
- `seq2seq/train_decompose.sh`: Train CogComp/l2d-decomp
- `seq2seq/train_entailment.sh`: Train CogComp/l2d-entail
- `seq2seq/eval_entailment.sh`: Evaluate entailment model

In addition, we provide the generation and evaluation code for overnight and torque experiments in `seq2seq/gen_seq.py`
- To generate the top 10 candidates, use `gen_output()`.
- To evaluate the generated candidates, use `evaluate_top()`.
See code comments for more detail.

### DecompEntail pipeline
The DecompEntail pipeline can be run with the following steps:
1. Generate decompositions given raw questions. 
    * This can be done by `generate_decomposition()` in `decompose/gen_facts.py`. See comments for more detail.
2. Format generated decompositions into l2d-entail readable forms.
    * This can be done by `format_to_entailment_model()` in `decompose/gen_facts.py`.
3. Run l2d-entail to get entailment scores.
    * This can be done by `seq2seq/eval_entailment.sh` and replacing the input file with the output file from the previous step. 
    * If you run an aggregation with different seeds, concatenate the output files into one file and use as an input to the script.
4. Majority vote to derive final labels based on entailment scores.
    * The previous step will output two files `eval_probs.txt` and `eval_results_lm.txt`. Replace the path in `decompose/evaluator.py` and compute accuracy.

## Citation
See the following paper: 
```
@inproceedings{ZRYR22,
    author = {Ben Zhou, Kyle Richardson, Xiaodong Yu and Dan Roth},
    title = {Learning to Decompose: Hypothetical Question Decomposition Based on Comparable Texts},
    booktitle = {EMNLP},
    year = {2022},
}
```