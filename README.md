# learning-to-decompose (WIP)
Code and Data for Zhou et al. 2022: Learning to Decompose: Hypothetical Question Decomposition Based on Comparable Texts

## Data
We provide the following data:
- `comparable_text_pretrain.txt.zip` [Google Drive](https://drive.google.com/file/d/1EI21HDzVl-ajqAUCKOq-PZvJbYRHLxXO/view?usp=share_link): Distant supervision data that we used for pre-train DecompT5 as described in Section 3.
- `data/decomposition_train.txt`: The decomposition supervision we used to train the decomposition model in DecompEntail (on top of DecompT5).
- `data/entailment_train.txt`: The entailment supervision we used to train the entailment model in DecompEntail (On top of T5-3b).
- `data/strategyqa/*`: StrategyQA train/dev/test splits we used for experiments.
- `data/hotpotqa/*`: HotpotQA binary questions we used for experiments.
- `data/overnight/*`:
- `data/torque/*`


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