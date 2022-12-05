import numpy as np


def softmax(x):
    return list(np.exp(x) / np.sum(np.exp(x), axis=0))


def evaluate_multi():
    # Below are examples of our strategyqa outputs.
    # Replace the paths with your own experiments.
    gold_lines = [x.strip() for x in open("reported_outputs/stqa/decomp.txt").readlines()]
    predicted_lines = [x.strip() for x in open("reported_outputs/stqa/eval_results_lm.txt").readlines()]
    predicted_probs = [(softmax([float(y) for y in x.strip().split("\t")])) for x in open("reported_outputs/stqa/eval_probs.txt").readlines()]
    gold_map = {}
    predicted_map = {}
    predicted_probs_map = {}
    gold_line_map = {}
    for i, line in enumerate(gold_lines):
        question = line.split("\t")[0].split(" </s> Decomposition: ")[0]
        if question not in gold_map:
            gold_map[question] = line.split("\t")[1].split()[0]
            predicted_map[question] = []
            predicted_probs_map[question] = []
            gold_line_map[question] = []
        predicted_map[question].append(predicted_lines[i])
        if predicted_lines[i] == "yes":
            predicted_probs_map[question].append(predicted_probs[i][0])
        else:
            predicted_probs_map[question].append(predicted_probs[i][1])
        gold_line_map[question].append([gold_lines[i].split("\t")[0].split(" </s> Decomposition: ")[1], predicted_lines[i]])
    total = 0
    correct = 0
    for key in gold_map:
        total += 1
        yes_count = 0.0
        no_count = 0.0
        for i, vote in enumerate(predicted_map[key]):
            assert vote in ["yes", "no"]
            if vote == "yes":
                yes_count += predicted_probs_map[key][i]
            else:
                no_count += predicted_probs_map[key][i]
        prediction = "yes"
        if no_count > yes_count:
            prediction = "no"
        if prediction == gold_map[key]:
            correct += 1

    print("Accuracy: {}".format(str(float(correct) / float(total))))


evaluate_multi()
