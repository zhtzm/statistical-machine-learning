def Gini(D):
    n_examples = len(D)
    labels = D[:, -1]
    label_n = {}

    for label in labels:
        if label not in label_n.keys():
            label_n[label] = 0
        label_n[label] += 1

    for key in label_n.keys():
        label_n[key] /= n_examples
        label_n[key] **= 2

    return 1 - sum(label_n.values())
