def find_most_label(labels):
    n_label = {}
    for i in labels:
        if i in n_label.keys():
            n_label[i] += 1
        else:
            n_label[i] = 1

    most_label = None
    most_n = 0
    for k, v in n_label.items():
        if v > most_n:
            most_label = k

    return most_label
