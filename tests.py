from data import RawIterator, RandomOrderIterator, CharacterTable
import numpy as np


it = RawIterator('datasets/iam_online_db')
for serrie, trans in it.get_lines():
    xs = []
    ys = []
    for i in range(0, len(serrie), 2):
        xs.append(serrie[i])

    for i in range(1, len(serrie), 2):
        ys.append(serrie[i])

    a = np.array(xs)
    d = a[1:] - a[:-1]

    print(np.std(xs), np.mean(xs))
    print(np.std(d), np.mean(d), np.sum(d))

    break
