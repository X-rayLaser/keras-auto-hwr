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

    print(xs)
    print(ys)
    print(len(serrie))

    print(np.mean(ys))
    print(np.std(ys))

    break
