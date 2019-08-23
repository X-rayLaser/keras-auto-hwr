from data import RawIterator, RandomOrderIterator, CharacterTable, PreLoadedIterator
import numpy as np


it = RandomOrderIterator('datasets/iam_online_db')
hwr = []
t = []
for serrie, trans in it.get_lines():
    hwr.append(serrie)
    t.append(trans)
    if len(hwr) > 64:
        break

preloaded = PreLoadedIterator(hwr, t)
for i in range(1):
    for serrie, trans in preloaded.get_lines():
        print(trans)
    print()
