from data.char_table import CharacterTable
from sources.preloaded import PreLoadedSource
from sources.iam_online import RawIterator, RandomOrderIterator
import numpy as np


it = RandomOrderIterator('datasets/iam_online_db')
hwr = []
t = []
for serrie, trans in it.get_sequences():
    hwr.append(serrie)
    t.append(trans)
    if len(hwr) > 64:
        break

preloaded = PreLoadedSource(hwr, t)
for i in range(1):
    for serrie, trans in preloaded.get_sequences():
        print(trans)
    print()
