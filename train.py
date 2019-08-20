from data import DataFactory
from models import SequenceToSequenceTrainer


class CharacterTable:
    def __init__(self, corpus):
        charset = set(corpus + self.sentinel + self.start)
        self._char_to_index = dict((ch, index) for index, ch in enumerate(charset))
        self._index_to_char = dict((v, k) for k, v in self._char_to_index.items())

    def save(self, path):
        import json
        d = {
            'char_to_index': self._char_to_index,
            'index_to_char': self._index_to_char
        }
        json_str = json.dumps(d)
        with open(path, 'w') as f:
            f.write(json_str)

    def load(self, path):
        import json
        with open(path, 'r') as f:
            s = f.read()

        d = json.loads(s)
        self._char_to_index = d['char_to_index']
        self._index_to_char = d['index_to_char']

        d = {}
        for k, v in self._index_to_char.items():
            d[int(k)] = v

        self._index_to_char = d

    def is_sentinel(self, ch):
        return self.sentinel == ch

    @property
    def sentinel(self):
        return '\n'

    @property
    def start(self):
        return '_'

    def encode(self, ch):
        return self._char_to_index[ch]

    def decode(self, index):
        if index in self._index_to_char:
            return self._index_to_char[index]
        else:
            raise Exception('Unknown {}'.format(index))
            return '?'

    def __len__(self):
        return len(self._char_to_index)


charset = ''.join([chr(i) for i in range(32, 128)])
char_table = CharacterTable(charset)

max_examples = 1500
batch_size = 64
epochs = 30

factory = DataFactory('datasets/iam_online_db', char_table,
                      num_examples=max_examples)

train_gen = factory.training_generator()
val_gen = factory.validation_generator()

trainer = SequenceToSequenceTrainer(decimal_table=char_table)

print(int(len(train_gen) / batch_size))
print(len(val_gen))
trainer.fit_generator(
    train_gen.get_examples(batch_size=batch_size),
    steps_per_epoch=int(len(train_gen) / batch_size),
    validation_data=val_gen.get_examples(batch_size),
    validation_steps=1,
    epochs=epochs,
)
