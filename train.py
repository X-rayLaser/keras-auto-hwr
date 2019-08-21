from data import DataFactory, CharacterTable
from models import SequenceToSequenceTrainer

charset = ''.join([chr(i) for i in range(32, 128)])
char_table = CharacterTable(charset)

max_examples = 8
batch_size = 8
epochs = 125

factory = DataFactory('datasets/iam_online_db', char_table,
                      num_examples=max_examples)

train_gen = factory.training_generator()
val_gen = factory.validation_generator()

trainer = SequenceToSequenceTrainer(char_table=char_table)

print(int(len(train_gen) / batch_size))
print(len(val_gen))
trainer.fit_generator(
    train_gen.get_examples(batch_size=batch_size),
    steps_per_epoch=int(len(train_gen) / batch_size) + 1,
    validation_data=val_gen.get_examples(batch_size),
    validation_steps=1,
    epochs=epochs,
)


from estimate import CharacterErrorRate
estimator = CharacterErrorRate(trainer.get_inference_model(), num_trials=8)
error_rate = estimator.estimate(train_gen)
print(error_rate)
