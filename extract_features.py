from data.char_table import CharacterTable
from sources.compiled import CompilationSource
from data.generators import AutoEncoderGenerator
from models.seq2seq import Seq2seqAutoencoder
from sources.preloaded import PreLoadedSource


charset = ''.join([chr(i) for i in range(32, 128)])
char_table = CharacterTable(charset)

source = CompilationSource('./compiled/data.json')


def fetch_strokes(source):
    cropped_strokes = []
    dummy_out = []
    for strokes, _ in source.get_sequences():
        for stroke in strokes:
            max_x = max([x for x, y in stroke.points])
            min_x = min([x for x, y in stroke.points])

            width = max_x - min_x

            max_y = max([y for x, y in stroke.points])
            min_y = min([y for x, y in stroke.points])
            height = max_y - min_y

            points = []
            for x, y in stroke.points:
                points.append((x - min_x, y - min_y))

            cropped_strokes.append(points)
            dummy_out.append('')

    return PreLoadedSource(cropped_strokes, dummy_out)


strokes_source = fetch_strokes(source)

from data.preprocessing import PreProcessor

train_gen = AutoEncoderGenerator(strokes_source,
                                 pre_processor=PreProcessor(), channels=2)

val_gen = train_gen
batch_size = 1
lrate = 0.001
validation_steps = 14
epochs = 100

auto_encoder = Seq2seqAutoencoder(input_channels=2, output_channels=2)
auto_encoder.fit_generator(
    lrate,
    train_gen,
    val_gen,
    train_gen.get_examples(batch_size=batch_size),
    steps_per_epoch=int(len(train_gen) / batch_size) + 1,
    validation_data=val_gen.get_examples(batch_size),
    validation_steps=validation_steps,
    epochs=epochs)
