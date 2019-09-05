from data.char_table import CharacterTable
from sources.compiled import CompilationSource
from data.generators import AutoEncoderGenerator
from models.seq2seq import Seq2seqAutoencoder
from sources.preloaded import PreLoadedSource


charset = ''.join([chr(i) for i in range(32, 128)])
char_table = CharacterTable(charset)

source = CompilationSource('./compiled/data.json')


def fetch_strokes(source, num_strokes):
    cropped_strokes = []
    dummy_out = []
    for strokes, text in source.get_sequences():
        for stroke in strokes:
            if len(cropped_strokes) > num_strokes:
                return PreLoadedSource(cropped_strokes, dummy_out)
            max_x = max([x for x, y in stroke.points])
            min_x = min([x for x, y in stroke.points])

            max_y = max([y for x, y in stroke.points])
            min_y = min([y for x, y in stroke.points])

            width = max_x - min_x
            height = max_y - min_y

            max_side = max(width, height) + 0.00001
            if max_side < 1:
                continue

            points = [(0, 0)]
            for x, y in stroke.points:
                x = (x - min_x) / max_side
                y = (y - min_y) / max_side
                points.append((x, y))

            if len(points) > 20:
                points = points[::10]
            points.append((0, 0))

            cropped_strokes.append(points)
            dummy_out.append('')

    return PreLoadedSource(cropped_strokes, dummy_out)


strokes_source = fetch_strokes(source, 0)

from data.preprocessing import PreProcessor

train_gen = AutoEncoderGenerator(strokes_source,
                                 pre_processor=PreProcessor(), channels=2)

val_gen = train_gen
batch_size = 1
lrate = 0.001
validation_steps = 14
epochs = 1001

auto_encoder = Seq2seqAutoencoder(encoding_size=128, input_channels=2, output_channels=2)
auto_encoder.fit_generator(
    lrate,
    train_gen,
    val_gen,
    train_gen.get_examples(batch_size=batch_size),
    steps_per_epoch=int(len(train_gen) / batch_size) + 1,
    validation_data=val_gen.get_examples(batch_size),
    validation_steps=validation_steps,
    epochs=epochs)
