from sources.iam_online import OnlineSource, LinesSource
from sources.preloaded import PreLoadedSource
import json


def load_data(source, num_lines):
    hand_writings = []
    transcriptions = []
    for strokes_list, transcription in source.get_sequences():
        hand_writings.append(strokes_list)
        transcriptions.append(transcription)

        if len(transcriptions) > num_lines:
            break

    return PreLoadedSource(hand_writings, transcriptions)


def compile_data(source, destination):
    hand_writings = []
    transcriptions = []
    for strokes, transcription in source.get_sequences():
        strokes_list = []
        for stroke in strokes:
            strokes_list.append(stroke.points)

        hand_writings.append(strokes_list)
        transcriptions.append(transcription)

        fetched = len(transcriptions)
        if fetched % 500 == 0:
            print('Fetched {} examples'.format(fetched))

    d = {
        'hand_writings': hand_writings,
        'transcriptions': transcriptions
    }

    with open(destination, 'w') as f:
        f.write(json.dumps(d))


if __name__ == '__main__':
    import argparse
    import os
    from data.factories import DataSplitter

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--destination_dir', type=str, default='./')
    parser.add_argument('--num_lines', type=int, default=8)
    args = parser.parse_args()

    it = LinesSource(OnlineSource(args.data_path))
    preloaded = load_data(source=it, num_lines=args.num_lines)
    splitter = DataSplitter(preloaded)

    sources = [splitter.train_data(), splitter.validation_data(),
               splitter.test_data()]

    dest_root = args.destination_dir
    file_names = ['train.json', 'validation.json', 'test.json']
    destinations = [os.path.join(dest_root, f) for f in file_names]

    for i in range(len(file_names)):
        compile_data(sources[i], destinations[i])
