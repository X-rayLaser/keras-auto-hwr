from sources.iam_online import OnlineSource, LinesSource
import json


def compile_data(data_path, destination, num_lines):
    it = LinesSource(OnlineSource(data_path))

    hand_writings = []
    transcriptions = []
    for points, transcription in it.get_sequences():
        hand_writings.append(points)
        transcriptions.append(transcription)

        fetched = len(transcriptions)
        if fetched % 500 == 0:
            print('Fetched {} examples'.format(fetched))

        if fetched > num_lines:
            break

    d = {
        'hand_writings': hand_writings,
        'transcriptions': transcriptions
    }

    with open(destination, 'w') as f:
        f.write(json.dumps(d))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--destination', type=str, default='./')
    parser.add_argument('--num_lines', type=int, default=8)
    args = parser.parse_args()

    compile_data(args.data_path, args.destination, args.num_lines)
