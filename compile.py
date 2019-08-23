from data import RandomOrderIterator
import json


def compile_data(data_path, destination, num_lines):
    it = RandomOrderIterator(data_path)

    hand_writings = []
    transcriptions = []
    for points, transcription in it.get_lines():
        hand_writings.append(points)
        transcriptions.append(transcription)

        if len(transcriptions) > num_lines:
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
