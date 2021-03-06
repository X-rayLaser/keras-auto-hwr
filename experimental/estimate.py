import traceback
import numpy as np
from data.encodings import CharacterTable


class PerformanceMetric:
    def __init__(self, inference_model, num_trials):
        self._inference_model = inference_model
        self._num_trials = num_trials

    def estimate(self, test_data):
        raise NotImplementedError

    def compare(self, prediction, ground_true):
        errors = abs(len(prediction) - len(ground_true))

        seq_len = min(len(prediction), len(ground_true))
        for i in range(seq_len):
            if ground_true[i] != prediction[i]:
                errors += 1

        return errors

    def predict(self, input_sequence):
        # todo: refactor
        labels = self._inference_model.predict(
            input_sequence
        )
        char_table = CharacterTable()

        s = ''.join([char_table.decode(label) for label in labels])

        return s.strip()


class Seq2seqMetric(PerformanceMetric):
    def estimate(self, test_data):
        characters_total = 0
        errors_total = 0

        count = 0

        for (hand_writings, _), ys in test_data.get_examples():
            count += 1
            if count > self._num_trials:
                return errors_total / characters_total

            for i in range(len(ys)):
                classes = [(np.argmax(v)) for v in ys[i]]
                ground_true = ''.join([self._inference_model.char_table.decode(c) for c in classes]).strip()

                try:
                    x = hand_writings[i]
                    prediction = self.predict(x)
                except:
                    traceback.print_exc()
                    prediction = ''

                characters_total += len(ground_true)

                errors = self.compare(prediction, ground_true)
                errors_total += errors

                print(ground_true, '->', prediction)


class AttentionModelMetric(PerformanceMetric):
    def estimate(self, test_data):
        characters_total = 0
        errors_total = 0

        count = 0

        for inputs, ys in test_data.get_examples(batch_size=1):
            count += 1
            if count > self._num_trials:
                return errors_total / characters_total

            ys = np.array(ys)[:, 0, :]
            classes = [(np.argmax(v)) for v in ys]
            ground_true = ''.join([self._inference_model.char_table.decode(c) for c in classes]).strip()

            try:
                prediction = self.predict(inputs)
            except:
                traceback.print_exc()
                prediction = ''

            characters_total += len(ground_true)

            errors = self.compare(prediction, ground_true)
            errors_total += errors

            print(ground_true, '->', prediction)
