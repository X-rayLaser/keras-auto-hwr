import traceback
import numpy as np


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
        s = self._inference_model.predict(
            input_sequence
        )
        return s.strip()


class CharacterErrorRate(PerformanceMetric):
    def estimate(self, test_data):
        characters_total = 0
        errors_total = 0

        for (hand_writings, _), ys in test_data.get_examples(batch_size=self._num_trials):

            count = 0
            for i in range(len(ys)):
                count += 1
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
            return errors_total / characters_total
