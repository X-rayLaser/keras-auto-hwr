class MiniBatchGenerator:
    def __init__(self, source, example_adapter, batch_size=16):
        self._source = source
        self._example_adapter = example_adapter
        self._batch_size = batch_size

    def prepare_batch(self, seqs_in, seqs_out):
        return self._example_adapter.adapt_batch(seqs_in, seqs_out)

    def _raw_batches(self, batch_size):
        while True:
            seqs_in = []
            seqs_out = []
            for handwriting, transcription in self._source.get_sequences():
                seqs_in.append(handwriting)
                seqs_out.append(transcription)

                if len(seqs_in) >= batch_size:
                    yield seqs_in, seqs_out

                    seqs_in = []
                    seqs_out = []

            if len(seqs_in) > 0:
                yield seqs_in, seqs_out

    def get_examples(self):
        for seqs_in, seqs_out in self._raw_batches(self._batch_size):
            yield self.prepare_batch(seqs_in, seqs_out)

    def __len__(self):
        return len(self._source)
