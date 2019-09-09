from data.char_table import CharacterTable
from sources.compiled import CompilationSource
from data.generators import AutoEncoderGenerator
from models.seq2seq import Seq2seqAutoencoder
from sources.preloaded import PreLoadedSource
from data.preprocessing import PreProcessor, DeltaSignal, SequencePadding, StreamSplit, Normalization
from models.seq2seq import SequenceToSequenceTrainer
from sources.iam_online import StrokesSource, BadStrokeException
from train_autoencoder import VanillaAutoEncoder, FeedForwardAutoEncoderGenerator, padded_source


lrate = 0.001
validation_steps = 10
feature_extraction_epochs = 2
embedding_size = 8

num_train_examples = 1024
num_val_examples = 128
end2end_epochs = 1000
encoding_size = 16

charset = ''.join([chr(i) for i in range(32, 128)])
char_table = CharacterTable(charset)


compilation_train_source = CompilationSource('./compiled/train.json')

compilation_validation_source = CompilationSource('./compiled/validation.json')

#auto_encoder = Seq2seqAutoencoder(encoding_size=embedding_size, input_channels=2, output_channels=2)

seq_len = 508
auto_encoder = VanillaAutoEncoder(seq_len * 2, embedding_size)
auto_encoder.load('./weights/auto_encoder')


def get_embeddings(encoder, source, num_examples):
    import numpy as np

    embeddings = []
    transcriptions = []

    max_emb_len = 0
    max_transciption_len = 0

    for strokes, transcription in source.get_sequences():
        transcription = transcription.split(' ')[-1]
        print(transcription)
        if len(embeddings) > num_examples:
            break
        embedding_seq = []
        for stroke in strokes:
            try:
                deltas = stroke.stroke_to_points()
                while len(deltas) < seq_len:
                    deltas.append((0, 0))

                deltas = deltas[:seq_len]
                n = len(deltas)
                x = np.array(deltas)

                x = x.reshape((1, n * 2))

                embedding = encoder.predict(x)[0]
                embedding_seq.append(embedding)
            except BadStrokeException:
                pass

        if len(embedding_seq) > 1:
            max_emb_len = max(max_emb_len, len(embedding_seq))
            max_transciption_len = max(max_transciption_len, len(transcription))
            embeddings.append(embedding_seq)
            transcriptions.append(transcription)

    return embeddings, transcriptions, max_emb_len, max_transciption_len


def pad_sequences(embeddings, transcriptions, max_emb_len, max_transcription_len):
    padded_embeddings = []
    padded_transcriptions = []
    for i in range(len(transcriptions)):
        emb = embeddings[i]
        t = transcriptions[i]

        while len(emb) < max_emb_len:
            emb.append([0] * embedding_size)

        while len(t) < max_transcription_len:
            t += '\n'

        padded_embeddings.append(emb[:max_emb_len])
        #padded_transcriptions.append(t[:max_transcription_len])
        padded_transcriptions.append(transcriptions[i])

    return padded_embeddings, padded_transcriptions


from data.generators import DataSetGenerator, AttentionModelDataGenerator

preprocessor = PreProcessor()

encoder = auto_encoder.get_encoder()

train_embeddings, train_transcriptions, max_emb_len, max_transcription_len = get_embeddings(encoder, compilation_train_source, num_train_examples)

train_embeddings, train_transcriptions = pad_sequences(train_embeddings, train_transcriptions, max_emb_len, max_transcription_len)

val_embeddings, val_transcriptions, _, _ = get_embeddings(encoder, compilation_validation_source, num_val_examples)

val_embeddings, val_transcriptions = pad_sequences(val_embeddings, val_transcriptions, max_emb_len, max_transcription_len)

preloaded_train = PreLoadedSource(train_embeddings, train_transcriptions)
preloaded_val = PreLoadedSource(val_embeddings, val_transcriptions)

Tx = max_emb_len
Ty = max_transcription_len + 1

#train_gen = AttentionModelDataGenerator(preloaded_train, char_table, preprocessor, Tx, Ty, encoder_states=encoding_size, channels=embedding_size)
#val_gen = AttentionModelDataGenerator(preloaded_val, char_table, preprocessor, Tx, Ty, encoder_states=encoding_size, channels=embedding_size)

#from models.attention import Seq2SeqWithAttention

#trainer = Seq2SeqWithAttention(char_table, encoding_size=encoding_size, Tx=Tx, Ty=Ty, channels=embedding_size)

batch_size = 16
from models.convnet import ConvolutionalRecognizer, ConvNetGenerator, Vocabulary

transcriptions = []
for _, transcription in compilation_train_source.get_sequences():
    transcriptions.append(transcription)

vocab = Vocabulary(transcriptions, max_size=1000)
print('vocab len', len(vocab))

trainer = ConvolutionalRecognizer(vocab, Tx, channels=embedding_size, embedding_size=16)

train_gen = ConvNetGenerator(vocab, preloaded_train, preprocessor, channels=embedding_size)
val_gen = ConvNetGenerator(vocab, preloaded_val, preprocessor, channels=embedding_size)

trainer.fit_generator(lrate, train_gen, val_gen, train_gen.get_examples(batch_size=batch_size),
                      steps_per_epoch=int(len(train_gen) / batch_size) + 1,
                      validation_data=val_gen.get_examples(batch_size),
                      validation_steps=validation_steps,
                      epochs=end2end_epochs)
