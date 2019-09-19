from sources.preloaded import PreLoadedSource


def words_source(source):
    def remove_apostrpohs(seq):
        res = ''.join(seq.split('&apos;'))
        res = ''.join(res.split('&quot;'))
        return res

    def clean(seq):
        s = ''
        for ch in seq.strip():
            if ch.isalpha():
                s += ch

        return s

    points = []
    transcriptions = []
    for seq_in, transcription in source.get_sequences():
        transcription = remove_apostrpohs(transcription)

        words = [clean(word) for word in transcription.split(' ')]

        points.append(seq_in)
        transcriptions.append(words)

    return PreLoadedSource(points, transcriptions)


def embeddings_source(source, num_examples):
    from train_on_embeddings import auto_encoder, get_embeddings
    embeddings, transcriptions, _, _ = get_embeddings(auto_encoder.get_encoder(), source, num_examples)
    return PreLoadedSource(embeddings, transcriptions)


def labels_source(source, mapping_table):
    seqs_in = []
    seqs_out = []

    for seq_in, seq_out in source.get_sequences():
        tmp = []

        for ch in seq_out:
            tmp.append(mapping_table.encode(ch))

        seqs_in.append(seq_in)
        seqs_out.append(tmp)

    return PreLoadedSource(seqs_in, seqs_out)


def ctc_adapted_source(source, padding_value=0):
    seqs_in = []
    seqs_out = []
    for seq_in, seq_out in source.get_sequences():
        seqs_in_pad = list(seq_in)

        while len(seqs_in_pad) <= 2 * len(seq_out) + 1:
            n = len(seqs_in_pad[0])
            seqs_in_pad.append([padding_value] * n)
        seqs_in.append(seqs_in_pad)

        seqs_out.append(seq_out)

    return PreLoadedSource(seqs_in, seqs_out)
