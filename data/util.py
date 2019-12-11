def estimate_screen_width(provider):
    right_most_xs = []
    for xs, _ in provider.get_sequences():
        points = flatten(xs)
        right_most_xs.append(max([x for x, y, t in points]))

    return int(round(max(right_most_xs)))


def estimate_letter_height(provider):
    def lc_letter_filter(item):
        xs, transcription = item
        return has_only_lower_case_letters(transcription)

    seqs = filter(lc_letter_filter, provider.get_sequences())
    hand_writings = [xs for xs, transcription in seqs]

    heights = [avg_stroke_height(xs) for xs in hand_writings]
    avg = sum(heights) / len(heights)
    return int(round(avg))


def has_only_lower_case_letters(s):
    words = s.split(' ')
    for w in words:
        if not w.isalpha():
            return False

        if not w.islower():
            return False

    return True


def flatten(strokes):
    res = []
    for stroke in strokes:
        res.extend(stroke)
    return res


def avg_stroke_height(xs, batch=1):
    num_batches = int(len(xs) / batch)

    heights = []
    for i in range(num_batches):
        strokes = xs[i:i + batch]
        points = flatten(strokes)
        ylist = [y for x, y, t in points]
        height = max(ylist) - min(ylist)
        heights.append(height)

    return sum(heights) / len(heights)
