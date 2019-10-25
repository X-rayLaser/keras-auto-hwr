class SearchPath:
    def __init__(self, path=None):
        if path is None:
            self._path = []
        else:
            self._path = path

    def branch_off(self, label, p):
        path = self._path + [(label, p)]
        return SearchPath(path)

    @property
    def labels(self):
        return [label for label, p in self._path]

    @property
    def likelihood(self):
        if self._path:
            probs = [p for label, p in self._path]
            res = 1
            for p in probs:
                res *= p
            return res
        return 0


class PathBuilder:
    def __init__(self, roots):
        self._paths = []
        for label, p in roots:
            search_path = SearchPath()
            search_path = search_path.branch_off(label, p)
            self._paths.append(search_path)

    def make_step(self, pmfs):
        if len(pmfs) != len(self._paths):
            raise WrongNumberOfPMFsException()

        candidates = []
        for i in range(len(self._paths)):
            search_path = self._paths[i]
            pmf = pmfs[i]
            for label, p in enumerate(pmf):
                candidates.append(search_path.branch_off(label, p))

        self._paths = self._best_paths(candidates, limit=len(pmfs))

    def _best_paths(self, paths, limit):
        return sorted(paths, key=lambda c: c.likelihood, reverse=True)[:limit]

    @property
    def best_path(self):
        best_path = self._best_paths(self._paths, limit=1)[0]
        return best_path.labels

    @property
    def paths(self):
        res = []
        for search_path in self._paths:
            res.append(search_path.labels)
        return res


class WrongNumberOfPMFsException(Exception):
    pass


class StatesKeeper:
    def __init__(self, initial_state):
        self._paths = {}
        self._initial_state = initial_state

    def store(self, path, state):
        self._paths[tuple(path)] = state

    def retrieve(self, path):
        if path:
            return self._paths[tuple(path)]
        else:
            return self._initial_state


class BaseBeamSearch:
    def __init__(self, start_of_seq, end_of_seq, beam_size=3, max_len=150):
        self._sos = start_of_seq
        self._eos = end_of_seq
        self._beam_size = beam_size
        self._max_len = max_len

    def _without_last(self, path):
        return path[:-1]

    def _remove_special(self, path):
        path = path[1:]
        if path[-1] == self._eos:
            return self._without_last(path)
        return path

    def _split_path(self, path):
        prefix = self._without_last(path)
        last_one = path[-1]
        return prefix, last_one

    def generate_sequence(self):
        y0 = self._sos

        decoder_state = self.get_initial_state()

        keeper = StatesKeeper(decoder_state)

        builder = PathBuilder([(y0, 1.0)])

        for _ in range(self._max_len):
            pmfs = []
            for path in builder.paths:
                prefix, label = self._split_path(path)
                state = keeper.retrieve(prefix)
                next_pmf, next_state = self.decode_next(label, state)
                keeper.store(path, next_state)
                pmfs.append(next_pmf)

            builder.make_step(pmfs)
            if builder.best_path[-1] == self._eos:
                break

        return self._remove_special(builder.best_path)

    def get_initial_state(self):
        raise NotImplementedError

    def decode_next(self, prev_y, prev_state):
        raise NotImplementedError


class BeamCandidate:
    def __init__(self, full_sequence, character, likelihood, state):
        self.full_sequence = full_sequence
        self.character = character
        self.likelihood = likelihood
        self.state = state

    def branch_off(self, character, likelihood, state):
        seq = self.full_sequence + character
        return BeamCandidate(seq, character, likelihood, state)


# todo: consider better implementation for StatesKeeper
