import multiprocessing

import numpy as np
from root_numpy import hist2array
from rootpy.io import root_open

import ROOT


def _extract_hist_from_path(args):
    obj, path = args
    if isinstance(obj, str):
        with root_open(obj, 'READ') as f:
            return _extract_hist_from_path([f, path])
    head = path.split(".")[0]
    tail = path.split(".")[1:]
    next_obj = getattr(obj, head, None)

    if next_obj is None:
        next_obj = obj.FindObject(head)
        if next_obj == None:  # TObject to Null is '==' None but not 'is' None...
            raise ValueError("Object {} not found.".format(head))
    if isinstance(next_obj, (ROOT.TH1, ROOT.THnBase)):
        values, edges = hist2array(next_obj, return_edges=True)
    else:
        values, edges = _extract_hist_from_path([next_obj, ".".join(tail)])
    # delete list before closing file because... root...
    if isinstance(next_obj, ROOT.TList):
        next_obj.Delete()
    return values, edges


class Sample_set():
    def __init__(self, path, bootstrapper):
        self.path = path
        self.bootstrapper = bootstrapper

    def values(self):
        try:
            return np.sum(self.bootstrapper.current_weights * self.values_set, axis=-1, dtype=np.float64)
        except AttributeError:
            raise RuntimeError("Call Bootstrapper.draw first")

    def read_files(self, files):
        with root_open(files[0], 'READ') as f:
            values, self.edges = _extract_hist_from_path([f, self.path])
        self.values_set = np.zeros(shape=(values.shape + (len(files), )), dtype=np.float64)
        nworkers = 10
        pool = multiprocessing.Pool(nworkers)
        values_edges = pool.map(_extract_hist_from_path, zip(files, [self.path] * len(files)))
        for iset, (values, _) in enumerate(values_edges):
            self.values_set[..., iset] = values
        # for iset, fname in enumerate(files):
        #     def _worker():
        #         with root_open(fname, 'READ') as f:
        #             self.values_set[..., iset], _ = extract_hist_from_path(f, self.path)
        #     t = threading.Thread(target=_worker)
        #     t.start()
        # for t in threads:
        #     t.join()
        pool.close()


class Bootstrapper():
    def __init__(self):
        self.sample_sets = []

    def register(self, name, path):
        setattr(self, name, Sample_set(path, self))
        self.sample_sets.append(getattr(self, name))

    def read_files(self, files):
        self.nsamples = len(files)
        for s in self.sample_sets:
            s.read_files(files)

    def draw(self):
        indices, counts = np.unique(np.random.choice(self.nsamples, size=self.nsamples),
                                    return_counts=True)
        # create an array with weights for each sub-sample item
        self.current_weights = np.zeros(shape=(self.nsamples, ), dtype=np.int)
        self.current_weights[indices] = counts
