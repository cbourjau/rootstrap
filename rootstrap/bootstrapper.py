import numpy as np
from root_numpy import hist2array
from rootpy.io import root_open
import ROOT


def extract_hist_from_path(f, path):
    head = path.split(".")[0]
    tail = path.split(".")[1:]
    obj = getattr(f, head, None)
    if obj is None:
        obj = f.FindObject(head)
    if isinstance(obj, (ROOT.TH1, ROOT.THnBase)):
        values, edges = hist2array(obj, return_edges=True)
    else:
        values, edges = extract_hist_from_path(obj, ".".join(tail))
    # delete list before closing file because... root...
    obj.Delete()
    return values, edges


class Sample_set():
    def __init__(self, path, bootstrapper):
        self.path = path
        self.bootstrapper = bootstrapper

    def values(self):
        return np.sum(self.bootstrapper.current_weights * self.values_set, axis=-1, dtype=np.float32)

    def read_files(self, files):
        with root_open(files[0], 'READ') as f:
            values, self.edges = extract_hist_from_path(f, self.path)
        self.values_set = np.zeros(shape=(values.shape + (len(files), )), dtype=np.float32)
        for iset, fname in enumerate(files):
            with root_open(fname, 'READ') as f:
                self.values_set[..., iset], _ = extract_hist_from_path(f, self.path)


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
