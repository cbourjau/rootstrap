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
        pool.close()


class Bootstrapper():
    def __init__(self):
        self.sample_sets = []
        # list of all registered sources as `Sample_set`
        self.sources = []
        # list of all static sources and their set-up callbacks (name, callback)
        self.static_sources = []
        # list of all registered observable callbacks as tuples: (name, callback)
        # This has to be a list since the order matters
        self._obs_callback = []
        # Dictionary of all the observables `Collector`s
        self._obs_collectors = {}
        # Function called to compute sample weights
        self.sample_weight = lambda cls: 1

    def register_histogram_source(self, name, path):
        """
        Register a histogram which is present in all the given files and consitutes a sums

        Parameters
        ----------
        name : str
            The name under which the given source will be accessable
        path : Path to the histogram within the root file
        """
        setattr(self, name, Sample_set(path, self))
        self.sample_sets.append(getattr(self, name))

    def register_observable(self, name, edges, callback):
        """
        Register an observable which will be calculated in each iteration

        Parameters
        ----------
        name : str
            Name of this observable; must be unique. If other
            calculations depend on this value, it can be accessed
            throught this name
        edges : str, list, function
            Edges of this observable. If `str`, import edges from a
            source histogram of this name. If `list` of nd.array, use
            these values as bin edges. If `function` (this)->(list), use this
            function to calculate bin edges. Called once.
        callback : fn (this) -> ndarray
            Called once per iteration. Must return an nd.array of
            dimensionality fitting to the observables bin edges.
        """
        pass

    def register_static_source(self, name, callback):
        """
        Register a souce which is static over all iterations but is
        needed for the calculation of an observable. I.e., dead
        regions of a detector or efficiencies.

        Parameters
        ----------
        name : str
            The name under which the given source will be accessable
        callback : function
            Called once with access to all the histograms. Must return `[values, edges]`
        """
        self.static_sources.append((name, callback))

    def register_sample_weight(self, callback):
        """
        Register the logic for calculating a given iteration's sample
        weight. If this function is not called, all samples are
        weighted equally. Usually one would like to weight a sample by
        e.g. number of events

        Parameters
        ----------
        callback : function
            Function called once per iteration. Must return a scalar value
        """
        self.sample_weight = callback

    def bootstrap(self, ntimes):
        """
        Bootstrap this configuration `ntimes`.

        Parameters
        ----------
        ntimes : int
            Number of iterations

        Returns
        -------
        dict : Dictionary of all the registered observables
        """
        for n in range(ntimes):
            self._draw()
            # dictionary with values of observables from the current iteration
            self.current_observables = {}
            for name, cback in self._obs_callbacks:
                self.current_observables[name] = cback(self)

        result = {}
        for name, col in self._obs_collectors:
            result[name] = col
        return result

    def read_files(self, files):
        self.nsamples = len(files)
        for s in self.sample_sets:
            s.read_files(files)

    def _draw(self):
        indices, counts = np.unique(np.random.choice(self.nsamples, size=self.nsamples),
                                    return_counts=True)
        # create an array with weights for each sub-sample item
        self.current_weights = np.zeros(shape=(self.nsamples, ), dtype=np.int)
        self.current_weights[indices] = counts
