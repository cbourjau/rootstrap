import multiprocessing
from collections import OrderedDict

import numpy as np
from tqdm import trange
from root_numpy import hist2array
from rootpy.io import root_open
import ROOT

from .collector import Collector


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
    def __init__(self, path, bootstrapper, files):
        self.path = path
        self.bootstrapper = bootstrapper
        self._read_files(files)

    def values(self):
        try:
            return np.sum(self.bootstrapper.current_weights * self.values_set, axis=-1, dtype=np.float64)
        except AttributeError:
            raise RuntimeError("Call Bootstrapper.draw first")

    def integrated_sample(self):
        """
        Sums all samples together.
        """
        return np.sum(self.values_set, axis=-1, dtype=np.float64)

    def _read_files(self, files):
        """
        Read in the full sample set and initiate the edges.
        """
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
    def __init__(self, files):
        # list of files which will be bootstrapped
        self.files = files
        # dict of all registered sources as `Sample_set`
        self.sources = {}
        # dict of all static sources and their set-up callbacks
        self.static_sources = {}
        # OrderedDict of all registered observable callbacks and the
        # function to comput the sample weight. The order is important
        # since latter calculation may depend on earlier ones. Order
        # is determined by the order the callbacks were registered in
        self._obs_callbacks = OrderedDict()
        # Dictionary of all the observables `Collector`s
        self._obs_collectors = {}
        # dictionary with values of observables from the current
        # iteration This dictionary is reset at every iteration. It is
        # ment to be used for observable callbacks to access results
        # from previous calculations
        self.current_observables = {}

    def register_histogram_source(self, name, path):
        """
        Register a histogram which is present in all the given files and consitutes a sums

        Parameters
        ----------
        name : str
            The name under which the given source will be accessable
        path : Path to the histogram within the root file
        """
        self.sources[name] = Sample_set(path, self, self.files)

    def register_observable(self, name, edges, callback, weight=lambda _: 1):
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
        weight : fn (this) -> scalar or np.ndarray
            Weight which is applied for a given sample when passing it
            to the `Collector` (usually number of events). Return
            value has to be broadcastable to the shape of the
            observable
        """
        self._obs_callbacks[name] = (callback, weight)
        if isinstance(edges, str):
            _source_edges = self.sources[edges].edges
            self._obs_collectors[name] = Collector(edges=_source_edges)
        elif isinstance(edges, (list, tuple)):
            self._obs_collectors[name] = Collector(edges=edges)
        elif callable(edges):
            self._obs_collectors[name] = Collector(edges=edges(self))
        else:
            raise ValueError("`edges` has invalid type {}".format(type(edges)))

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
            Called once with access to all the histograms. Must return np.ndarray of values
        """
        self.static_sources[name] = callback(self)

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
        for n in trange(ntimes, desc="Bootsrapping"):
            self._draw()
            # dictionary with values of observables from the current iteration
            self.current_observables = {}
            for name, (cback, weight) in self._obs_callbacks.items():
                _tmp = cback(self)
                # cache the results for latter calculations in this iteration
                self.current_observables[name] = _tmp
                # add the result to the respective collector to keep track of the mean and sigma
                self._obs_collectors[name].add(_tmp, weight(self))

        result = {}
        for name, col in self._obs_collectors.items():
            result[name] = col
        return result

    def _draw(self):
        nsamples = len(self.files)
        indices, counts = np.unique(np.random.choice(nsamples, size=nsamples),
                                    return_counts=True)
        # create an array with weights for each sub-sample item
        self.current_weights = np.zeros(shape=(nsamples, ), dtype=np.int)
        self.current_weights[indices] = counts
