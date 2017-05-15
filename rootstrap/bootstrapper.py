import multiprocessing

import numpy as np
from root_numpy import hist2array
from rootpy.io import root_open
from rootpy.plotting import Hist

from ROOT import TH1, THnBase, TList


def flatten_TCrap(thing):
    """
    Walk all paths in `thing` and yield the full path and the final histogram
    """
    if isinstance(thing, (TH1, THnBase, Hist)):
        yield [thing.GetName(), hist2array(thing, return_edges=True)]

    for el in thing:
        if isinstance(el, (TH1, THnBase, Hist)):
            yield [thing.GetName() + "." + el.GetName(), hist2array(el, return_edges=True)]
            continue
        for k, v in flatten_TCrap(el):
            yield el.GetName() + "." + k, v
        # Clean up ROOT's pathetic handling of TLists...
        if isinstance(el, TList):
            el.Delete()


def merge_arrays(flat_dict, hist_name, target_dim=None):
    """
    Due to restrictions on the grid, multiplicity is binned over
    different histograms. This function removes the histogram from the
    given dict!!!

    Parameters
    ----------
    flat_dict : dict
        Dictionary reflecting the flattened root-file folder structure
    hist_name : str
        Name of the histogram which will be merged over different TLists (eg. 'pairs_itsits')
    target_dim : int
        The dimension which represents the merged TLists (eg. -2 for mult), if None, create new dimension

    Returns
    -------
    list :
        List of [values, edges]
    """
    def compute_new_edges(_edges):
        if target_dim is None:
            # We have no information about the binning of that new dimension
            new_e = np.arange(0, len(_edges) + 1)
        else:
            # The information for the new bins is already saved in one of the existing
            # dimensions of the original histogram
            new_e = None
            for e in _edges:
                if new_e is None:
                    new_e = list(e)
                else:
                    new_e.append(e[1])
            new_e = np.array(new_e)
        all_edges = _edges[0]
        if target_dim:
            all_edges[target_dim] = new_e
            return all_edges
        else:
            return [new_e, all_edges]

    sorted_keys = sorted([k for k in flat_dict.keys() if hist_name in k])
    _vals, _edges = zip(*[flat_dict[k] for k in sorted_keys])
    if target_dim:
        _vals = np.concatenate(_vals, target_dim)
    else:
        _vals = np.stack(_vals, axis=0)
    return _vals, compute_new_edges(_edges)


# def _extract_hist_from_path(args):
#     obj, path = args
#     if isinstance(obj, str):
#         with root_open(obj, 'READ') as f:
#             return _extract_hist_from_path([f, path])
#     head = path.split(".")[0]
#     tail = path.split(".")[1:]
#     next_obj = getattr(obj, head, None)

#     if next_obj is None:
#         next_obj = obj.FindObject(head)
#         if next_obj == None:  # TObject to Null is '==' None but not 'is' None...
#             raise ValueError("Object {} not found in {}.".format(head, obj))
#     if isinstance(next_obj, (ROOT.TH1, ROOT.THnBase)):
#         values, edges = hist2array(next_obj, return_edges=True)
#     else:
#         values, edges = _extract_hist_from_path([next_obj, ".".join(tail)])
#     # delete list before closing file because... root...
#     if isinstance(next_obj, ROOT.TList):
#         next_obj.Delete()
#     return values, edges


class Sample_set():
    def __init__(self, bootstrapper, edges, first_value):
        self.edges = edges
        self._bs = bootstrapper
        # number of value_sets appended to this sample set
        self._filled_sets = 0
        self.values_set = np.zeros(shape=(first_value.shape + (len(self._bs.files), )),
                                   dtype=first_value.dtype)
        self.values_set[..., self._filled_sets] = first_value

    def append(self, values):
        """
        Append a set of values (e.g. the results of a run) to this sample set
        """
        self._filled_sets += 1
        self.values_set[..., self._filled_sets] = values

    def values(self):
        try:
            return np.sum(self._bs.current_weights * self.values_set.astype(np.float64),
                          axis=-1)
        except AttributeError:
            raise RuntimeError("Call Bootstrapper.draw first")

    # def read_files(self, files):
    #     with root_open(files[0], 'READ') as f:
    #         values, self.edges = _extract_hist_from_path([f, self.path])
    #     self.values_set = np.zeros(shape=(values.shape + (len(files), )), dtype=values.dtype)
    #     nworkers = 9
    #     pool = multiprocessing.Pool(nworkers)
    #     values_edges = pool.map(_extract_hist_from_path, zip(files, [self.path] * len(files)))
    #     for iset, (values, _) in enumerate(values_edges):
    #         self.values_set[..., iset] = values
    #     pool.close()


def _fixed_arrays_from_file(args):
    (fname, root_dir, dim_from_dirnames) = args
    with root_open(fname, 'READ') as f:
        # This probably breaks if there is more then one level of depth!
        if root_dir == ".":
            thing = f
        else:
            thing = f.__getattr__(root_dir)
        # get a flat dictionary of all the histograms below the root directory
        d = {k: v for k, v in flatten_TCrap(thing)}
    # find all the histogram names which we want to register / keep
    hnames = list(set([k.split('.')[-1] for k in d.keys()]))
    # FIXME: merge the multiplicity dimension, which is currently in the folder hirachy,
    # into the ndarray. Clearly, this shouldn't depend on my analysis!
    merged_d = {hname: merge_arrays(d, hname, dim_from_dirnames) for hname in hnames}
    return merged_d


class Bootstrapper():
    """
    This class ties together logically dependent `Sample_set`s. I.e.,
    we might have a various histograms which were produced in parallel
    like a "eventCounter" and a "particle pair counter". Individual
    sample sets are registered with the `register` function. The IO is
    derefered until the `read_files` function is called.
    """
    def __init__(self, files, root_dir, dim_from_dir_names=None):
        self.files = files
        self.root_dir = root_dir
        self.sample_sets = []
        self._dim_from_dir_names = dim_from_dir_names

    # def register(self, name, path):
    #     setattr(self, name, Sample_set(path, self))
    #     self.sample_sets.append(getattr(self, name))

    def read_files(self):
        self.nsamples = len(self.files)
        nworkers = 9
        pool = multiprocessing.Pool(nworkers)
        # create a list of dicts containing the "fixed" histograms
        dict_merged_hists = pool.map(_fixed_arrays_from_file,
                                     zip(self.files,
                                         [self.root_dir] * len(self.files),
                                         [self._dim_from_dir_names] * len(self.files)))
        pool.close()
        while dict_merged_hists:
            merged_d = dict_merged_hists.pop()
            while merged_d:
                hname, (values, edges) = merged_d.popitem()
                try:
                    getattr(self, hname).append(values)
                except AttributeError:
                    setattr(self, hname, Sample_set(self, edges=edges, first_value=values))

    def draw(self):
        indices, counts = np.unique(np.random.choice(self.nsamples, size=self.nsamples),
                                    return_counts=True)
        # create an array with weights for each sub-sample item
        self.current_weights = np.zeros(shape=(self.nsamples, ), dtype=np.int)
        self.current_weights[indices] = counts
