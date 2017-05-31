import numpy as np
from rootpy.plotting import Hist1D, Hist2D, Hist3D
from root_numpy import array2hist


class Collector():
    def __init__(self, edges=None):
        self.nentries = None
        self.sum = None
        self.square_sum = None
        self.edges = edges

    def add(self, points, weight=1):
        points = np.array(points)
        if self.nentries is None:
            self.nentries = np.zeros_like(points).astype(np.float128)
            self.sum = np.zeros_like(points).astype(np.float128)
            self.square_sum = np.zeros_like(points).astype(np.float128)

        self.sum += np.ma.fix_invalid(points * weight, fill_value=0)
        self.square_sum += np.ma.fix_invalid(points, fill_value=0)**2 * weight
        # Only increas counts in the non-nan bins, but we have to make
        # sure the weights broadcast correctly
        self.nentries[~np.isnan(points)] += (np.ones_like(points) * weight)[~np.isnan(points)]

    def mean(self):
        mean = np.copy(self.sum)
        mean[self.nentries == 0] = np.nan
        mean[self.nentries > 0] /= self.nentries[self.nentries > 0]
        return mean.astype(np.float64)

    def sigma(self):
        sigma = np.zeros_like(self.sum)
        sigma[self.nentries <= 1] = np.nan
        sigma[self.nentries > 1] = np.sqrt(
            (1.0 / (self.nentries[self.nentries > 1] - 1) *
             (self.square_sum - self.nentries * (self.mean() ** 2))[self.nentries > 1])
        )
        return sigma.astype(np.float64)

    def as_root_hist(self, invalid_fill_value=0):
        if not self.edges:
            raise ValueError("'edges' need to be set befor exporting to root hist")
        if len(self.edges) == 1:
            h = Hist1D(*self.edges, type='D')
        elif len(self.edges) == 2:
            h = Hist2D(*self.edges, type='D')
        elif len(self.edges) == 3:
            h = Hist3D(*self.edges, type='D')
        else:
            raise ValueError("Export to root histogram only supported for ndim <= 3")
        array2hist(np.ma.fix_invalid(self.mean(), fill_value=invalid_fill_value), h,
                   errors=np.ma.fix_invalid(self.sigma(), fill_value=0))
        return h
