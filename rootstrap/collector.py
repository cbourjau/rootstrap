import numpy as np


class Collector():
    def __init__(self):
        self.nentries = None
        self.sum = None
        self.square_sum = None

    def add(self, points):
        if self.nentries is None:
            self.nentries = np.zeros_like(points)
            self.sum = np.zeros_like(points)
            self.square_sum = np.zeros_like(points)

        self.sum += np.ma.fix_invalid(points, fill_value=0)
        self.square_sum += np.ma.fix_invalid(points, fill_value=0)**2
        self.nentries[~np.isnan(points)] += 1

    def mean(self):
        mean = np.copy(self.sum).astype(np.float32)
        mean[self.nentries == 0] = np.nan
        mean[self.nentries > 0] /= self.nentries[self.nentries > 0]
        return mean

    def sigma(self):
        sigma = np.zeros_like(self.sum).astype(np.float32)
        sigma[self.nentries <= 1] = np.nan
        sigma[self.nentries > 1] = np.sqrt(
            (1.0 / (self.nentries[self.nentries > 1] - 1) *
             (self.square_sum - self.nentries * (self.mean() ** 2))[self.nentries > 1])
        )
        return sigma
