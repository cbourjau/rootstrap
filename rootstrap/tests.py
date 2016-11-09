from glob import glob
from unittest import TestCase
from rootstrap import Bootstrapper, Collector
from rootpy.io import root_open
from rootpy.plotting import Hist1D


class Test_end2end(TestCase):
    def setUp(self):
        for i in range(5):
            with root_open("AnalysisResults{}.root".format(i), 'w') as f:
                h = Hist(10, 0, 1, name=str(1))
                h.FillRandom(100)
                h.Write()

    def test_end2end(self):
        files = glob("./AnalysisResults*.root")

        bs = Bootstrapper()  # set other options here as well
        bs.register("proton_dist", path="sums.proton_dist")
        bs.register("pion_dist", path="sums.pion_dist")
        bs.read_files(files)

        ratio_collector = Collector()
        for i in range(100):
            bs.draw()  # set up a new bootstrapped sample
            ratio_collector.add(bs.proton_dist.values / bs.pion_dist.values)

        print "Mean:", ratio_collector.mean
        print "Sigma:", ratio_collector.sigma  # assuming gaussian
