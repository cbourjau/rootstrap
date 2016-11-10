from glob import glob
from unittest import TestCase

import numpy as np

from rootstrap import Bootstrapper, Collector
from rootpy.io import root_open
from rootpy.plotting import Hist1D


class Test_end2end(TestCase):
    def setUp(self):
        for i in range(5):
            with root_open("AnalysisResults{}.root".format(i), 'w') as f:
                h_proton = Hist1D(3, 0, 3, name="proton_dist")
                h_pion = Hist1D(3, 0, 3, name="pion_dist")
                # + 1 so that we don't divide by 0 later on
                [h_proton.Fill(x, i + 1) for x in range(h_proton.GetNbinsX())]
                [h_pion.Fill(x, i + 1) for x in range(h_proton.GetNbinsX())]
                h_proton.Write()
                h_pion.Write()
                f.Write()

    def test_end2end(self):
        files = glob("./AnalysisResults*.root")

        bs = Bootstrapper()  # set other options here as well
        bs.register("proton_dist", path="proton_dist")
        bs.register("pion_dist", path="pion_dist")
        bs.read_files(files)

        proton_collector = Collector()
        ratio_collector = Collector()
        for i in range(10000):
            bs.draw()  # set up a new bootstrapped sample
            proton_collector.add(bs.proton_dist.values())
            ratio_collector.add(bs.proton_dist.values() / bs.pion_dist.values())

        # 5 samples; each bin is sum(range(5) + 1) == 15
        mean, sigma = proton_collector.mean(), proton_collector.sigma()
        self.assertTrue(np.all(mean - sigma < np.array([15, 15, 15])))
        self.assertTrue(np.all(mean + sigma > np.array([15, 15, 15])))
        np.testing.assert_array_equal(ratio_collector.mean(), np.array([1., 1., 1.]))


class Test_collector(TestCase):
    def test_mean(self):
        points = np.linspace(0, 10, num=50)
        print points
        c = Collector()
        [c.add([p]) for p in points]
        self.assertEqual(np.mean(points), c.mean()[0])
        self.assertLessEqual(np.std(points) * 0.98, c.sigma()[0])
        self.assertLessEqual(c.sigma()[0], np.std(points) * 1.02)
