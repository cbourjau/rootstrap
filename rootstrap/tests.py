from glob import glob
from unittest import TestCase

import numpy as np

from rootstrap import Bootstrapper, Collector
from rootpy.io import root_open
from rootpy.plotting import Hist1D, Hist2D

NBINS_FROM_FOLDER = 3


class Test_end2end(TestCase):
    def setUp(self):
        for ifile in range(5):
            with root_open("AnalysisResults{}.root".format(ifile), 'RECREATE') as f:
                for i in range(NBINS_FROM_FOLDER):
                    f.mkdir("dir{}".format(i))
                    f.cd("dir{}".format(i))
                    h_proton = Hist1D(3, 0, 3, name="proton_dist")
                    h_pion = Hist1D(3, 0, 3, name="pion_dist")

                    # + 1 so that we don't divide by 0 later on
                    [h_proton.Fill(x, ifile + 1) for x in range(h_proton.GetNbinsX())]
                    [h_pion.Fill(x, ifile + 1) for x in range(h_proton.GetNbinsX())]
                    h_proton.Write()
                    h_pion.Write()

    def test_end2end(self):
        files = glob("./AnalysisResults*.root")

        bs = Bootstrapper(files, root_dir=".", dim_from_dir_names=None)
        bs.read_files()

        proton_collector = Collector()
        ratio_collector = Collector()

        bs.draw()  # set up a new bootstrapped sample
        # Check resulting shape; compare with setUp()
        self.assertEqual(bs.proton_dist.values().shape, (NBINS_FROM_FOLDER, 3))
        # print bs.pion_dist.values_set
        for i in range(10000):
            bs.draw()  # set up a new bootstrapped sample
            proton_collector.add(bs.proton_dist.values())
            ratio_collector.add(bs.proton_dist.values() / bs.pion_dist.values())

        # 5 samples; each bin is sum(range(5) + 1) == 15
        mean, sigma = proton_collector.mean(), proton_collector.sigma()
        self.assertTrue(np.all(mean - sigma < np.array([15, 15, 15])))
        self.assertTrue(np.all(mean + sigma > np.array([15, 15, 15])))
        np.testing.assert_array_equal(ratio_collector.mean(),
                                      np.array([[1., 1., 1.]] * NBINS_FROM_FOLDER))


class Test_end2end_dim_exists_in_original_hists(TestCase):
    def setUp(self):
        for ifile in range(5):
            with root_open("AnalysisResults{}.root".format(ifile), 'RECREATE') as f:
                for i in range(NBINS_FROM_FOLDER):
                    f.mkdir("dir{}".format(i))
                    f.cd("dir{}".format(i))
                    h_proton = Hist2D(3, 0, 3, 1, i * 10, (i + 1) * 10, name="proton_dist")
                    h_pion = Hist2D(3, 0, 3, 1, i * 10, (i + 1) * 10, name="pion_dist")

                    # + 1 so that we don't divide by 0 later on
                    [h_proton.Fill(x, i * 10, ifile + 1) for x in range(h_proton.GetNbinsX())]
                    [h_pion.Fill(x, i * 10, ifile + 1) for x in range(h_proton.GetNbinsX())]
                    h_proton.Write()
                    h_pion.Write()

    def test_end2end(self):
        files = glob("./AnalysisResults*.root")

        bs = Bootstrapper(files, root_dir=".", dim_from_dir_names=1)
        bs.read_files()

        proton_collector = Collector()
        ratio_collector = Collector()

        bs.draw()  # set up a new bootstrapped sample
        # Check resulting shape; compare with setUp()
        self.assertEqual(bs.proton_dist.values().shape, (3, NBINS_FROM_FOLDER))
        # print bs.pion_dist.values_set
        for i in range(10000):
            bs.draw()  # set up a new bootstrapped sample
            proton_collector.add(bs.proton_dist.values())
            ratio_collector.add(bs.proton_dist.values() / bs.pion_dist.values())

        # 5 samples; each bin is sum(range(5) + 1) == 15
        mean, sigma = proton_collector.mean(), proton_collector.sigma()
        self.assertTrue(np.all(mean - sigma < np.array([15, 15, 15])))
        self.assertTrue(np.all(mean + sigma > np.array([15, 15, 15])))
        np.testing.assert_array_equal(ratio_collector.mean(),
                                      np.array([[1., 1., 1.]] * NBINS_FROM_FOLDER))


class Test_collector(TestCase):
    def test_mean(self):
        points = np.linspace(0, 10, num=50)
        c = Collector()
        [c.add([p]) for p in points]
        self.assertEqual(np.mean(points), c.mean()[0])
        self.assertLessEqual(np.std(points) * 0.98, c.sigma()[0])
        self.assertLessEqual(c.sigma()[0], np.std(points) * 1.02)

    def test_as_hist(self):
        points = np.linspace(0, 10, num=1000).reshape((10, 10, 10))
        c = Collector(edges=[range(10 + 1), range(10 + 1)])
        [c.add(p) for p in points]
        c.as_root_hist()
