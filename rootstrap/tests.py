from glob import glob
from unittest import TestCase

import numpy as np

from rootstrap import Bootstrapper, Collector
from rootpy.io import root_open
from rootpy.plotting import Hist1D
from ROOT import TList, TObject


class Test_bootstrapper(TestCase):
    def setUp(self):
        for i in range(5):
            tl = TList()
            with root_open("AnalysisResults{}.root".format(i), 'RECREATE') as f:
                f.mkdir("dir")
                f.cd("dir")
                h_proton = Hist1D(3, 0, 3, name="proton_dist")
                h_pion = Hist1D(3, 0, 3, name="pion_dist")

                # + 1 so that we don't divide by 0 later on
                [h_proton.Fill(x, i + 1) for x in range(h_proton.GetNbinsX())]
                [h_pion.Fill(x, i + 1) for x in range(h_proton.GetNbinsX())]
                tl.Add(h_proton)
                tl.Add(h_pion)
                tl.Write("list")

    def test_observables_edges(self):
        """
        Test the different ways of defining the observables edges
        """
        files = glob("./AnalysisResults*.root")
        bs = Bootstrapper(files)
        bs.register_histogram_source("proton_dist", path="dir.list.proton_dist")

        def dummy_calc(bs):
            pass
        bs.register_observable(name='by_list', edges=[[1, 2, 3]], callback=dummy_calc)
        np.testing.assert_array_equal([[1, 2, 3]], bs._obs_collectors['by_list'].edges)

        bs.register_observable(name='by_source_name', edges='proton_dist', callback=dummy_calc)
        np.testing.assert_array_equal([[0., 1., 2., 3.]], bs._obs_collectors['by_source_name'].edges)

        def calc_edges(strapper):
            return [[1, 2, 3]]
        bs.register_observable('by_callback', edges=calc_edges, callback=dummy_calc)
        np.testing.assert_array_equal([[1, 2, 3]], bs._obs_collectors['by_callback'].edges)

    def test_static_sources(self):
        """
        Test initiation of static sources I.e., dead detector regions
        which are infered from the full sample.
        """
        files = glob("./AnalysisResults*.root")
        bs = Bootstrapper(files)
        bs.register_histogram_source("proton_dist", path="dir.list.proton_dist")
        bs.register_static_source('static', lambda bs: bs.sources['proton_dist'].integrated_sample())
        np.testing.assert_array_equal(bs.static_sources['static'], [15, 15, 15])

    def test_end2end(self):
        files = glob("./AnalysisResults*.root")

        bs = Bootstrapper(files)
        bs.register_histogram_source("proton_dist", path="dir.list.proton_dist")
        bs.register_histogram_source("pion_dist", path="dir.list.pion_dist")

        def proton_pion_ration(strapper):
            return strapper.sources['proton_dist'].values() / strapper.sources['proton_dist'].values()

        bs.register_observable(name='ratio', edges='proton_dist', callback=proton_pion_ration)
        bs.register_observable(name='protons',
                               edges='proton_dist',
                               callback=lambda bs: bs.sources['proton_dist'].values())

        res = bs.bootstrap(100)
        # 5 samples; each bin is sum(range(5) + 1) == 15
        mean, sigma = res['protons'].mean(), res['protons'].sigma()
        self.assertTrue(np.all(mean - sigma < np.array([15, 15, 15])))
        self.assertTrue(np.all(mean + sigma > np.array([15, 15, 15])))
        np.testing.assert_array_equal(res['ratio'].mean(), np.array([1., 1., 1.]))


class Test_collector(TestCase):
    def test_mean(self):
        points = np.linspace(0, 10, num=50)
        print points
        c = Collector()
        [c.add([p]) for p in points]
        self.assertEqual(np.mean(points), c.mean()[0])
        self.assertLessEqual(np.std(points) * 0.98, c.sigma()[0])
        self.assertLessEqual(c.sigma()[0], np.std(points) * 1.02)
