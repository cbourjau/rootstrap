

.. code-block:: python

    from glob import glob
    from rootstrap import Bootstrapper, Collector

    files = glob("./*/AnalysisResults.root")

    bs = Bootstrapper()  # set other options here as well
    bs.register("proton_dist", path="sums.proton_dist")
    bs.register("pion_dist",  path="sums.pion_dist")
    bs.read_files(files)

    ratio_collector = Collector()
    for i in range(100):
        bs.draw()  # set up a new bootstrapped sample
	ratio_collector.add(bs.proton_dist.values / bs.pion_dist.values)

    print "Mean:", ratio_collector.mean
    print "Sigma:", ratio_collector.sigma  # assuming gaussian
    print "Edges:", bs.proton_dist.edges
