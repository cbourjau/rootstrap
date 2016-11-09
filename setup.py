from setuptools import setup

install_requires = ['numpy', 'root_numpy', 'rootpy']
tests_require = ['nose']

setup(
    name='rootstrap',
    version='0.0.1',
    description="Bootstrap a sample of histograms distributed over various root files",
    author='Christian Bourjau',
    author_email='christian.bourjau@cern.ch',
    packages=['rootstrap', 'rootstrap.tests'],
    long_description=open('README.rst').read(),
    url='https://github.com/cbourjau/rootstrap',
    keywords=['alice'],
    # scripts=glob('scripts/*'),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Development Status :: 3 - Alpha",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=install_requires,
    extras_require={'test': tests_require},
    test_suite='nose.collector',
)
