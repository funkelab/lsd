from setuptools import setup
import subprocess

setup(
        name='lsd',
        version='0.1',
        description='Local Shape Descriptors.',
        url='https://github.com/funkey/lsd',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        license='MIT',
        packages=[
            'lsd',
            'lsd.gp',
        ],
        install_requires=[
            "numpy",
            "scipy",
            "h5py",
            "scikit-image",
            "requests",
            "gunpowder"
        ]
)
