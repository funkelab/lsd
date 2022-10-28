from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
from setuptools import find_packages

from pathlib import Path

this_directory = Path(__file__).parent

long_description = (this_directory / "README.md").read_text()

setup(
        name='lsds',
        version='0.1.1',
        description='Local Shape Descriptors.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/funkey/lsd',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        license='MIT',
        packages=find_packages(),
        ext_modules=[
            Extension(
                'lsd.post.merge_tree',
                sources=[
                    'lsd/post/merge_tree.pyx'
                ],
                extra_compile_args=['-O3'],
                language='c++')
        ],
        cmdclass={'build_ext': build_ext},
        install_requires=[
            "mahotas",
            "numpy",
            "scipy",
            "h5py",
            "scikit-image",
            "requests",
            "cython",
            "gunpowder",
        ]
)
