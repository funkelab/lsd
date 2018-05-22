from setuptools import setup
import subprocess
try:
    import base_string as string_types
except ImportError:
    string_types = str

extras_require = {
}


dep_set = set()
for value in extras_require.values():
    if isinstance(value, string_types):
        dep_set.add(value)
    else:
        dep_set.update(value)

extras_require['full'] = list(dep_set)

subprocess.call('pip install git+https://github.com/funkey/augment#egg=augment'.split())

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
            'lsd.gunpowder',
        ],
        install_requires=[
            "numpy",
            "scipy",
            "h5py",
            "scikit-image",
            "requests",
            "gunpowder"
        ],
        extras_require=extras_require,
)
