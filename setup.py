from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
import numpy as np

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
        ext_modules=[
            Extension(
                'lsd.replace_values',
                sources=['lsd/replace_values.pyx'],
                extra_compile_args=['-O3'],
		include_dirs=[np.get_include()],
                language='c++')
        ],
        cmdclass={'build_ext': build_ext}
)
