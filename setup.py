from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

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
            'lsd.persistence',
        ],
        ext_modules=[
            Extension(
                'lsd.replace_values',
                sources=[
                    'lsd/replace_values.pyx'
                ],
                extra_compile_args=['-O3'],
                language='c++'),
            Extension(
                'lsd.connected_components',
                sources=[
                    'lsd/connected_components.pyx',
                    'lsd/find_connected_components.cpp'
                ],
                extra_compile_args=['-O3', '-std=c++11'],
                language='c++')
        ],
        cmdclass={'build_ext': build_ext}
)
