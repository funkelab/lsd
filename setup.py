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
            'lsd.train.gp',
            'lsd.post.persistence',
        ],
        ext_modules=[
            Extension(
                'lsd.post.merge_tree',
                sources=[
                    'lsd/post/merge_tree.pyx'
                ],
                extra_compile_args=['-O3'],
                language='c++')
        ],
        cmdclass={'build_ext': build_ext}
)
