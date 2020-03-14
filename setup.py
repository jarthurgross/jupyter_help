from setuptools import setup

requires = [
        'matplotlib',
        'mayavi',
        'sympy',
        'numpy',
        'scipy',
        'colorcet',
        'hsluv',
         ]

setup(name='jupyter_helper',
      install_requires=requires,
      packages=['jupyter_helper'],
      package_dir={'': 'src'},
     )
