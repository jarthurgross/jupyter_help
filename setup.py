from setuptools import setup

requires = [
        'matplotlib',
        'sympy',
        'numpy',
         ]

setup(name='jupyter_helper',
      install_requires=requires,
      packages=['jupyter_helper'],
      package_dir={'': 'src'},
     )
