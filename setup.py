from setuptools import setup, find_packages

setup(
    name='cqws',
    version='1.0',
    description='Solver of Quantum Wells trough Finite Difference Method, with arbitrary potential',
    author='O. Ruiz-Cigarrillo',
    author_email='ruizsocar.1393@gmail.com',
    url='https://github.com/NanophotonIICOs/cqws-codes.git',
    packages=find_packages('cqws'),
    package_dir={'': 'cqws'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'numba',
        'scipy',
        'Ipython',
        'tabulate',
        'pandas',
        'h5py',
        'ipykernel',
    ],
)
