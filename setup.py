from setuptools import setup, find_packages

setup(
    name='run',
    version='0.1.0',
    py_modules=['run'],
    include_package_data=True,
    install_requires=[
        'Click',
        'pandas',
        'numpy',
        'statsmodels',
        'tqdm',
        'openpyxl',
        'xlsxwriter',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'run = run:main',
        ],
    },
    packages=find_packages(),
)
