# coding: utf-8
# create by tongshiwei on 2019/6/25
import logging
from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-pep8>=1',
]

try:
    import mxnet

    mxnet_requires = []
except ModuleNotFoundError:
    mxnet_requires = ["mxnet"]
except Exception as e:
    mxnet_requires = []
    logging.error(e)

setup(
    name='CangJie',
    version='0.0.2',
    packages=find_packages(),
    python_requires='>=3.6',
    long_description='Refer to full documentation https://github.com/bigdata-ustc/XKT/blob/master/README.md'
                     ' for detailed information.',
    description='This project aims to '
                'provide multiple knowledge tracing models.',
    extras_require={
        'test': test_deps,
    },
    install_requires=mxnet_requires + [
        'tqdm',
        'fire',
        'longling==1.3.8',
        'gensim',
        'Pillow',
        'gluonnlp',
        'jieba',
        'numpy',
    ],  # And any other dependencies foo needs
    entry_points={
        "console_scripts": [
            "cangjie = CangJie.main:cli",
        ],
    },
)
