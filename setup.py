from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='NlpToolkit-WordToVec',
    version='1.0.5',
    packages=['WordToVec'],
    url='https://github.com/StarlangSoftware/WordToVec-Py',
    license='',
    author='olcay',
    author_email='olcay.yildiz@ozyegin.edu.tr',
    description='Word2Vec Library',
    install_requires=['NlpToolkit-Corpus'],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
