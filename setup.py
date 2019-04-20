from setuptools import setup

setup(
    name='nmt-cli',
    version='0.1.0',
    py_modules=['nmt_cli'],
    packages=['nmt', 'scripts'],
    entry_points='''
        [console_scripts]
        nmt=nmt_cli:cli
    ''',
)
