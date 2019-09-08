from setuptools import setup

setup(
    name='nmt-cli',
    version='0.12.4',
    py_modules=['nmt_cli'],
    packages=['nmt', 'cloud_runner'],
    entry_points='''
        [console_scripts]
        nmt=nmt_cli:cli
    ''',
)
