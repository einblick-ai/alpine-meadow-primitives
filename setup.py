import os
from setuptools import setup, find_packages


packages = find_packages(where='.')
with open('requirements.txt') as f:
    install_requires = [r.rstrip() for r in f.readlines()
                        if not r.startswith('#') and not r.startswith('git+')]


with open(os.path.join(
        os.path.dirname(__file__), 'alpine_meadow', 'primitives', '__init__.py')) as f:
    for line in f:
        if line.startswith('__version__ ='):
            _, _, version = line.partition('=')
            VERSION = version.strip(" \n'\"")
            break
    else:
        raise RuntimeError('unable to read the version from alpine_meadow/__init__.py')


setup(
    name='alpine-meadow-primitives',
    author='Zeyuan Shang',
    author_email='zs@einblick.ai',
    description='Einblick ML primitives',
    version=VERSION,
    packages=packages,
    install_requires=install_requires,
    include_package_data=True,
    python_requires='>=3.6.*',
    url='https://www.einblick.ai'
)
