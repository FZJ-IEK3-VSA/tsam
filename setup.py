import os, setuptools
dir_path = os.path.dirname(os.path.realpath(__file__))
with open( os.path.join(dir_path,'requirements.txt') ) as f:
    required_packages = f.read().splitlines()

setuptools.setup(
    name='tsam',
    version='0.9.5',
    author='Leander Kotzur',
    url='',
    include_package_data=True, # includes all files in sdist that are tracked in git, additionall using the MANIFEST.in to exclude some of them
    packages = setuptools.find_packages(),
    py_modules = [],
    install_requires = required_packages,
    setup_requires = ['setuptools-git']
)
