from distutils.core import setup

setup(
    name='tsam',
    version='0.9.2',
    author='Leander Kotzur',
    url='',
    packages = ['tsam',
                'tsam.utils',],
    install_requires = [
        "sklearn>=0.0",
        "pandas>=0.18.1",
        "numpy>=1.11.0",
		"pyomo>=5.3"
    ]
)

