from setuptools import setup, find_packages

setup(
    name='fl_byzantine_library',
    version='0.1.0',
    description='A Federated Learning library with Byzantine-resilient aggregation and attack simulation.',
    author='Kerem Özfatura',
    author_email='aozfatura22@ku.edu.tr',
    url='https://github.com/keremozfatura/FL-Byzantine-Library',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        # Add other dependencies as needed
    ],
    python_requires='>=3.7',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'fl-byzantine=main:main',
        ],
    },
) 