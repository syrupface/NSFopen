# from setuptools import setup
# setup()
import os
from setuptools import setup, find_packages

def load_doc_file(readme_file_path: str) -> str:
    doc_str = ""
    with open(readme_file_path, "r", encoding="utf-8") as fh:
        doc_str = fh.read()
    return doc_str

def load_version(readme_file_path: str) -> str:
    ver_str = ""
    with open(readme_file_path, "r", encoding="utf-8") as fh:
        ver_str = fh.read()
    _, version = ver_str.split("=")
    version = version.replace('"', "").replace(" ", "")
    return version

SKIP_FILES = {'tokens.json', '.env'}              # credentials never go into the package
SKIP_SUFFIXES = ('.pyc', '.token')


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        if '__pycache__' in path or '.ipynb_checkpoints' in path:
            continue
        for filename in filenames:
            if filename in SKIP_FILES or filename.endswith(SKIP_SUFFIXES):
                continue
            paths.append(os.path.join('..', path, filename))
    return paths

package_data_files = []
package_data_files += package_files('NSFopen/example')

long_description_file = load_doc_file('README.md')

setup(
    name='NSFopen',
    version=load_version('NSFopen/_version.py'),
    author='Nanosurf AG',
    author_email='scripting@nanosurf.com',
    description='Python module to open and analyze Nanosurf NID and NHF files',
    long_description=long_description_file,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=find_packages(
        include=['NSFopen', 'NSFopen.*'],
    ),
    package_data={'': package_data_files},
    include_package_data = True,
    zip_safe=False,
    install_requires=['matplotlib>=3.6', 'numpy', 'scipy', 'pandas', 'h5py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows"
    ],
    entry_points={
        'console_scripts': [
            'NSFopen_help = NSFopen:help',
        ]
    },
    python_requires='>=3.9'
)

