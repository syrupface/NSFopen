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

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
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
    description='Python module to open Nanosurf NID and NHF files',
    long_description=long_description_file,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=find_packages(
        include=['*'],
    ),
    package_data={'': package_data_files},
    include_package_data = True,
    zip_safe=False,
    install_requires=['matplotlib', 'numpy', 'scipy', 'pandas', 'h5py'],
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

