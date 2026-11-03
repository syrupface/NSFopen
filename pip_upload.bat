py -m pip install --upgrade build
py -m build
py -m twine upload --repository pypi dist/*