rem Builds NSFopen and uploads it to PyPI. Old builds are removed first (twine refuses files already on PyPI).
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
py -m pip install --upgrade build twine
py -m build || exit /b 1
py -m twine check dist/* || exit /b 1
py -m twine upload --repository pypi dist/*
