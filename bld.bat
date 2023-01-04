"%PYTHON%" -m build
"%PYTHON%" -m pip install .
if errorlevel 1 exit 1