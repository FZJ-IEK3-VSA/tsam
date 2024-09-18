#!make
LOCAL_VENVS_DIR=~/.virtualenvs
PROJECT_NAME=tsam
PYTHON=python3.11
LOCAL_VENV_DIR := ${LOCAL_VENVS_DIR}/${PROJECT_NAME}


test:
	. ${LOCAL_VENV_DIR}/bin/activate; pytest

sdist:
	. ${LOCAL_VENV_DIR}/bin/activate; ${PYTHON} setup.py sdist

upload:
	twine upload dist/*

clean:
	rm dist/*

dist: sdist upload clean



setup_venv:
	mkdir -p  ${LOCAL_VENVS_DIR}
	${PYTHON} -m venv ${LOCAL_VENV_DIR}
	. ${LOCAL_VENV_DIR}/bin/activate; pip install -r requirements.txt; pip install -e .
