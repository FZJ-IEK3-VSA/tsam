test:
	pytest

sdist :
	python setup.py sdist

upload:
	twine upload dist/*

clean:
	rm dist/*

dist: sdist upload clean

