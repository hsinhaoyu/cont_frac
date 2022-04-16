tangle: org/cont_frac.org
	find src/ ! -name 'zz*.py' -type f -maxdepth 1 -exec rm -f {} +
	./tangle.sh org/cont_frac.org
	./tangle.sh org/tests.org
	./tangle.sh org/display.org
	yapf --in-place --recursive src/

test: tangle
	pytest -s src/

typing: tangle
	mypy src/

html: tangle
	./org2html.sh org/cont_frac.org
