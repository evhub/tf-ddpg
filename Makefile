.PHONY: install
install: build
	pip install -Ue .

.PHONY: build
build:
	coconut ddpg-source ddpg --no-tco --strict --jobs sys

.PHONY: clean
clean:
	rm -rf ./ddpg
	-find . -name '*.pyc' -delete
	-find . -name '__pycache__' -delete
