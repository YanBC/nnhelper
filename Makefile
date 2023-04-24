all: pycmds cudacmds

pycmds:
	pip install --upgrade pip
	pip install --upgrade pipx
	pipx ensurepath
	pipx install --verbose .

cudacmds: device_query

device_query:
	make -C src/device_query all
