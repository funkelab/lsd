TAG="lsd:v0.8"

TMP_FILE:=$(shell mktemp).img

.PHONY: default push
default:
	-rm -rf lsd
	cp -rp ../lsd .
	cp -p ../requirements.txt .
	cp -p ../setup.py .
	sudo singularity build $(TMP_FILE) Singularity
	cp $(TMP_FILE) $(TAG).img
	sudo rm $(TMP_FILE)
