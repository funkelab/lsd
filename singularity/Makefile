TAG="lsd:v0.8"

.PHONY: default push
default:
	-rm -rf lsd
	cp -rp ../lsd .
	cp -p ../requirements.txt .
	cp -p ../setup.py .
	sudo singularity build $(TAG).img Singularity

