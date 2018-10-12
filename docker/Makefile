TAG="funkey/lsd:v0.4_no-conda"

.PHONY: default push
default:
	#-docker rmi -f $(TAG)
	-rm -rf lsd
	cp -rp ../lsd .
	cp -p ../requirements.txt .
	cp -p ../setup.py .
	docker build -t $(TAG) .

push: default
	docker push $(TAG)

test: default
	docker run --rm $(TAG)
