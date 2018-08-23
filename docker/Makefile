TAG="funkey/lsd:v0.4"

.PHONY: default push
default:
	#-docker rmi -f $(TAG)
	-rm -rf lsd
	cp -r ../lsd .
	cp ../requirements.txt .
	cp ../setup.py .
	docker build -t $(TAG) .

push: default
	docker push $(TAG)

test: default
	docker run --rm $(TAG)
