TAG="funkey/lsd:v0.3"

.PHONY: default push
default:
	#-docker rmi -f $(TAG)
	-rm -rf lsd
	cp -r ../lsd .
	cp ../requirements.txt .
	docker build -t $(TAG) .

push: default
	docker push $(TAG)

test: default
	docker run --rm $(TAG)
