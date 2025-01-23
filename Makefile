build:
	docker build -t svdvoort/brats2020_scan20:cuda113 .

inspect:
	dive svdvoort/brats2020_scan20:cuda113

run:
	docker run -it svdvoort/brats2020_scan20:cuda113

push:
	docker push svdvoort/brats2020_scan20:cuda113
