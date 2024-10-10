
# Run `make reader-python stream=streamname`
reader-python:
	python read2pipe.py --stream $(stream) | ffplay -fflags nobuffer -probesize 32 -flags low_delay -

file-reader:
	python read2pipe.py --stream $(stream)

file-publisher:
	python file_publisher.py --stream $(stream) --local $(local)
