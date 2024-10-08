
# Run `make reader-python stream=streamname`
reader-python:
	python read2pipe.py --stream $(stream) | ffplay -fflags nobuffer -probesize 32 -flags low_delay -
