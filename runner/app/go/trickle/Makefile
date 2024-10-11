
# Run `make reader stream=streamname`
subscriber:
	go run read2pipe.go trickle_subscriber.go trickle_publisher.go --stream $(stream) | ffplay -probesize 32 -fflags nobuffer -flags low_delay -

server:
	go run trickle_server.go

# Listens for a connection from MediaMTX
# Run `make subscriber-example stream=streamname`
subscriber-example:
	go run subscriber-example.go trickle_subscriber.go trickle_publisher.go --stream $(stream)

OS := $(shell uname)

# Set the file name depending on the OS
ifeq ($(OS), Darwin)
	SELECT_FILE := select_darwin.go
else ifeq ($(OS), Linux)
	SELECT_FILE := select_linux.go
else
	$(error Unsupported OS: $(OS))
endif

pubsub:
	go run pubsub-mediamtx.go rtmp2segment.go $(SELECT_FILE) trickle_publisher.go trickle_subscriber.go --out $(out)

publisher:
	go run publisher-mediamtx.go rtmp2segment.go $(SELECT_FILE) trickle_publisher.go

tester:
	go run trickle_tester.go file2segment.go $(SELECT_FILE) trickle_publisher.go --stream $(stream) --local $(local)
