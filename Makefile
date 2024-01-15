codegen:
	go install github.com/deepmap/oapi-codegen/v2/cmd/oapi-codegen@latest
	oapi-codegen -package worker -generate types,client,chi-server,spec runner/openapi.json > worker/runner.gen.go