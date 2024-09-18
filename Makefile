# This make command generates golang bindings using the FastAPI schema.
# NOTE: 'awk' command temporarily resolves a warning due to 'oapi-codegen' and 'OpenAPI'
# version mismatch (refer: https://github.com/deepmap/oapi-codegen/issues/373).
codegen:
	go run github.com/deepmap/oapi-codegen/v2/cmd/oapi-codegen@v2.2.0 \
		-package worker \
		-generate types,client,chi-server,spec \
		runner/openapi.yaml \
		| awk '!/WARNING/' > worker/runner.gen.go
