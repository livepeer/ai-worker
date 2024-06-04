# NOTE: The 'awk' command is a workaround for a warning causing syntax error. Temporary
# solution until issue (https://github.com/deepmap/oapi-codegen/issues/373) is resolved.
codegen:
	go install github.com/deepmap/oapi-codegen/v2/cmd/oapi-codegen@latest
	oapi-codegen -package worker -generate types,client,chi-server,spec runner/openapi.json > worker/runner.gen.go
	oapi-codegen -package worker -generate types,client,chi-server,spec runner/openapi.json | awk '!/WARNING/' > worker/runner.gen.go
