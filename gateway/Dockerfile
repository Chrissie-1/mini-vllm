# Build stage: generates the gRPC bindings from the .proto, then compiles.
# Generating rather than committing them means the Go and Python sides cannot
# drift from the contract in proto/inference.proto.
FROM golang:1.23-bookworm AS build

RUN apt-get update \
 && apt-get install -y --no-install-recommends protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*

RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.35.2 \
 && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1

WORKDIR /src

# Dependencies first so edits to the source do not invalidate the module cache.
COPY gateway/go.mod gateway/go.sum ./gateway/
RUN cd gateway && go mod download

COPY proto ./proto
COPY gateway ./gateway

RUN protoc -I proto \
      --go_out=gateway/internal/pb --go_opt=paths=source_relative \
      --go-grpc_out=gateway/internal/pb --go-grpc_opt=paths=source_relative \
      proto/inference.proto

WORKDIR /src/gateway
RUN go vet ./... \
 && go test ./... \
 && CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/server

# Runtime stage: static binary on a distroless base, no shell, non-root.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/gateway /gateway
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/gateway"]
