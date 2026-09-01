// Package pb holds the protobuf and gRPC bindings generated from
// proto/inference.proto.
//
// The generated files are produced at build time (see gateway/Dockerfile and
// `make proto`) rather than committed, so the wire contract can only ever come
// from the .proto file itself.
package pb
