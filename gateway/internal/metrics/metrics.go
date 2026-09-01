// Package metrics holds the gateway's Prometheus collectors.
//
// These deliberately describe the *control plane* only -- admission, routing,
// and the cost of the gRPC hop. Anything about batching or the KV cache is the
// worker's to report, and duplicating it here would produce two numbers that
// drift apart.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Requests is labelled by outcome so shed load is distinguishable from
	// worker errors on the same graph.
	Requests = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "gateway_requests_total",
		Help: "HTTP requests by route and outcome.",
	}, []string{"route", "outcome"})

	Latency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "gateway_latency_seconds",
		Help:    "End-to-end latency as the client experiences it.",
		Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60},
	}, []string{"route"})

	// TTFB is the streaming equivalent of latency: the wait before anything
	// appears on screen.
	TTFB = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "gateway_time_to_first_byte_seconds",
		Help:    "Streaming requests: wait until the first token reaches the client.",
		Buckets: []float64{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5},
	})

	Tokens = promauto.NewCounter(prometheus.CounterOpts{
		Name: "gateway_completion_tokens_total",
		Help: "Completion tokens returned to clients.",
	})

	QueueDepth = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gateway_admission_slots_used",
		Help: "Admission slots currently held.",
	})

	QueueCapacity = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gateway_admission_slots_total",
		Help: "Admission slot limit.",
	})

	WorkerInflight = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "gateway_worker_inflight",
		Help: "Requests dispatched to each worker and not yet returned.",
	}, []string{"worker"})

	WorkerHealthy = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "gateway_worker_healthy",
		Help: "1 when the worker is in rotation, 0 when it has been ejected.",
	}, []string{"worker"})

	WorkersAvailable = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "gateway_workers_healthy",
		Help: "Healthy workers in the pool.",
	})
)
