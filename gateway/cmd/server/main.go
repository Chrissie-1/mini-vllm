// Command server is the mini-vLLM HTTP gateway.
//
// It terminates HTTP, applies admission control, and forwards to a pool of
// Python inference workers over gRPC. Splitting the two is not ceremony: the
// gateway is I/O-bound and must stay responsive under load, while a worker holds
// the GIL and a model and can only do one forward pass at a time. Keeping them
// in separate processes means slow inference cannot stall connection handling,
// and workers scale horizontally without touching the routing layer.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/mini-vllm/gateway/internal/metrics"
	"github.com/mini-vllm/gateway/internal/pb"
	"github.com/mini-vllm/gateway/internal/scheduler"
)

type config struct {
	addr        string
	workers     []string
	maxInflight int
	timeout     time.Duration
	healthEvery time.Duration
}

func loadConfig() config {
	cfg := config{}
	workers := flag.String("workers", env("WORKER_ADDRS", "localhost:50051"),
		"comma-separated worker gRPC addresses")
	flag.StringVar(&cfg.addr, "addr", env("LISTEN_ADDR", ":8080"), "HTTP listen address")
	flag.IntVar(&cfg.maxInflight, "max-inflight", envInt("MAX_INFLIGHT", 64),
		"admission limit across all workers")
	flag.DurationVar(&cfg.timeout, "timeout", envDuration("REQUEST_TIMEOUT", 120*time.Second),
		"per-request upstream timeout")
	flag.DurationVar(&cfg.healthEvery, "health-interval",
		envDuration("HEALTH_INTERVAL", 5*time.Second), "worker health poll interval")
	flag.Parse()

	for _, w := range strings.Split(*workers, ",") {
		if w = strings.TrimSpace(w); w != "" {
			cfg.workers = append(cfg.workers, w)
		}
	}
	return cfg
}

// ---------------------------------------------------------------------------
// Wire types. Deliberately OpenAI-shaped so existing clients work unchanged.
// ---------------------------------------------------------------------------

type completionRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int32   `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
	TopK        int32   `json:"top_k"`
	TopP        float32 `json:"top_p"`
	Stream      bool    `json:"stream"`
}

type usage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

type completionResponse struct {
	ID           string `json:"id"`
	Object       string `json:"object"`
	Created      int64  `json:"created"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
	Worker       string `json:"worker"`
	Usage        usage  `json:"usage"`
}

type server struct {
	pool *scheduler.Pool
	cfg  config
	log  *slog.Logger
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg := loadConfig()

	pool, err := scheduler.New(cfg.workers, cfg.maxInflight, log)
	if err != nil {
		log.Error("failed to build worker pool", "err", err)
		os.Exit(1)
	}
	defer pool.Close()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go pool.HealthLoop(ctx, cfg.healthEvery)

	metrics.QueueCapacity.Set(float64(cfg.maxInflight))
	srv := &server{pool: pool, cfg: cfg, log: log}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/completions", srv.handleCompletions)
	mux.HandleFunc("GET /health", srv.handleHealth)
	mux.HandleFunc("GET /ready", srv.handleReady)
	mux.Handle("GET /metrics", srv.instrumentedMetrics())

	httpSrv := &http.Server{
		Addr:    cfg.addr,
		Handler: mux,
		// No WriteTimeout: streaming responses are long-lived by design and a
		// write deadline would sever them mid-generation.
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	go func() {
		log.Info("gateway listening", "addr", cfg.addr, "workers", cfg.workers)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("http server failed", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Error("graceful shutdown failed", "err", err)
	}
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

func (s *server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	route := "/v1/completions"

	var req completionRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		s.fail(w, route, "bad_request", http.StatusBadRequest, "invalid JSON body")
		return
	}
	if strings.TrimSpace(req.Prompt) == "" {
		s.fail(w, route, "bad_request", http.StatusBadRequest, "prompt must not be empty")
		return
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = 50
	}

	release, err := s.pool.Acquire()
	if err != nil {
		// Shed rather than queue: the client is better served by a fast 503.
		s.fail(w, route, "shed", http.StatusServiceUnavailable,
			"gateway at capacity, retry shortly")
		return
	}
	defer release()
	s.syncGauges()

	worker, done, err := s.pool.Pick()
	if err != nil {
		s.fail(w, route, "no_workers", http.StatusServiceUnavailable,
			"no healthy inference workers")
		return
	}
	defer done()

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.timeout)
	defer cancel()

	grpcReq := &pb.GenerateRequest{
		Prompt:      req.Prompt,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopK:        req.TopK,
		TopP:        req.TopP,
		RequestId:   requestID(r),
	}

	if req.Stream {
		s.streamCompletion(ctx, w, worker, grpcReq, route, start)
		return
	}

	resp, err := worker.Client.Generate(ctx, grpcReq)
	if err != nil {
		s.failUpstream(w, route, err)
		return
	}

	metrics.Tokens.Add(float64(resp.GetCompletionTokens()))
	metrics.Requests.WithLabelValues(route, "ok").Inc()
	metrics.Latency.WithLabelValues(route).Observe(time.Since(start).Seconds())

	writeJSON(w, http.StatusOK, completionResponse{
		ID:           resp.GetRequestId(),
		Object:       "text_completion",
		Created:      time.Now().Unix(),
		Text:         resp.GetText(),
		FinishReason: resp.GetFinishReason(),
		Worker:       worker.Addr,
		Usage: usage{
			PromptTokens:     resp.GetPromptTokens(),
			CompletionTokens: resp.GetCompletionTokens(),
			TotalTokens:      resp.GetPromptTokens() + resp.GetCompletionTokens(),
		},
	})
}

// streamCompletion proxies the worker's token stream to the client as SSE.
//
// Every chunk is flushed explicitly. Without that, Go's buffering would hold
// tokens back and deliver the whole response at once -- which looks identical to
// a non-streaming endpoint and defeats the point.
func (s *server) streamCompletion(
	ctx context.Context,
	w http.ResponseWriter,
	worker *scheduler.Worker,
	req *pb.GenerateRequest,
	route string,
	start time.Time,
) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.fail(w, route, "error", http.StatusInternalServerError, "streaming unsupported")
		return
	}

	stream, err := worker.Client.GenerateStream(ctx, req)
	if err != nil {
		s.failUpstream(w, route, err)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	first := true
	tokens := 0
	for {
		tok, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			// Headers are already sent, so the error has to travel in-band.
			s.log.Warn("stream aborted", "worker", worker.Addr, "err", err)
			fmt.Fprintf(w, "data: {\"error\":%q}\n\n", status.Convert(err).Message())
			flusher.Flush()
			metrics.Requests.WithLabelValues(route, "upstream_error").Inc()
			return
		}
		if tok.GetDone() {
			break
		}
		if first {
			metrics.TTFB.Observe(time.Since(start).Seconds())
			first = false
		}
		tokens++
		payload, _ := json.Marshal(map[string]any{
			"id":   req.GetRequestId(),
			"text": tok.GetText(),
		})
		fmt.Fprintf(w, "data: %s\n\n", payload)
		flusher.Flush()
	}

	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()

	metrics.Tokens.Add(float64(tokens))
	metrics.Requests.WithLabelValues(route, "ok").Inc()
	metrics.Latency.WithLabelValues(route).Observe(time.Since(start).Seconds())
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	used, total := s.pool.Capacity()
	writeJSON(w, http.StatusOK, map[string]any{
		"status":          "ok",
		"workers_total":   len(s.pool.Workers()),
		"workers_healthy": s.pool.Healthy(),
		"slots_used":      used,
		"slots_total":     total,
	})
}

// handleReady is the load-balancer probe: the gateway is only useful when at
// least one worker can actually serve.
func (s *server) handleReady(w http.ResponseWriter, r *http.Request) {
	if s.pool.Healthy() == 0 {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"ready": false})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ready": true})
}

func (s *server) instrumentedMetrics() http.Handler {
	handler := promhttp.Handler()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.syncGauges()
		handler.ServeHTTP(w, r)
	})
}

// syncGauges refreshes point-in-time values that have no natural event to hook.
func (s *server) syncGauges() {
	used, _ := s.pool.Capacity()
	metrics.QueueDepth.Set(float64(used))
	metrics.WorkersAvailable.Set(float64(s.pool.Healthy()))
	for _, worker := range s.pool.Workers() {
		metrics.WorkerInflight.WithLabelValues(worker.Addr).Set(float64(worker.Inflight()))
		healthy := 0.0
		if worker.Healthy() {
			healthy = 1.0
		}
		metrics.WorkerHealthy.WithLabelValues(worker.Addr).Set(healthy)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func (s *server) fail(w http.ResponseWriter, route, outcome string, code int, msg string) {
	metrics.Requests.WithLabelValues(route, outcome).Inc()
	writeJSON(w, code, map[string]string{"error": msg})
}

// failUpstream translates gRPC status codes into the HTTP codes a client can act
// on: back off (503), retry later (504), or fix the request (400).
func (s *server) failUpstream(w http.ResponseWriter, route string, err error) {
	st := status.Convert(err)
	switch st.Code() {
	case codes.ResourceExhausted:
		s.fail(w, route, "worker_shed", http.StatusServiceUnavailable, st.Message())
	case codes.DeadlineExceeded:
		s.fail(w, route, "timeout", http.StatusGatewayTimeout, "generation timed out")
	case codes.Unavailable:
		s.fail(w, route, "worker_down", http.StatusServiceUnavailable, "worker unavailable")
	case codes.InvalidArgument:
		s.fail(w, route, "bad_request", http.StatusBadRequest, st.Message())
	default:
		s.log.Error("upstream failure", "code", st.Code(), "msg", st.Message())
		s.fail(w, route, "error", http.StatusBadGateway, "inference failed")
	}
}

func writeJSON(w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(body)
}

func requestID(r *http.Request) string {
	if id := r.Header.Get("X-Request-Id"); id != "" {
		return id
	}
	return fmt.Sprintf("cmpl-%d", time.Now().UnixNano())
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	var out int
	if _, err := fmt.Sscanf(os.Getenv(key), "%d", &out); err == nil {
		return out
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return fallback
}
