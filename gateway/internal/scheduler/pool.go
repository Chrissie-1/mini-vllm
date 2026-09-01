// Package scheduler owns worker selection and admission control.
//
// The gateway is the control plane: it decides *whether* a request runs and
// *where*, and it must stay responsive while workers are saturated. Two
// mechanisms do that work.
//
// Admission control is a buffered channel used as a semaphore. Capacity is
// finite and known, so an overloaded gateway rejects with 503 immediately
// instead of accumulating an unbounded queue of requests whose clients have
// already given up. Shedding load is a feature; timing out is not.
//
// Routing is least-in-flight rather than round-robin. Generation times vary by
// more than an order of magnitude with output length, so round-robin reliably
// parks short requests behind long ones on an unlucky worker. Least-in-flight
// tracks actual occupancy and is self-correcting.
package scheduler

import (
	"context"
	"errors"
	"log/slog"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/mini-vllm/gateway/internal/pb"
)

// ErrNoCapacity is returned when the admission queue is full. Maps to HTTP 503.
var ErrNoCapacity = errors.New("scheduler: no capacity")

// ErrNoWorkers is returned when every worker has failed its health check.
var ErrNoWorkers = errors.New("scheduler: no healthy workers")

// Worker is one Python inference process.
type Worker struct {
	Addr   string
	Client pb.InferenceClient

	conn     *grpc.ClientConn
	inflight atomic.Int64
	healthy  atomic.Bool
	// maxBatch is reported by the worker; used only for observability.
	maxBatch atomic.Int32
}

func (w *Worker) Inflight() int64 { return w.inflight.Load() }
func (w *Worker) Healthy() bool   { return w.healthy.Load() }
func (w *Worker) MaxBatch() int32 { return w.maxBatch.Load() }

// Pool is the set of workers plus the admission semaphore.
type Pool struct {
	workers []*Worker
	slots   chan struct{}
	log     *slog.Logger
}

// New dials every worker address. Dialling is lazy in gRPC, so a worker that is
// still loading its model does not block startup; the health loop will pick it
// up once it is serving.
func New(addrs []string, maxInflight int, log *slog.Logger) (*Pool, error) {
	if len(addrs) == 0 {
		return nil, errors.New("scheduler: no worker addresses configured")
	}
	if maxInflight <= 0 {
		maxInflight = 64
	}

	p := &Pool{
		slots: make(chan struct{}, maxInflight),
		log:   log,
	}
	for _, addr := range addrs {
		conn, err := grpc.NewClient(addr,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDefaultCallOptions(
				grpc.MaxCallRecvMsgSize(16*1024*1024),
				grpc.MaxCallSendMsgSize(16*1024*1024),
			),
		)
		if err != nil {
			p.Close()
			return nil, err
		}
		w := &Worker{Addr: addr, Client: pb.NewInferenceClient(conn), conn: conn}
		// Assume healthy until proven otherwise, so the first request does not
		// have to wait for a health tick.
		w.healthy.Store(true)
		p.workers = append(p.workers, w)
	}
	return p, nil
}

// Workers exposes the pool for metrics collection.
func (p *Pool) Workers() []*Worker { return p.workers }

// Capacity reports the admission limit and current occupancy.
func (p *Pool) Capacity() (used, total int) { return len(p.slots), cap(p.slots) }

// Acquire takes an admission slot. It never blocks: a full queue is a 503, not a
// wait, because a client that is already timing out gains nothing from queueing.
func (p *Pool) Acquire() (release func(), err error) {
	select {
	case p.slots <- struct{}{}:
		var once atomic.Bool
		return func() {
			if once.CompareAndSwap(false, true) {
				<-p.slots
			}
		}, nil
	default:
		return nil, ErrNoCapacity
	}
}

// Pick returns the healthy worker with the fewest in-flight requests and marks
// one more request against it. The returned function must be called when the
// request finishes.
func (p *Pool) Pick() (*Worker, func(), error) {
	var best *Worker
	var bestLoad int64
	for _, w := range p.workers {
		if !w.Healthy() {
			continue
		}
		if load := w.Inflight(); best == nil || load < bestLoad {
			best, bestLoad = w, load
		}
	}
	if best == nil {
		return nil, nil, ErrNoWorkers
	}

	best.inflight.Add(1)
	var once atomic.Bool
	return best, func() {
		if once.CompareAndSwap(false, true) {
			best.inflight.Add(-1)
		}
	}, nil
}

// HealthLoop polls every worker until ctx is cancelled. A worker that fails is
// taken out of rotation but keeps being polled, so it rejoins automatically
// once it recovers -- no restart or manual intervention needed.
func (p *Pool) HealthLoop(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	p.probeAll(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.probeAll(ctx)
		}
	}
}

func (p *Pool) probeAll(ctx context.Context) {
	for _, w := range p.workers {
		probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		resp, err := w.Client.Health(probeCtx, &pb.HealthRequest{})
		cancel()

		was := w.healthy.Load()
		now := err == nil && resp.GetReady()
		w.healthy.Store(now)
		if now {
			w.maxBatch.Store(resp.GetMaxBatchSize())
		}
		if was != now {
			p.log.Warn("worker health changed",
				"addr", w.Addr, "healthy", now, "err", err)
		}
	}
}

// Healthy counts workers currently in rotation.
func (p *Pool) Healthy() int {
	n := 0
	for _, w := range p.workers {
		if w.Healthy() {
			n++
		}
	}
	return n
}

// Close shuts every connection down.
func (p *Pool) Close() {
	for _, w := range p.workers {
		if w.conn != nil {
			_ = w.conn.Close()
		}
	}
}
