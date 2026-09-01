package scheduler

import (
	"io"
	"log/slog"
	"sync"
	"testing"
)

func testPool(t *testing.T, addrs []string, maxInflight int) *Pool {
	t.Helper()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	p, err := New(addrs, maxInflight, log)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(p.Close)
	return p
}

func TestNewRejectsEmptyWorkerList(t *testing.T) {
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	if _, err := New(nil, 4, log); err == nil {
		t.Fatal("expected an error for an empty worker list")
	}
}

func TestAcquireShedsWhenFull(t *testing.T) {
	p := testPool(t, []string{"localhost:1"}, 2)

	first, err := p.Acquire()
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	if _, err = p.Acquire(); err != nil {
		t.Fatalf("second acquire: %v", err)
	}

	// The third must be shed immediately rather than blocking.
	if _, err = p.Acquire(); err != ErrNoCapacity {
		t.Fatalf("expected ErrNoCapacity, got %v", err)
	}

	// Releasing one slot lets exactly one more request in.
	first()
	if _, err = p.Acquire(); err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
}

func TestReleaseIsIdempotent(t *testing.T) {
	p := testPool(t, []string{"localhost:1"}, 1)
	release, err := p.Acquire()
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}

	// A double release must not free a slot that was never taken; otherwise a
	// deferred release on a retry path would inflate capacity without limit.
	release()
	release()

	if used, _ := p.Capacity(); used != 0 {
		t.Fatalf("expected 0 slots used, got %d", used)
	}
	if _, err = p.Acquire(); err != nil {
		t.Fatalf("acquire after double release: %v", err)
	}
	if _, err = p.Acquire(); err != ErrNoCapacity {
		t.Fatalf("capacity leaked: expected ErrNoCapacity, got %v", err)
	}
}

func TestPickChoosesLeastLoadedWorker(t *testing.T) {
	p := testPool(t, []string{"a:1", "b:1", "c:1"}, 16)
	p.Workers()[0].inflight.Store(5)
	p.Workers()[1].inflight.Store(2)
	p.Workers()[2].inflight.Store(9)

	worker, done, err := p.Pick()
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	defer done()
	if worker.Addr != "b:1" {
		t.Fatalf("expected the least loaded worker b:1, got %s", worker.Addr)
	}
	if got := worker.Inflight(); got != 3 {
		t.Fatalf("Pick must count the new request: want 3, got %d", got)
	}
}

func TestPickSkipsUnhealthyWorkers(t *testing.T) {
	p := testPool(t, []string{"a:1", "b:1"}, 16)
	// The idle worker is unhealthy, so the busy one must still be chosen.
	p.Workers()[0].healthy.Store(false)
	p.Workers()[1].inflight.Store(7)

	worker, done, err := p.Pick()
	if err != nil {
		t.Fatalf("Pick: %v", err)
	}
	defer done()
	if worker.Addr != "b:1" {
		t.Fatalf("expected healthy worker b:1, got %s", worker.Addr)
	}
}

func TestPickFailsWhenEveryWorkerIsDown(t *testing.T) {
	p := testPool(t, []string{"a:1", "b:1"}, 16)
	for _, w := range p.Workers() {
		w.healthy.Store(false)
	}
	if _, _, err := p.Pick(); err != ErrNoWorkers {
		t.Fatalf("expected ErrNoWorkers, got %v", err)
	}
	if n := p.Healthy(); n != 0 {
		t.Fatalf("expected 0 healthy workers, got %d", n)
	}
}

func TestPickSpreadsConcurrentLoad(t *testing.T) {
	const workers, requests = 4, 200
	addrs := []string{"a:1", "b:1", "c:1", "d:1"}
	p := testPool(t, addrs, requests)

	var wg sync.WaitGroup
	releases := make(chan func(), requests)
	for i := 0; i < requests; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, done, err := p.Pick()
			if err != nil {
				t.Errorf("Pick: %v", err)
				return
			}
			releases <- done
		}()
	}
	wg.Wait()
	close(releases)

	// Least-in-flight should hold every worker within one request of the mean.
	want := int64(requests / workers)
	for _, w := range p.Workers() {
		if diff := w.Inflight() - want; diff > 1 || diff < -1 {
			t.Errorf("worker %s has %d in flight, want ~%d", w.Addr, w.Inflight(), want)
		}
	}

	for done := range releases {
		done()
	}
	for _, w := range p.Workers() {
		if w.Inflight() != 0 {
			t.Errorf("worker %s leaked %d in-flight requests", w.Addr, w.Inflight())
		}
	}
}
