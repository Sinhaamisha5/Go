# Go Concurrency & Parallelism - Interview Q&A

## Section 2: Concurrency and Parallelism (15 Questions)

---

## Q11: What is the difference between concurrency and parallelism in Go?

### Answer:

**Concurrency** = Dealing with multiple things at once (structure)  
**Parallelism** = Doing multiple things at once (execution)

### Analogy:
- **Concurrency**: One chef managing multiple dishes (switching between tasks)
- **Parallelism**: Multiple chefs cooking different dishes simultaneously

### Key Differences:

| Aspect | Concurrency | Parallelism |
|--------|-------------|-------------|
| Definition | Task composition | Task execution |
| CPU Cores | Can work on 1 core | Requires multiple cores |
| Focus | Structure/Design | Performance |
| Example | Event loop | Multi-threaded computation |

### Code Example:

```go
// Concurrent: Structure allows handling multiple tasks
func apiGateway() {
    userCh := make(chan string)
    orderCh := make(chan string)
    
    // These CAN run in parallel if multiple cores available
    go func() { userCh <- callUserService() }()
    go func() { orderCh <- callOrderService() }()
    
    user := <-userCh
    order := <-orderCh
    // Collected results concurrently (and possibly in parallel)
}
```

### Real-World Use:

**Web Server Handling 1000 Requests:**
- **Concurrent**: Server has goroutines for each request (structure)
- **Parallel**: If 8 CPU cores, 8 requests execute simultaneously (execution)

**Result**: On 1 core = concurrent but not parallel (context switching)  
On 8 cores = both concurrent AND parallel

---

## Q12: How do goroutines differ from OS threads?

### Answer:

| Feature | Goroutines | OS Threads |
|---------|-----------|------------|
| Stack Size | ~2KB (grows dynamically) | ~1-2MB (fixed) |
| Creation Cost | Cheap (~few μs) | Expensive (~few ms) |
| Switching Cost | Low (user space) | High (kernel space) |
| Management | Go runtime | OS kernel |
| Scheduling | M:N cooperative | 1:1 preemptive |
| Max Practical | Millions | Thousands |

### Code Example:

```go
func goroutineExample() {
    // Can easily create 100,000 goroutines
    for i := 0; i < 100000; i++ {
        go func(id int) {
            time.Sleep(time.Second)
            // Only uses ~2KB per goroutine
        }(i)
    }
    // Total memory: ~200MB for 100k goroutines
    // With OS threads: Would need 100-200GB! ❌
}
```

### M:N Scheduling Model:

```
M Goroutines → N OS Threads
    
Go Runtime Scheduler
├── G: Goroutine (user code)
├── M: Machine (OS thread)
└── P: Processor (execution context)

Example: 100,000 goroutines → 8 OS threads (on 8-core CPU)
```

### Real-World Use:

**Web Server:**
```go
func handleConnections() {
    for {
        conn, _ := listener.Accept()
        // Each connection gets its own goroutine
        go handleConnection(conn)
    }
}
// Can handle 100k+ concurrent connections easily!
```

**Why This Matters for SRE:**
- Handle massive concurrent loads (C10k problem solved)
- Lower memory footprint = cost savings
- Better resource utilization

---

## Q13: What is a race condition, and how can it be detected in Go?

### Answer:

**Race Condition**: When multiple goroutines access shared data concurrently, and at least one is writing, without proper synchronization.

### Detection:

```bash
# Run with race detector
go run -race program.go
go test -race
go build -race
```

### Example of Race Condition:

```go
// BAD: Race condition
var counter int

func increment() {
    for i := 0; i < 1000; i++ {
        go func() {
            counter++ // UNSAFE! Read-Modify-Write is not atomic
        }()
    }
}
// Result: counter might be 723 instead of 1000 (data race)
```

### How to Fix:

#### Solution 1: Mutex
```go
var counter int
var mu sync.Mutex

func increment() {
    for i := 0; i < 1000; i++ {
        go func() {
            mu.Lock()
            counter++
            mu.Unlock()
        }()
    }
}
// Result: counter = 1000 ✓
```

#### Solution 2: Atomic Operations
```go
var counter int64

func increment() {
    for i := 0; i < 1000; i++ {
        go func() {
            atomic.AddInt64(&counter, 1)
        }()
    }
}
// Faster than mutex for simple operations
```

#### Solution 3: Channels
```go
func increment() {
    counterCh := make(chan int)
    
    // Single goroutine owns the counter
    go func() {
        counter := 0
        for range counterCh {
            counter++
        }
    }()
    
    for i := 0; i < 1000; i++ {
        go func() {
            counterCh <- 1
        }()
    }
}
```

### Real-World Impact:

**Metrics System:**
```go
type MetricsCollector struct {
    mu           sync.RWMutex
    requestCount int
    errorCount   int
}

func (m *MetricsCollector) RecordRequest() {
    m.mu.Lock()
    m.requestCount++
    m.mu.Unlock()
}

func (m *MetricsCollector) GetMetrics() (int, int) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    return m.requestCount, m.errorCount
}
```

**Without synchronization**: Corrupted metrics, wrong dashboards, bad decisions!

---

## Q14: Explain how channels work in Go.

### Answer:

**Channels** are typed conduits for goroutine communication.

### Core Operations:

```go
ch := make(chan int)    // Create channel
ch <- 42                // Send
value := <-ch           // Receive
close(ch)               // Close
```

### How Channels Work Internally:

```
Channel = Queue + Synchronization

[Sender] ---> [Channel Buffer] ---> [Receiver]
              (FIFO Queue)
              
Unbuffered: Sender blocks until receiver ready
Buffered: Sender blocks only when buffer full
```

### Example:

```go
func channelExample() {
    ch := make(chan string)
    
    // Sender goroutine
    go func() {
        ch <- "Hello"
        fmt.Println("Sent!")
    }()
    
    // Receiver
    msg := <-ch
    fmt.Println("Received:", msg)
}
```

### Channel Types:

#### 1. Unbuffered Channel
```go
ch := make(chan int)
// Synchronous: Sender waits for receiver
```

#### 2. Buffered Channel
```go
ch := make(chan int, 10)
// Asynchronous up to buffer size
```

#### 3. Directional Channels
```go
func sender(ch chan<- int) {   // Send-only
    ch <- 1
}

func receiver(ch <-chan int) { // Receive-only
    val := <-ch
}
```

### Real-World Use: Log Processing Pipeline

```go
func logPipeline() {
    logs := make(chan string, 100)
    filtered := make(chan string, 100)
    
    // Stage 1: Collect logs
    go func() {
        for i := 0; i < 1000; i++ {
            logs <- fmt.Sprintf("Log %d", i)
        }
        close(logs)
    }()
    
    // Stage 2: Filter
    go func() {
        for log := range logs {
            if strings.Contains(log, "ERROR") {
                filtered <- log
            }
        }
        close(filtered)
    }()
    
    // Stage 3: Store
    for log := range filtered {
        storeLog(log)
    }
}
```

---

## Q15: What's the difference between buffered and unbuffered channels?

### Answer:

| Feature | Unbuffered | Buffered |
|---------|-----------|----------|
| Creation | `make(chan T)` | `make(chan T, size)` |
| Send Blocks | Until receiver ready | Until buffer full |
| Synchronization | Strong (rendezvous) | Weak (up to size) |
| Use Case | Signaling, coordination | Work queues, throughput |

### Visual Representation:

```
UNBUFFERED:
Sender   Channel   Receiver
  |        ||         |
  |------->||-------->|  (Immediate handoff)
  BLOCKS   SYNC      READY
  
BUFFERED (size=3):
Sender    Channel      Receiver
  |      [1][2][3]        |
  |----->  BUFFER  ------>|
  SENDS   (async)      RECEIVES
```

### Code Example:

```go
func compareChannels() {
    // UNBUFFERED - Synchronous
    unbuffered := make(chan int)
    go func() {
        fmt.Println("Sending...")
        unbuffered <- 1      // BLOCKS until received
        fmt.Println("Sent!") // Only prints after receive
    }()
    time.Sleep(100 * time.Millisecond)
    <-unbuffered
    
    // BUFFERED - Asynchronous
    buffered := make(chan int, 2)
    buffered <- 1
    buffered <- 2
    fmt.Println("Sent 2 items without blocking!")
    // No receiver yet, but didn't block
}
```

### When to Use Each:

#### Unbuffered:
```go
// 1. Signal/Event notification
done := make(chan struct{})
go work(done)
<-done // Wait for completion

// 2. Guaranteed delivery
result := make(chan Data)
go process(result)
data := <-result // Sender waits until we receive
```

#### Buffered:
```go
// 1. Work queue
jobs := make(chan Job, 100)
for i := 0; i < 10; i++ {
    go worker(jobs)
}

// 2. Rate limiting
limiter := make(chan struct{}, 5) // Max 5 concurrent
for _, task := range tasks {
    limiter <- struct{}{} // Blocks when 5 concurrent
    go func(t Task) {
        defer func() { <-limiter }()
        t.Execute()
    }(task)
}
```

### Real-World: Job Queue

```go
type JobQueue struct {
    jobs chan Job
}

func NewJobQueue(bufferSize int) *JobQueue {
    return &JobQueue{
        jobs: make(chan Job, bufferSize),
    }
}

func (q *JobQueue) Submit(job Job) error {
    select {
    case q.jobs <- job:
        return nil
    default:
        return errors.New("queue full")
    }
}

// Buffer prevents producers from blocking immediately
// Provides backpressure when system is overloaded
```

---

## Q16: How do you avoid goroutine leaks?

### Answer:

**Goroutine Leak**: A goroutine that never exits, causing memory leak.

### Common Causes:

1. **Blocked on channel** (no sender/receiver)
2. **Infinite loop** without exit condition
3. **No context cancellation**
4. **Waiting on condition that never happens**

### Examples of Leaks:

#### BAD Example 1: Blocked Channel
```go
func leakyFunction() {
    ch := make(chan int)
    go func() {
        val := <-ch // Blocks forever! No one sends
        fmt.Println(val)
    }()
    // Function returns, goroutine still blocked (LEAK)
}
```

#### BAD Example 2: No Timeout
```go
func leakyHTTP() {
    go func() {
        resp, _ := http.Get("http://slow-api.com")
        // If this hangs, goroutine leaks forever
        defer resp.Body.Close()
    }()
}
```

### How to Fix:

#### Solution 1: Use Context
```go
func noLeakFunction(ctx context.Context) {
    ch := make(chan int)
    go func() {
        select {
        case val := <-ch:
            fmt.Println(val)
        case <-ctx.Done():
            fmt.Println("Cancelled, exiting")
            return
        }
    }()
}

// Usage:
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()
noLeakFunction(ctx)
```

#### Solution 2: Buffered Channel
```go
func noLeakHTTP() {
    result := make(chan string, 1) // Buffered!
    
    go func() {
        resp, _ := http.Get("http://api.com")
        result <- resp.Status
        // Even if no one receives, this won't block
    }()
    
    select {
    case status := <-result:
        fmt.Println(status)
    case <-time.After(2 * time.Second):
        return // Timeout, but goroutine won't leak
    }
}
```

#### Solution 3: Done Channel
```go
type Worker struct {
    done chan struct{}
}

func (w *Worker) Start() {
    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()
        
        for {
            select {
            case <-ticker.C:
                w.doWork()
            case <-w.done:
                return // Proper cleanup
            }
        }
    }()
}

func (w *Worker) Stop() {
    close(w.done)
}
```

### Real-World: HTTP Request with Timeout

```go
func fetchWithTimeout(url string) ([]byte, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
    defer cancel() // Ensures cleanup
    
    resultCh := make(chan []byte, 1) // Buffered!
    errCh := make(chan error, 1)
    
    go func() {
        req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
        resp, err := http.DefaultClient.Do(req)
        if err != nil {
            errCh <- err
            return
        }
        defer resp.Body.Close()
        
        data, err := io.ReadAll(resp.Body)
        if err != nil {
            errCh <- err
            return
        }
        resultCh <- data
    }()
    
    select {
    case data := <-resultCh:
        return data, nil
    case err := <-errCh:
        return nil, err
    case <-ctx.Done():
        return nil, ctx.Err()
    }
}
```

### How to Detect Leaks:

```go
// Monitor goroutine count
fmt.Println("Goroutines:", runtime.NumGoroutine())

// Use pprof
import _ "net/http/pprof"
// Visit: http://localhost:6060/debug/pprof/goroutine
```

---

## Q17: What does the select statement do in Go?

### Answer:

**select** multiplexes multiple channel operations. It waits on multiple channels and executes the first case that's ready.

### Syntax:

```go
select {
case val := <-ch1:
    // Received from ch1
case ch2 <- value:
    // Sent to ch2
case <-time.After(1 * time.Second):
    // Timeout
default:
    // Non-blocking, executes if no case ready
}
```

### Key Properties:

1. **Blocks** until one case can proceed
2. If **multiple cases** ready, chooses **randomly**
3. **default** makes it non-blocking
4. Empty select `select{}` blocks forever

### Examples:

#### Example 1: Timeout Pattern
```go
func requestWithTimeout() {
    result := make(chan string, 1)
    
    go func() {
        time.Sleep(2 * time.Second)
        result <- "data"
    }()
    
    select {
    case data := <-result:
        fmt.Println("Got:", data)
    case <-time.After(1 * time.Second):
        fmt.Println("Timeout!")
    }
}
```

#### Example 2: Multiple Event Sources
```go
func eventLoop() {
    ticker := time.NewTicker(time.Second)
    interrupt := make(chan os.Signal, 1)
    signal.Notify(interrupt, syscall.SIGINT)
    
    for {
        select {
        case <-ticker.C:
            fmt.Println("Tick")
        case <-interrupt:
            fmt.Println("Interrupted!")
            return
        }
    }
}
```

#### Example 3: Non-blocking Receive
```go
func tryReceive(ch <-chan int) {
    select {
    case val := <-ch:
        fmt.Println("Received:", val)
    default:
        fmt.Println("No data available")
    }
}
```

#### Example 4: Fan-In (Multiple Producers)
```go
func fanIn(ch1, ch2 <-chan string) <-chan string {
    out := make(chan string)
    go func() {
        for {
            select {
            case msg := <-ch1:
                out <- msg
            case msg := <-ch2:
                out <- msg
            }
        }
    }()
    return out
}
```

### Real-World: Service with Graceful Shutdown

```go
type Service struct {
    stopCh   chan struct{}
    dataCh   chan Data
    tickerCh <-chan time.Time
}

func (s *Service) Run() {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case data := <-s.dataCh:
            s.processData(data)
            
        case <-ticker.C:
            s.healthCheck()
            
        case <-s.stopCh:
            fmt.Println("Shutting down gracefully...")
            s.cleanup()
            return
        }
    }
}

func (s *Service) Stop() {
    close(s.stopCh)
}
```

### Common Patterns:

```go
// 1. Quit channel
select {
case <-quit:
    return
default:
    // Continue
}

// 2. Timeout on send
select {
case ch <- value:
    // Sent successfully
case <-time.After(time.Second):
    // Send timed out
}

// 3. First response wins
select {
case r := <-api1:
    return r
case r := <-api2:
    return r
}
```

---

## Q18: How can you implement worker pools using goroutines and channels?

### Answer:

**Worker Pool**: Fixed number of goroutines processing jobs from a shared queue.

### Benefits:
- **Limit concurrency** (prevent resource exhaustion)
- **Better resource utilization**
- **Backpressure handling**
- **Controlled parallelism**

### Basic Implementation:

```go
type Job struct {
    ID   int
    Data string
}

type Result struct {
    Job    Job
    Output string
}

func workerPool(numWorkers int, jobs <-chan Job, results chan<- Result) {
    var wg sync.WaitGroup
    
    // Create workers
    for w := 1; w <= numWorkers; w++ {
        wg.Add(1)
        go worker(w, jobs, results, &wg)
    }
    
    // Wait for all workers to finish
    wg.Wait()
    close(results)
}

func worker(id int, jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
    defer wg.Done()
    
    for job := range jobs {
        fmt.Printf("Worker %d processing job %d\n", id, job.ID)
        
        // Simulate work
        time.Sleep(time.Second)
        
        results <- Result{
            Job:    job,
            Output: fmt.Sprintf("Processed by worker %d", id),
        }
    }
}

// Usage
func main() {
    jobs := make(chan Job, 100)
    results := make(chan Result, 100)
    
    // Start worker pool
    go workerPool(5, jobs, results)
    
    // Send jobs
    for i := 1; i <= 20; i++ {
        jobs <- Job{ID: i, Data: fmt.Sprintf("task-%d", i)}
    }
    close(jobs)
    
    // Collect results
    for i := 1; i <= 20; i++ {
        result := <-results
        fmt.Println(result.Output)
    }
}
```

### Production-Ready Worker Pool:

```go
type WorkerPool struct {
    workers    int
    jobs       chan Job
    results    chan Result
    ctx        context.Context
    cancel     context.CancelFunc
    wg         sync.WaitGroup
}

func NewWorkerPool(workers int, bufferSize int) *WorkerPool {
    ctx, cancel := context.WithCancel(context.Background())
    
    wp := &WorkerPool{
        workers: workers,
        jobs:    make(chan Job, bufferSize),
        results: make(chan Result, bufferSize),
        ctx:     ctx,
        cancel:  cancel,
    }
    
    wp.start()
    return wp
}

func (wp *WorkerPool) start() {
    for i := 0; i < wp.workers; i++ {
        wp.wg.Add(1)
        go wp.worker(i)
    }
}

func (wp *WorkerPool) worker(id int) {
    defer wp.wg.Done()
    
    for {
        select {
        case job, ok := <-wp.jobs:
            if !ok {
                return // Channel closed
            }
            
            result := wp.processJob(job)
            
            select {
            case wp.results <- result:
            case <-wp.ctx.Done():
                return
            }
            
        case <-wp.ctx.Done():
            return
        }
    }
}

func (wp *WorkerPool) processJob(job Job) Result {
    // Process job
    time.Sleep(time.Second)
    return Result{Job: job, Output: "done"}
}

func (wp *WorkerPool) Submit(job Job) error {
    select {
    case wp.jobs <- job:
        return nil
    case <-wp.ctx.Done():
        return errors.New("pool is shutting down")
    }
}

func (wp *WorkerPool) Results() <-chan Result {
    return wp.results
}

func (wp *WorkerPool) Shutdown() {
    close(wp.jobs)      // No more jobs
    wp.cancel()         // Cancel context
    wp.wg.Wait()        // Wait for workers
    close(wp.results)   // Close results
}
```

### Real-World Use Cases:

#### 1. Image Processing
```go
func processImages(imageURLs []string) {
    pool := NewWorkerPool(10, 100)
    defer pool.Shutdown()
    
    // Submit jobs
    for _, url := range imageURLs {
        pool.Submit(Job{Data: url})
    }
    
    // Collect results
    for range imageURLs {
        result := <-pool.Results()
        fmt.Println("Processed:", result.Output)
    }
}
```

#### 2. API Rate Limiting
```go
type APIWorkerPool struct {
    *WorkerPool
    rateLimiter *rate.Limiter
}

func (p *APIWorkerPool) worker(id int) {
    defer p.wg.Done()
    
    for job := range p.jobs {
        // Wait for rate limiter
        p.rateLimiter.Wait(p.ctx)
        
        // Process API call
        p.callAPI(job)
    }
}
```

#### 3. Database Batch Processor
```go
func processDatabaseRecords(records []Record) {
    pool := NewWorkerPool(5, 50)
    defer pool.Shutdown()
    
    for _, record := range records {
        pool.Submit(Job{Data: record.ID})
    }
}
```

---

## Q19: How do you use sync.WaitGroup and sync.Mutex in concurrent programs?

### Answer:

### sync.WaitGroup

**Purpose**: Wait for a collection of goroutines to finish.

**Methods**:
- `Add(n)`: Add n goroutines to wait for
- `Done()`: Decrement counter (goroutine finished)
- `Wait()`: Block until counter reaches 0

#### Basic Example:
```go
func waitGroupExample() {
    var wg sync.WaitGroup
    
    for i := 1; i <= 3; i++ {
        wg.Add(1) // Add before starting goroutine
        
        go func(id int) {
            defer wg.Done() // Always use defer!
            
            fmt.Printf("Goroutine %d working\n", id)
            time.Sleep(time.Second)
        }(i)
    }
    
    wg.Wait() // Block until all Done()
    fmt.Println("All goroutines finished")
}
```

#### Real-World: Parallel Database Queries
```go
func fetchMultipleUsers(ids []int) []User {
    var wg sync.WaitGroup
    var mu sync.Mutex
    users := make([]User, 0, len(ids))
    
    for _, id := range ids {
        wg.Add(1)
        
        go func(userID int) {
            defer wg.Done()
            
            user := queryDatabase(userID)
            
            mu.Lock()
            users = append(users, user) // Protect shared slice
            mu.Unlock()
        }(id)
    }
    
    wg.Wait()
    return users
}
```

### sync.Mutex

**Purpose**: Mutual exclusion - protect shared data from concurrent access.

**Methods**:
- `Lock()`: Acquire lock (blocks if already locked)
- `Unlock()`: Release lock
- `RWMutex`: Read-Write mutex (multiple readers OR one writer)

#### Basic Example:
```go
type SafeCounter struct {
    mu    sync.Mutex
    count int
}

func (c *SafeCounter) Inc() {
    c.mu.Lock()
    c.count++
    c.mu.Unlock()
}

func (c *SafeCounter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.count
}
```

#### RWMutex Example:
```go
type Cache struct {
    mu    sync.RWMutex
    items map[string]string
}

func (c *Cache) Get(key string) (string, bool) {
    c.mu.RLock()         // Multiple readers allowed
    defer c.mu.RUnlock()
    
    val, ok := c.items[key]
    return val, ok
}

func (c *Cache) Set(key, value string) {
    c.mu.Lock()          // Exclusive lock for writes
    defer c.mu.Unlock()
    
    c.items[key] = value
}
```

### Real-World Examples:

#### 1. Metrics Collector
```go
type MetricsCollector struct {
    mu           sync.RWMutex
    requests     int
    errors       int
    responseTime time.Duration
}

func (m *MetricsCollector) RecordRequest(duration time.Duration, err error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    m.requests++
    if err != nil {
        m.errors++
    }
    m.responseTime += duration
}

func (m *MetricsCollector) GetStats() (requests, errors int, avgTime time.Duration) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    
    requests = m.requests
    errors = m.errors
    if requests > 0 {
        avgTime = m.responseTime / time.Duration(requests)
    }
    return
}
```

#### 2. Connection Pool
```go
type ConnectionPool struct {
    mu          sync.Mutex
    connections []net.Conn
    maxSize     int
}

func (p *ConnectionPool) Get() (net.Conn, error) {
    p.mu.Lock()
    defer p.mu.Unlock()
    
    if len(p.connections) == 0 {
        return nil, errors.New("no connections available")
    }
    
    conn := p.connections[len(p.connections)-1]
    p.connections = p.connections[:len(p.connections)-1]
    return conn, nil
}

func (p *ConnectionPool) Put(conn net.Conn) {
    p.mu.Lock()
    defer p.mu.Unlock()
    
    if len(p.connections) < p.maxSize {
        p.connections = append(p.connections, conn)
    } else {
        conn.Close() // Pool full, close connection
    }
}
```

### Common Patterns:

```go
// Pattern 1: Always use defer with Unlock
func (c *Counter) Inc() {
    c.mu.Lock()
    defer c.mu.Unlock() // Ensures unlock even if panic
    c.value++
}

// Pattern 2: Minimize critical section
func (c *Cache) Get(key string) value {
    c.mu.RLock()
    val := c.items[key]
    c.mu.RUnlock()
    
    // Don't hold lock during expensive operation
    return processValue(val)
}

// Pattern 3: WaitGroup with goroutine spawning
func processItems(items []Item) {
    var wg sync.WaitGroup
    
    for _, item := range items {
        wg.Add(1) // Before go statement!
        go func(i Item) {
            defer wg.Done()
            process(i)
        }(item)
    }
    
    wg.Wait()
}
```

### When to Use What:

| Use Case | Tool |
|----------|------|
| Wait for goroutines to finish | WaitGroup |
| Protect shared data | Mutex |
| More reads than writes | RWMutex |
| Simple counter | atomic package |
| Communication | Channels |

---

## Q20: What is the use of context.Context in concurrent Go applications?

### Answer:

**context.Context** carries deadlines, cancellation signals, and request-scoped values across API boundaries and goroutines.

### Core Functions:

```go
// Create contexts
ctx := context.Background()                              // Root context
ctx, cancel := context.WithCancel(parent)               // Manual cancellation
ctx, cancel := context.WithTimeout(parent, 5*time.Second)  // Timeout
ctx, cancel := context.WithDeadline(parent, time.Time)  // Absolute deadline
ctx = context.WithValue(parent, key, value)             // Store values
```

### Key Methods:

```go
ctx.Done() <-chan struct{}     // Returns channel closed when cancelled
ctx.Err() error