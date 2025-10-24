package main

import (
	"errors"
	"fmt"
	"time"
)

// ======================
// Q1: What are the key features that make Go suitable for DevOps/SRE?
// ======================
/*
ANSWER:
1. Fast Compilation - Deploy quickly, iterate fast
2. Static Binary - No dependencies, easy deployment
3. Built-in Concurrency - Handle many operations efficiently
4. Strong Standard Library - HTTP, JSON, networking out of the box
5. Simple Syntax - Easy to read/maintain, good for ops tools
6. Cross-platform - Compile for any OS from one machine
7. Low Memory Footprint - Efficient for containerized workloads

Example: Simple HTTP health checker (production-ready in minutes)
*/

func healthCheckExample() {
	// Single binary, no dependencies, handles concurrency easily
	client := &http.Client{Timeout: 2 * time.Second}
	
	endpoints := []string{
		"https://api1.example.com/health",
		"https://api2.example.com/health",
	}
	
	// Concurrent checks with goroutines
	results := make(chan string, len(endpoints))
	for _, url := range endpoints {
		go func(u string) {
			if _, err := client.Get(u); err != nil {
				results <- fmt.Sprintf("%s: DOWN", u)
			} else {
				results <- fmt.Sprintf("%s: UP", u)
			}
		}(url)
	}
	
	// Collect results
	for range endpoints {
		fmt.Println(<-results)
	}
}

// =============================================================================
// Q2: How does Go's garbage collection work? Why important for long-running?
// =============================================================================
/*
ANSWER:
- Go uses concurrent, tri-color mark-and-sweep GC
- Runs concurrently with application (low STW pauses)
- Tunable with GOGC environment variable (default 100)
- Important for SRE because:
  * Low latency - minimal stop-the-world pauses (<1ms typical)
  * Predictable - no sudden long pauses like Java
  * Self-tuning - adapts to workload
  * Good for 24/7 services - won't cause unexpected outages

GC Phases:
1. Mark Setup (STW - brief)
2. Marking (concurrent)
3. Mark Termination (STW - brief)
4. Sweep (concurrent)

Memory Management Best Practices:
*/

func gcExample() {
	// BAD: Creates garbage every iteration
	badExample := func() {
		for i := 0; i < 1000; i++ {
			data := make([]byte, 1024) // New allocation each time
			_ = data
		}
	}
	
	// GOOD: Reuse allocations
	goodExample := func() {
		data := make([]byte, 1024) // Allocate once
		for i := 0; i < 1000; i++ {
			// Reuse the same slice
			_ = data
		}
	}
	
	// Use sync.Pool for frequently allocated objects
	var bufferPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 1024)
		},
	}
	
	poolExample := func() {
		buf := bufferPool.Get().([]byte)
		defer bufferPool.Put(buf) // Return to pool
		// Use buf...
	}
	
	_, _, _ = badExample, goodExample, poolExample
}

// =============================================================================
// Q3: Difference between var, :=, and const
// =============================================================================

func variableDeclarations() {
	// var - Explicit type, can be package or function scope
	var name string              // Zero value: ""
	var age int = 25            // Explicit initialization
	var isActive bool           // Zero value: false
	
	// := - Short declaration, type inference, ONLY in functions
	count := 10                 // Type inferred as int
	message := "Hello"          // Type inferred as string
	
	// Multiple declaration
	x, y := 1, 2
	
	// const - Compile-time constant, immutable
	const MaxRetries = 3
	const Timeout = 5 * time.Second
	const Pi = 3.14159
	
	// const can only be: string, bool, numeric types
	// const data := []int{1,2,3}  // ERROR: can't use := with const
	
	fmt.Println(name, age, isActive, count, message, x, y, MaxRetries, Timeout, Pi)
}

/*
KEY DIFFERENCES:

var:
- Can declare without initialization (gets zero value)
- Can be used at package level
- Can redeclare in different scope

:=:
- Short, concise (type inference)
- ONLY in functions (not package level)
- Left side must have at least one NEW variable
- Common in idiomatic Go

const:
- Compile-time constant
- Must be initialized
- Immutable
- Only basic types (string, bool, numeric)
*/

// =============================================================================
// Q4: Zero Values in Go
// =============================================================================

func zeroValuesExample() {
	// Every type has a zero value (no "undefined" or null pointer exceptions)
	
	var i int           // 0
	var f float64       // 0.0
	var b bool          // false
	var s string        // "" (empty string)
	var ptr *int        // nil
	var slice []int     // nil (but safe to use with len, cap, range)
	var m map[string]int // nil (must use make() before writing)
	var ch chan int     // nil
	
	// Zero values are SAFE and USABLE
	fmt.Println(len(slice))  // Works! Returns 0
	
	// This is valid:
	for _, v := range slice { // No panic on nil slice
		fmt.Println(v)
	}
	
	// But map needs initialization before writing:
	// m["key"] = 1  // PANIC: assignment to entry in nil map
	m = make(map[string]int) // Now safe
	m["key"] = 1
	
	fmt.Println(i, f, b, s, ptr, slice, m, ch)
}

/*
WHY ZERO VALUES MATTER:

1. No uninitialized variables - safe by default
2. Simple initialization - var x int works
3. Useful defaults - empty string, 0, false often what you want
4. sync.Mutex zero value is ready to use!

Example: Struct with usable zero value
*/

type Buffer struct {
	data []byte
	pos  int
}

func (b *Buffer) Write(p []byte) {
	// Even if Buffer is zero-valued, this works!
	b.data = append(b.data, p...) // append works on nil slice
	b.pos += len(p)
}

// =============================================================================
// Q5: Error Handling vs Exceptions
// =============================================================================

func errorHandlingExample() error {
	/*
	Go uses explicit error returns instead of exceptions
	
	Advantages:
	- Clear error flow (you see where errors can occur)
	- Forces you to handle errors
	- No hidden control flow
	- Better for reliability (SRE mindset)
	
	Disadvantages:
	- More verbose (if err != nil everywhere)
	- Can be tedious
	*/
	
	// Multi-return pattern
	data, err := readFile("config.json")
	if err != nil {
		return fmt.Errorf("failed to read config: %w", err) // Error wrapping
	}
	
	// Error wrapping preserves context
	if err := processData(data); err != nil {
		return fmt.Errorf("processing failed: %w", err)
	}
	
	return nil
}

func readFile(path string) ([]byte, error) {
	// Simulate file read
	if path == "" {
		return nil, errors.New("empty path")
	}
	return []byte("data"), nil
}

func processData(data []byte) error {
	if len(data) == 0 {
		return errors.New("empty data")
	}
	return nil
}

// Custom errors with context
type ValidationError struct {
	Field string
	Value interface{}
	Msg   string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation error: field=%s, value=%v, reason=%s",
		e.Field, e.Value, e.Msg)
}

// Error checking with errors.Is and errors.As
func errorCheckingExample() {
	var ErrNotFound = errors.New("not found")
	
	err := doSomething()
	
	// Check specific error
	if errors.Is(err, ErrNotFound) {
		fmt.Println("Handle not found case")
	}
	
	// Check error type
	var valErr *ValidationError
	if errors.As(err, &valErr) {
		fmt.Printf("Validation failed for field: %s\n", valErr.Field)
	}
}

func doSomething() error {
	return &ValidationError{Field: "email", Value: "", Msg: "required"}
}

/*
COMPARISON WITH EXCEPTIONS:

Go (Explicit):
data, err := readFile()
if err != nil {
    return err
}

Java/Python (Exceptions):
try {
    data = readFile()
} catch (IOException e) {
    // Handle
}

Go forces you to think about errors at every step - better for reliability!
*/

// =============================================================================
// Q6: init() function
// =============================================================================

var GlobalConfig map[string]string

// init() runs automatically before main()
func init() {
	fmt.Println("init() called - runs before main()")
	
	// Common uses:
	// 1. Initialize package-level variables
	GlobalConfig = make(map[string]string)
	GlobalConfig["env"] = "production"
	
	// 2. Register drivers/plugins
	// sql.Register("postgres", &postgresDriver{})
	
	// 3. Validate configuration
	// 4. Set up logging
}

// Multiple init() functions run in order they appear
func init() {
	fmt.Println("Second init() - also runs before main()")
}

/*
INIT() KEY POINTS:

1. Runs automatically before main()
2. Can have multiple init() functions
3. Runs in order of appearance
4. Runs after package-level variable initialization
5. Used for setup, registration, validation

Execution Order:
1. Package-level variable initialization
2. init() functions
3. main() function

WARNING: Don't overuse init() - makes code harder to test
*/

// =============================================================================
// Q7: Arrays vs Slices
// =============================================================================

func arraysVsSlices() {
	// ARRAY - Fixed size, value type
	var arr [3]int                    // Array of exactly 3 ints
	arr[0] = 1
	arr[1] = 2
	arr[2] = 3
	fmt.Printf("Array: %v, Length: %d\n", arr, len(arr))
	
	// Arrays are VALUE types (copied when passed)
	arr2 := arr  // Creates a COPY
	arr2[0] = 99
	fmt.Printf("arr: %v, arr2: %v (different!)\n", arr, arr2)
	
	// SLICE - Dynamic size, reference type
	slice := []int{1, 2, 3}          // No size specified = slice
	slice = append(slice, 4, 5)      // Can grow
	fmt.Printf("Slice: %v, Length: %d, Capacity: %d\n", 
		slice, len(slice), cap(slice))
	
	// Slices are REFERENCE types (share underlying array)
	slice2 := slice  // Points to same array
	slice2[0] = 99
	fmt.Printf("slice: %v, slice2: %v (same!)\n", slice, slice2)
	
	// Creating slices
	s1 := make([]int, 5)       // Length 5, capacity 5
	s2 := make([]int, 5, 10)   // Length 5, capacity 10
	s3 := []int{1, 2, 3}       // Literal
	
	// Slicing
	s4 := s3[1:3]  // [2, 3] - shares underlying array!
	
	fmt.Println(s1, s2, s3, s4)
}

/*
KEY DIFFERENCES:

ARRAYS:
- Fixed size (part of type: [3]int ≠ [4]int)
- Value type (copied)
- Rarely used in practice
- Use when size is known and unchanging

SLICES:
- Dynamic size
- Reference type (pointer to array)
- len() and cap()
- Can grow with append()
- Most common in Go

SLICE INTERNALS:
type slice struct {
    ptr *array  // Pointer to underlying array
    len int     // Current length
    cap int     // Capacity
}

GOTCHA: Slice modifications can affect other slices!
*/

// =============================================================================
// Q8: Type Conversions
// =============================================================================

func typeConversions() {
	// Go requires EXPLICIT conversions (no implicit)
	
	// Numeric conversions
	var i int = 42
	var f float64 = float64(i)  // Explicit conversion
	var u uint = uint(i)
	
	// String conversions
	s := string(i)  // WARNING: Converts to rune (character), not "42"!
	fmt.Printf("string(42) = %q (probably not what you want!)\n", s)
	
	// Correct way: use strconv
	import "strconv"
	s = strconv.Itoa(i)  // "42"
	i2, err := strconv.Atoi("42")
	if err != nil {
		fmt.Println("Conversion error:", err)
	}
	
	// Byte slice to string
	bytes := []byte("hello")
	str := string(bytes)
	bytes2 := []byte(str)
	
	// Type assertions (for interfaces)
	var val interface{} = "hello"
	
	// Safe type assertion with ok check
	if str, ok := val.(string); ok {
		fmt.Println("It's a string:", str)
	}
	
	// Unsafe type assertion (panics if wrong)
	// str := val.(string)  // Panics if val is not string
	
	// Type switch
	switch v := val.(type) {
	case string:
		fmt.Println("String:", v)
	case int:
		fmt.Println("Int:", v)
	default:
		fmt.Printf("Unknown type: %T\n", v)
	}
	
	fmt.Println(f, u, s, i2, str, bytes2)
}

/*
TYPE CONVERSION RULES:

1. Always explicit: float64(x), int(y)
2. Can lose precision: int(3.9) = 3
3. May panic: uint(-1) on some systems
4. string(123) converts to rune, use strconv instead
5. []byte ↔ string: efficient but makes copy

INTERFACE TYPE ASSERTIONS:
- val.(Type) - asserts val is Type (panics if wrong)
- val.(Type), ok - safe check (ok is false if wrong)
- switch v := val.(type) - check multiple types
*/

// =============================================================================
// Q9: Interfaces and Polymorphism
// =============================================================================

// Interface defines behavior (method set)
type Writer interface {
	Write([]byte) (int, error)
}

// Any type implementing Write() satisfies Writer interface
// No explicit "implements" keyword - implicit satisfaction!

type FileWriter struct {
	path string
}

func (f *FileWriter) Write(data []byte) (int, error) {
	fmt.Printf("Writing %d bytes to %s\n", len(data), f.path)
	return len(data), nil
}

type NetworkWriter struct {
	address string
}

func (n *NetworkWriter) Write(data []byte) (int, error) {
	fmt.Printf("Sending %d bytes to %s\n", len(data), n.address)
	return len(data), nil
}

// Polymorphism: function accepts any Writer
func writeData(w Writer, data []byte) error {
	_, err := w.Write(data)
	return err
}

func interfaceExample() {
	data := []byte("Hello, World!")
	
	// Both types implement Writer interface
	fw := &FileWriter{path: "/tmp/file.txt"}
	nw := &NetworkWriter{address: "192.168.1.1:8080"}
	
	writeData(fw, data)  // Uses FileWriter
	writeData(nw, data)  // Uses NetworkWriter
}

// Empty interface: interface{} accepts ANY type
func printAnything(v interface{}) {
	fmt.Printf("Type: %T, Value: %v\n", v, v)
}

// Common interfaces in stdlib
type errorInterface interface {
	Error() string
}

type Stringer interface {
	String() string
}

type Closer interface {
	Close() error
}

/*
INTERFACE KEY POINTS:

1. Implicit satisfaction - no "implements" keyword
2. Small interfaces better - io.Reader has 1 method!
3. Accept interfaces, return structs (common pattern)
4. interface{} accepts anything (now also: any)
5. nil interface ≠ interface with nil value

Duck typing: "If it walks like a duck and quacks like a duck..."
Go: "If it has a Write() method, it's a Writer"

BEST PRACTICES:
- Define interfaces where you USE them, not where you implement
- Keep interfaces small (1-3 methods)
- Common pattern: io.Reader, io.Writer, io.Closer
*/

// =============================================================================
// Q10: Struct Initialization Best Practices
// =============================================================================

type Config struct {
	Host    string
	Port    int
	Timeout time.Duration
	Enabled bool
}

func structInitialization() {
	// 1. ZERO VALUE (all fields get zero values)
	var c1 Config
	fmt.Printf("Zero value: %+v\n", c1)
	
	// 2. LITERAL with field names (BEST - explicit and clear)
	c2 := Config{
		Host:    "localhost",
		Port:    8080,
		Timeout: 5 * time.Second,
		Enabled: true,
	}
	
	// 3. LITERAL without field names (AVOID - fragile)
	// c3 := Config{"localhost", 8080, 5*time.Second, true}
	// BAD: breaks if struct fields reordered
	
	// 4. NEW (returns pointer)
	c4 := new(Config)  // Same as &Config{}
	c4.Host = "localhost"
	
	// 5. Constructor function (BEST PRACTICE for validation)
	c5, err := NewConfig("localhost", 8080)
	if err != nil {
		fmt.Println("Invalid config:", err)
	}
	
	// 6. Functional options pattern (for complex structs)
	c6 := NewConfigWithOptions(
		WithHost("localhost"),
		WithPort(8080),
		WithTimeout(5*time.Second),
	)
	
	fmt.Println(c2, c4, c5, c6)
}

// Constructor with validation
func NewConfig(host string, port int) (*Config, error) {
	if host == "" {
		return nil, errors.New("host cannot be empty")
	}
	if port <= 0 || port > 65535 {
		return nil, errors.New("invalid port")
	}
	
	return &Config{
		Host:    host,
		Port:    port,
		Timeout: 5 * time.Second,  // Default
		Enabled: true,              // Default
	}, nil
}

// Functional options pattern (for many optional fields)
type ConfigOption func(*Config)

func WithHost(host string) ConfigOption {
	return func(c *Config) {
		c.Host = host
	}
}

func WithPort(port int) ConfigOption {
	return func(c *Config) {
		c.Port = port
	}
}

func WithTimeout(timeout time.Duration) ConfigOption {
	return func(c *Config) {
		c.Timeout = timeout
	}
}

func NewConfigWithOptions(opts ...ConfigOption) *Config {
	// Start with defaults
	c := &Config{
		Host:    "localhost",
		Port:    8080,
		Timeout: 5 * time.Second,
		Enabled: true,
	}
	
	// Apply options
	for _, opt := range opts {
		opt(c)
	}
	
	return c
}

/*
STRUCT INITIALIZATION BEST PRACTICES:

1. Use field names in literals
   ✓ Config{Host: "localhost", Port: 8080}
   ✗ Config{"localhost", 8080}

2. Use constructor functions for validation
   func NewConfig(...) (*Config, error)

3. Set defaults in constructor

4. Use functional options for many optional fields
   NewConfig(WithHost(...), WithPort(...))

5. Embed structs for composition
   type Server struct {
       Config
       router *Router
   }

6. Use pointers for large structs to avoid copying

7. Make zero value useful when possible
   sync.Mutex works with zero value!
*/

func main() {
	fmt.Println("=== Go Fundamentals for SRE Interview ===\n")
	
	// Uncomment to run examples:
	// healthCheckExample()
	// gcExample()
	variableDeclarations()
	zeroValuesExample()
	_ = errorHandlingExample()
	errorCheckingExample()
	arraysVsSlices()
	typeConversions()
	interfaceExample()
	structInitialization()

}
