"""
Async Task Queue for LLM Orchestration.

PRODUCTION SYSTEM FOR CONCURRENT USERS.

This module provides:
1. Background task processing for LLM calls
2. Request batching for efficiency
3. Priority queuing for important requests
4. Retry logic with exponential backoff
5. Circuit breaker for failing services
6. Result streaming for progressive UI updates

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────┐
│                         AsyncTaskQueue                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Submit      │─▶│ Priority    │─▶│ Batch       │─▶│ Worker      │    │
│  │ Task        │  │ Queue       │  │ Aggregator  │  │ Pool        │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                                                │              │
│         │                                                ▼              │
│         │                                         ┌─────────────┐      │
│         │                                         │ LLM Client  │      │
│         │                                         │ (with retry)│      │
│         │                                         └─────────────┘      │
│         │                                                │              │
│         ▼                                                ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Result Callbacks                              │   │
│  │  - Progressive UI updates                                        │   │
│  │  - Cache population                                              │   │
│  │  - Error handling                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

TRADE-OFFS:
- Batching: Improves throughput but adds latency (configurable batch_timeout)
- Workers: More workers = more concurrent calls but higher API cost
- Retry: Improves reliability but increases tail latency
- Queue size: Larger queue handles bursts but uses more memory

SCALING GUIDELINES:
- Single user: 1-2 workers, no batching
- 10 concurrent users: 4 workers, batch_size=5
- 100 concurrent users: 10 workers, batch_size=10
- 1000+ users: Use distributed queue (Redis/RabbitMQ)
"""
import asyncio
import logging
import queue
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Generic
import uuid

logger = logging.getLogger(__name__)


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0    # System tasks, immediate
    HIGH = 1        # User-initiated, interactive
    NORMAL = 2      # Background processing
    LOW = 3         # Batch jobs, reports


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRYING = auto()


@dataclass
class TaskResult:
    """Result from task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    start_time: float = 0
    end_time: float = 0
    retries: int = 0
    
    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0
    
    @property
    def is_success(self) -> bool:
        return self.status == TaskStatus.COMPLETED


@dataclass 
class Task:
    """Task definition for queue."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Execution settings
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Callbacks
    on_complete: Optional[Callable[[TaskResult], None]] = None
    on_progress: Optional[Callable[[str, float], None]] = None
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    
    # Internal
    _future: Optional[Future] = field(default=None, repr=False)
    _retries: int = 0
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value < other.priority.value


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()     # Normal operation
    OPEN = auto()       # Failing, reject requests
    HALF_OPEN = auto()  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker for failing services.
    
    Prevents cascading failures by temporarily rejecting
    requests when error rate exceeds threshold.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    
    TRADE-OFF: Fail-fast vs availability
    - Aggressive settings: Fast failure detection but may trigger on transient errors
    - Conservative settings: Higher availability but slower failure response
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_max_calls: Max calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._check_state_transition()
            return self._state
    
    def _check_state_transition(self):
        """Check if state should transition."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.recovery_timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if request can proceed."""
        with self._lock:
            self._check_state_transition()
            
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            
            return False  # OPEN
    
    def record_success(self):
        """Record successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker transitioning to CLOSED")
                self._state = CircuitState.CLOSED
            
            self._failure_count = 0
    
    def record_failure(self):
        """Record failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker transitioning to OPEN (half-open failure)")
                self._state = CircuitState.OPEN
            
            elif self._failure_count >= self.failure_threshold:
                logger.warning(f"Circuit breaker OPEN after {self._failure_count} failures")
                self._state = CircuitState.OPEN
    
    def get_status(self) -> Dict:
        """Get circuit breaker status."""
        return {
            "state": self.state.name,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure_time,
        }


# ============================================================================
# RETRY HANDLER
# ============================================================================

class RetryHandler:
    """
    Retry logic with exponential backoff.
    
    Handles transient failures gracefully.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to prevent thundering herd
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        import random
        
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Check if should retry based on error type."""
        if attempt >= self.max_retries:
            return False
        
        # Retry on transient errors
        error_str = str(error).lower()
        retryable = [
            "timeout",
            "rate limit",
            "429",
            "503",
            "502",
            "connection",
            "network",
        ]
        
        return any(r in error_str for r in retryable)


# ============================================================================
# ASYNC TASK QUEUE
# ============================================================================

class AsyncTaskQueue:
    """
    Production-grade async task queue for LLM operations.
    
    Features:
    - Priority-based task scheduling
    - Configurable worker pool
    - Automatic retry with exponential backoff
    - Circuit breaker for failing services
    - Progress callbacks for UI updates
    - Graceful shutdown
    
    Usage:
        queue = AsyncTaskQueue(
            worker_count=4,
            max_queue_size=1000
        )
        
        # Submit task
        task_id = queue.submit(
            task_type="generate_insight",
            payload={"query": "...", "context": {...}},
            on_complete=lambda result: update_ui(result)
        )
        
        # Wait for result
        result = queue.get_result(task_id, timeout=30)
        
        # Shutdown
        queue.shutdown()
    """
    
    def __init__(
        self,
        worker_count: int = 4,
        max_queue_size: int = 10000,
        circuit_breaker: CircuitBreaker = None,
        retry_handler: RetryHandler = None,
    ):
        """
        Initialize task queue.
        
        Args:
            worker_count: Number of worker threads
            max_queue_size: Maximum pending tasks
            circuit_breaker: Circuit breaker instance
            retry_handler: Retry handler instance
        """
        self.worker_count = worker_count
        self.max_queue_size = max_queue_size
        
        self._circuit = circuit_breaker or CircuitBreaker()
        self._retry = retry_handler or RetryHandler()
        
        # Task queue (priority queue)
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        
        # Task tracking
        self._tasks: Dict[str, Task] = {}
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.RLock()
        
        # Task handlers
        self._handlers: Dict[str, Callable] = {}
        
        # Worker pool
        self._executor = ThreadPoolExecutor(max_workers=worker_count)
        self._workers: List[threading.Thread] = []
        self._shutdown = threading.Event()
        
        # Stats
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "retried": 0,
            "rejected": 0,
            "avg_wait_time_ms": 0,
            "avg_exec_time_ms": 0,
        }
        
        # Start workers
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads."""
        for i in range(self.worker_count):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {self.worker_count} task workers")
    
    def _worker_loop(self):
        """Main worker loop."""
        while not self._shutdown.is_set():
            try:
                # Get task with timeout
                try:
                    priority, task = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Execute task
                self._execute_task(task)
                
            except Exception as e:
                logger.exception(f"Worker error: {e}")
    
    def _execute_task(self, task: Task):
        """Execute a single task."""
        # Check circuit breaker
        if not self._circuit.can_execute():
            self._handle_rejection(task, "Circuit breaker open")
            return
        
        # Get handler
        handler = self._handlers.get(task.task_type)
        if not handler:
            self._handle_failure(task, f"No handler for task type: {task.task_type}")
            return
        
        # Execute with retry
        start_time = time.time()
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Update status
            with self._lock:
                self._tasks[task.task_id] = task
            
            # Report progress
            if task.on_progress:
                task.on_progress(task.task_id, 0.0)
            
            # Execute handler
            output = handler(task.payload)
            
            # Success
            result.status = TaskStatus.COMPLETED
            result.result = output
            result.end_time = time.time()
            
            self._circuit.record_success()
            self._stats["completed"] += 1
            
            # Update avg exec time
            exec_time = result.duration_ms
            n = self._stats["completed"]
            self._stats["avg_exec_time_ms"] = (
                (self._stats["avg_exec_time_ms"] * (n-1) + exec_time) / n
            )
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Check retry
            if self._retry.should_retry(task._retries, e):
                task._retries += 1
                delay = self._retry.get_delay(task._retries)
                
                logger.info(f"Retrying task {task.task_id} in {delay:.1f}s (attempt {task._retries})")
                
                self._stats["retried"] += 1
                time.sleep(delay)
                
                # Re-queue
                self._queue.put((task.priority.value, task))
                return
            
            # Permanent failure
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.error_type = type(e).__name__
            result.end_time = time.time()
            result.retries = task._retries
            
            self._circuit.record_failure()
            self._stats["failed"] += 1
        
        # Store result
        with self._lock:
            self._results[task.task_id] = result
        
        # Callback
        if task.on_complete:
            try:
                task.on_complete(result)
            except Exception as e:
                logger.error(f"Callback error for task {task.task_id}: {e}")
        
        # Complete future if set
        if task._future:
            if result.is_success:
                task._future.set_result(result)
            else:
                task._future.set_exception(Exception(result.error))
    
    def _handle_rejection(self, task: Task, reason: str):
        """Handle rejected task."""
        self._stats["rejected"] += 1
        
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=reason,
            error_type="RejectedError"
        )
        
        with self._lock:
            self._results[task.task_id] = result
        
        if task.on_complete:
            task.on_complete(result)
    
    def _handle_failure(self, task: Task, error: str):
        """Handle failed task."""
        self._stats["failed"] += 1
        
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=error,
            error_type="HandlerError"
        )
        
        with self._lock:
            self._results[task.task_id] = result
        
        if task.on_complete:
            task.on_complete(result)
    
    def register_handler(
        self,
        task_type: str,
        handler: Callable[[Dict], Any]
    ):
        """
        Register a handler for a task type.
        
        Args:
            task_type: Task type identifier
            handler: Function that takes payload dict, returns result
        """
        self._handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def submit(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = 60.0,
        max_retries: int = 3,
        on_complete: Callable[[TaskResult], None] = None,
        on_progress: Callable[[str, float], None] = None,
        user_id: str = None,
    ) -> str:
        """
        Submit a task to the queue.
        
        Args:
            task_type: Type of task (must have registered handler)
            payload: Task payload dictionary
            priority: Task priority
            timeout: Execution timeout
            max_retries: Maximum retry attempts
            on_complete: Callback when task completes
            on_progress: Callback for progress updates
            user_id: Optional user identifier
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())[:8]
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout_seconds=timeout,
            max_retries=max_retries,
            on_complete=on_complete,
            on_progress=on_progress,
            user_id=user_id,
        )
        
        # Queue task
        try:
            self._queue.put_nowait((priority.value, task))
            self._stats["submitted"] += 1
            
            logger.debug(f"Submitted task {task_id} (type: {task_type}, priority: {priority.name})")
            
        except queue.Full:
            logger.error(f"Queue full, rejecting task {task_id}")
            self._handle_rejection(task, "Queue full")
        
        return task_id
    
    def submit_and_wait(
        self,
        task_type: str,
        payload: Dict[str, Any],
        timeout: float = 60.0,
        **kwargs
    ) -> TaskResult:
        """
        Submit task and wait for result.
        
        Args:
            task_type: Type of task
            payload: Task payload
            timeout: Wait timeout
            **kwargs: Additional submit arguments
            
        Returns:
            TaskResult
        """
        future = Future()
        
        task_id = str(uuid.uuid4())[:8]
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            timeout_seconds=timeout,
            _future=future,
            **kwargs
        )
        
        self._queue.put((task.priority.value, task))
        self._stats["submitted"] += 1
        
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__
            )
    
    def get_result(
        self,
        task_id: str,
        timeout: float = None
    ) -> Optional[TaskResult]:
        """
        Get result for a task.
        
        Args:
            task_id: Task ID
            timeout: Optional timeout to wait
            
        Returns:
            TaskResult if available
        """
        start = time.time()
        
        while True:
            with self._lock:
                if task_id in self._results:
                    return self._results[task_id]
            
            if timeout is None:
                return None
            
            if time.time() - start > timeout:
                return None
            
            time.sleep(0.1)
    
    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current status of a task."""
        with self._lock:
            if task_id in self._results:
                return self._results[task_id].status
            if task_id in self._tasks:
                return TaskStatus.RUNNING
        return TaskStatus.QUEUED
    
    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        # Note: Can only cancel before execution starts
        with self._lock:
            if task_id in self._tasks:
                return False  # Already running
            
            # Mark as cancelled
            self._results[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED
            )
            return True
    
    def get_stats(self) -> Dict:
        """Get queue statistics."""
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "worker_count": self.worker_count,
            "circuit_breaker": self._circuit.get_status(),
        }
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """
        Shutdown the task queue.
        
        Args:
            wait: Wait for pending tasks to complete
            timeout: Maximum time to wait
        """
        logger.info("Shutting down task queue...")
        
        self._shutdown.set()
        
        if wait:
            deadline = time.time() + timeout
            
            for worker in self._workers:
                remaining = deadline - time.time()
                if remaining > 0:
                    worker.join(timeout=remaining)
        
        self._executor.shutdown(wait=wait)
        
        logger.info("Task queue shutdown complete")


# ============================================================================
# LLM TASK HANDLERS
# ============================================================================

def create_llm_handler(
    llm_client: Any,
    orchestrator: Any = None,
) -> Callable:
    """
    Create an LLM task handler.
    
    Args:
        llm_client: LLM client instance
        orchestrator: Optional LLM orchestrator
        
    Returns:
        Handler function for LLM tasks
    """
    def handler(payload: Dict) -> Dict:
        task_type = payload.get("task_type", "generate")
        
        if task_type == "generate_insight" and orchestrator:
            return orchestrator.execute_tool(
                payload.get("tool", "generate_insights"),
                query=payload.get("query"),
                context=payload.get("context", {})
            )
        
        elif task_type == "raw_completion":
            response = llm_client.call(
                payload.get("prompt"),
                system_prompt=payload.get("system_prompt"),
                temperature=payload.get("temperature", 0.7),
                max_tokens=payload.get("max_tokens", 1000)
            )
            return {"response": response}
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    return handler


# ============================================================================
# BATCH AGGREGATOR
# ============================================================================

class BatchAggregator:
    """
    Aggregates similar tasks into batches for efficiency.
    
    Useful when multiple users submit similar queries -
    we can batch them into a single LLM call.
    
    TRADE-OFF: Latency vs Throughput
    - Small batch_timeout: Lower latency, less batching
    - Large batch_timeout: Higher batching, more latency
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout: float = 0.5,
        similarity_threshold: float = 0.9,
    ):
        """
        Initialize batch aggregator.
        
        Args:
            batch_size: Maximum batch size
            batch_timeout: Max time to wait for batch
            similarity_threshold: Threshold for grouping similar queries
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.similarity_threshold = similarity_threshold
        
        self._pending: Dict[str, List[Task]] = {}
        self._lock = threading.Lock()
    
    def add_task(self, task: Task, batch_key: str = None) -> Optional[List[Task]]:
        """
        Add task to batch.
        
        Args:
            task: Task to add
            batch_key: Optional key for batching similar tasks
            
        Returns:
            List of tasks if batch is ready, None otherwise
        """
        key = batch_key or task.task_type
        
        with self._lock:
            if key not in self._pending:
                self._pending[key] = []
            
            self._pending[key].append(task)
            
            if len(self._pending[key]) >= self.batch_size:
                batch = self._pending[key]
                self._pending[key] = []
                return batch
        
        return None
    
    def flush(self, batch_key: str = None) -> List[List[Task]]:
        """
        Flush all pending batches.
        
        Args:
            batch_key: Optional key to flush specific batch
            
        Returns:
            List of task batches
        """
        with self._lock:
            if batch_key:
                batch = self._pending.pop(batch_key, [])
                return [batch] if batch else []
            
            batches = list(self._pending.values())
            self._pending.clear()
            return [b for b in batches if b]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AsyncTaskQueue",
    "Task",
    "TaskResult",
    "TaskStatus",
    "TaskPriority",
    "CircuitBreaker",
    "RetryHandler",
    "BatchAggregator",
    "create_llm_handler",
]
