"""Job queue implementations for the scheduler."""

from __future__ import annotations

import heapq
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

from thrml_os.models.job import SamplingJob, JobPriority, JobStatus


class JobQueue:
    """Simple FIFO job queue.
    
    Thread-safe queue for pending jobs.
    """
    
    def __init__(self) -> None:
        self._queue: deque[SamplingJob] = deque()
        self._lock = threading.Lock()
    
    def push(self, job: SamplingJob) -> None:
        """Add a job to the queue."""
        with self._lock:
            self._queue.append(job)
    
    def pop(self) -> Optional[SamplingJob]:
        """Remove and return the next job, or None if empty."""
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None
    
    def peek(self) -> Optional[SamplingJob]:
        """Return the next job without removing it."""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None
    
    def remove(self, job_id: str) -> bool:
        """Remove a job by ID.
        
        Returns True if job was found and removed.
        """
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.id == job_id:
                    del self._queue[i]
                    return True
            return False
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)
    
    def __bool__(self) -> bool:
        return len(self) > 0
    
    def list_jobs(self) -> List[SamplingJob]:
        """Get list of all jobs in queue."""
        with self._lock:
            return list(self._queue)


@dataclass(order=True)
class PriorityItem:
    """Wrapper for priority queue ordering."""
    priority: int
    created_order: int  # Tie-breaker for same priority
    job: SamplingJob = field(compare=False)


class PriorityJobQueue:
    """Priority-based job queue.
    
    Jobs are ordered by priority level, then by submission time.
    Higher priority jobs (lower priority value) are processed first.
    """
    
    def __init__(self) -> None:
        self._heap: List[PriorityItem] = []
        self._lock = threading.Lock()
        self._counter = 0
    
    def push(self, job: SamplingJob) -> None:
        """Add a job to the queue."""
        with self._lock:
            item = PriorityItem(
                priority=job.priority.value,
                created_order=self._counter,
                job=job,
            )
            heapq.heappush(self._heap, item)
            self._counter += 1
    
    def pop(self) -> Optional[SamplingJob]:
        """Remove and return the highest priority job."""
        with self._lock:
            if self._heap:
                item = heapq.heappop(self._heap)
                return item.job
            return None
    
    def peek(self) -> Optional[SamplingJob]:
        """Return the highest priority job without removing it."""
        with self._lock:
            if self._heap:
                return self._heap[0].job
            return None
    
    def remove(self, job_id: str) -> bool:
        """Remove a job by ID.
        
        This is O(n) as we need to rebuild the heap.
        """
        with self._lock:
            for i, item in enumerate(self._heap):
                if item.job.id == job_id:
                    # Remove and rebuild heap
                    self._heap[i] = self._heap[-1]
                    self._heap.pop()
                    if self._heap:
                        heapq.heapify(self._heap)
                    return True
            return False
    
    def update_priority(self, job_id: str, new_priority: JobPriority) -> bool:
        """Update a job's priority.
        
        Returns True if job was found and updated.
        """
        with self._lock:
            for item in self._heap:
                if item.job.id == job_id:
                    item.job.priority = new_priority
                    item.priority = new_priority.value
                    heapq.heapify(self._heap)
                    return True
            return False
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)
    
    def __bool__(self) -> bool:
        return len(self) > 0
    
    def list_jobs(self) -> List[SamplingJob]:
        """Get list of all jobs ordered by priority."""
        with self._lock:
            items = sorted(self._heap)
            return [item.job for item in items]
    
    def get_by_priority(self, priority: JobPriority) -> List[SamplingJob]:
        """Get all jobs with a specific priority."""
        with self._lock:
            return [
                item.job for item in self._heap 
                if item.job.priority == priority
            ]
