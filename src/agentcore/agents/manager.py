"""Sub-agent manager for task delegation and parallel execution."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubAgentTask:
    """A task to be delegated to a sub-agent."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    prompt: str = ""
    parent_task_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""

    task_id: str
    success: bool
    output: Any = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# Type for the agent execution function
AgentRunner = Callable[[SubAgentTask], Awaitable[SubAgentResult]]


class SubAgentManager:
    """Manages sub-agent creation, task delegation, and result collection.

    Supports:
    - Sequential task execution
    - Parallel task execution (with concurrency limit)
    - Task decomposition (parent → children)
    - Result aggregation

    Usage:
        manager = SubAgentManager(agent_runner=my_agent_fn)
        task = SubAgentTask(description="Research X", prompt="Find info about X")
        result = await manager.delegate(task)
        # or for parallel:
        results = await manager.delegate_parallel([task1, task2, task3])
    """

    def __init__(
        self,
        agent_runner: AgentRunner | None = None,
        max_concurrent: int = 5,
    ) -> None:
        self._runner = agent_runner
        self._max_concurrent = max_concurrent
        self._tasks: dict[str, SubAgentTask] = {}
        self._results: dict[str, SubAgentResult] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def set_runner(self, runner: AgentRunner) -> None:
        """Set the agent execution function.

        The runner receives a SubAgentTask and must return a SubAgentResult.
        Typically this wraps AgentEngine.chat() with a specialized prompt.
        """
        self._runner = runner

    async def delegate(self, task: SubAgentTask) -> SubAgentResult:
        """Delegate a single task to a sub-agent."""
        if not self._runner:
            return SubAgentResult(task_id=task.task_id, success=False, error="No agent runner configured")

        task.status = TaskStatus.RUNNING
        self._tasks[task.task_id] = task

        try:
            async with self._semaphore:
                result = await self._runner(task)
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            task.result = result.output
            task.error = result.error
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            result = SubAgentResult(task_id=task.task_id, success=False, error=str(e))

        self._results[task.task_id] = result
        return result

    async def delegate_parallel(self, tasks: list[SubAgentTask]) -> list[SubAgentResult]:
        """Delegate multiple tasks in parallel (with concurrency limit)."""
        coros = [self.delegate(task) for task in tasks]
        return await asyncio.gather(*coros)

    async def delegate_sequential(self, tasks: list[SubAgentTask]) -> list[SubAgentResult]:
        """Delegate multiple tasks sequentially (each gets previous results)."""
        results = []
        for task in tasks:
            # Pass previous results as context
            task.metadata["previous_results"] = [
                {"task_id": r.task_id, "output": r.output}
                for r in results
            ]
            result = await self.delegate(task)
            results.append(result)
        return results

    def get_task(self, task_id: str) -> SubAgentTask | None:
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> SubAgentResult | None:
        return self._results.get(task_id)

    def list_tasks(self) -> list[SubAgentTask]:
        return list(self._tasks.values())

    def list_results(self) -> list[SubAgentResult]:
        return list(self._results.values())

    def clear(self) -> None:
        self._tasks.clear()
        self._results.clear()

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
