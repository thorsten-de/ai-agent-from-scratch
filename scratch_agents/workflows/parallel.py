"""Parallel workflow: run agents concurrently."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import List

from scratch_agents.agent import Agent
from scratch_agents.context import AgentResult, ExecutionContext


class ParallelWorkflowIncomplete(RuntimeError):
    """A branch needs attention; retain results for inspection or manual resume."""

    def __init__(self, branch_results, branch_errors=()):
        self.branch_results = tuple(branch_results)
        self.branch_errors = tuple(branch_errors)
        states = ", ".join(
            f"{name}: {result.status}"
            for name, result in self.branch_results if result.status != "complete"
        )
        super().__init__(f"Parallel workflow did not complete ({states}). "
                         "Inspect branch_results; automatic parallel resume is not supported.")


class ParallelWorkflow(Agent):
    """Run agents in parallel and combine results."""

    def __init__(
        self,
        agents: List[Agent],
        name: str = "parallel_workflow",
    ):
        self.agents = agents
        self.name = name

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> AgentResult:
        """Execute all agents concurrently."""
        # Each branch owns its execution state. Sharing context would race on
        # final_result/current_step and duplicate events during the merge.
        seed = context or ExecutionContext()
        existing_event_count = len(seed.events)
        branches = [
            ExecutionContext(
                events=deepcopy(seed.events),
                state=deepcopy(seed.state),
                memory_manager=seed.memory_manager,
            )
            for _ in self.agents
        ]

        results = await asyncio.gather(
            *[agent.run(user_input, context=branch, verbose=verbose)
              for agent, branch in zip(self.agents, branches)],
            return_exceptions=True,
        )
        # Retain ordinary failures alongside successful/pending results.
        errors = []
        for i, result in enumerate(results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                errors.append((self.agents[i].name, result))
                results[i] = AgentResult(output=None, context=branches[i], status="error")

        # A merged transcript cannot represent independent pending approvals.
        # Never discard them or label partial work complete. Completed branches
        # may already have performed work; callers must not blindly rerun them.
        if any(result.status != "complete" for result in results):
            raise ParallelWorkflowIncomplete(
                [(agent.name, result) for agent, result in zip(self.agents, results)], errors
            )

        merged_context = ExecutionContext(events=deepcopy(seed.events), state=deepcopy(seed.state))

        seen_user_event = False
        for result in results:
            new_events = result.context.events[existing_event_count:]
            for event in new_events:
                if event.author == "user":
                    if not seen_user_event:
                        merged_context.add_event(event)
                        seen_user_event = True
                else:
                    merged_context.add_event(event)

        # Combine outputs
        combined_output = "\n\n".join(
            f"[{agent.name}]\n{result.output}"
            for agent, result in zip(self.agents, results)
        )
        return AgentResult(output=combined_output, context=merged_context)
