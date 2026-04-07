from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

from agent import Agent


DONE_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    def __init__(self):
        self._sessions: dict[str, Agent] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="No message provided in request"))

        task = context.current_task
        if task and task.status.state in DONE_STATES:
            raise ServerError(error=InvalidRequestError(message=f"Task {task.id} has already finished (state: {task.status.state})"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        session_id = task.context_id
        agent = self._sessions.get(session_id)
        if not agent:
            agent = Agent()
            self._sessions[session_id] = agent

        updater = TaskUpdater(event_queue, task.id, session_id)

        await updater.start_work()
        try:
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as exc:
            print(f"Agent raised an exception: {exc}")
            await updater.failed(new_agent_text_message(f"Agent error: {exc}", context_id=session_id, task_id=task.id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
