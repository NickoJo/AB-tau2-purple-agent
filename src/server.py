import argparse
import logging
import os

import uvicorn

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill

from executor import Executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    base_url = args.card_url or f"http://{args.host}:{args.port}/"

    skill = AgentSkill(
        id="cs-agent-tau2",
        name="Multi-Domain Customer Service",
        description="Resolves customer requests across airline, retail, and telecom domains by following domain-specific policies and invoking the appropriate tools.",
        tags=["customer-service", "tool-use", "tau2"],
        examples=["I need to cancel my flight", "Can I return this item?", "My data plan ran out"],
    )

    card = AgentCard(
        name="purple-agent",
        description="An LLM-driven customer service agent that reads domain policy at runtime and uses tool calls to fulfil customer requests within policy constraints.",
        url=base_url,
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities={"streaming": False},
        skills=[skill],
    )

    store = InMemoryTaskStore()
    handler = DefaultRequestHandler(agent_executor=Executor(), task_store=store)
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
