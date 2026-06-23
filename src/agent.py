import importlib.resources
import os
from typing import override, Any

import dotenv
import langchain.agents
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel
from starlette.applications import Starlette

from tools.context import current_context
from tools.count_occurrence_records import count_occurrence_records
from tools.find_media_records import find_media_records
from tools.find_occurrence_records import find_occurrence_records
from util import update_llm_credentials


class IDigBioAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="iDigBio Search",
            description="Searches for information in the iDigBio portal (https://idigbio.org).",
            documentation_url="https://github.com/acislab/ichatbio-idigbio-agent",
            icon="https://raw.githubusercontent.com/acislab/ichatbio-idigbio-agent/refs/heads/main/src/resources/idigbio.png",
            entrypoints=[
                AgentEntrypoint(
                    id="search_idigbio",
                    description="Retrieves data from the iDigBio portal, including species occurrence records and associated media records. Can also provide breakdowns of record counts by record fields like scientific name and country."
                ),
            ],
        )

    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: BaseModel | None = None,
        metadata: dict[str, Any] | None = None
    ):
        """
        Executes a LangChain agent graph with `request` as input. The agent does not produce text responses directly,
        but must do so by calling tools. Only tools send response messages back iChatBio.
        """
        # If configured to use iChatBio as an LLM proxy, use access information provided in request metadata
        update_llm_credentials(metadata)

        # Give tools access to the `context` object so they can send response messages
        current_context.set(context)

        # Run the graph
        await self.langchain_agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": request},
                ]
            }
        )

    def __init__(self):
        control_loop_prompt = (
            importlib.resources.files()
            .joinpath("resources", "control_loop_prompt.md")
            .read_text()
        )

        # Build a LangChain agent graph
        self.langchain_agent = langchain.agents.create_agent(
            model=ChatOpenAI(
                model=os.getenv("LLM"),
                tool_choice="required",
                openai_api_key=lambda: os.getenv("OPENAI_API_KEY")
            ),
            tools=[
                find_occurrence_records,
                count_occurrence_records,
                find_media_records,
                abort,
                finish
            ],
            system_prompt=control_loop_prompt,
        )


@tool(return_direct=True)  # This tool ends the agent loop
async def abort(reason: str, runtime: ToolRuntime):
    """If you can't fulfill the user's request, abort instead and explain why."""
    await current_context.get().reply(reason)


@tool(return_direct=True)  # This tool ends the agent loop
async def finish(message: str, runtime: ToolRuntime):
    """Mark the user's request as successfully completed."""
    await current_context.get().reply(message)


def create_app() -> Starlette:
    dotenv.load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None:
        raise ValueError("OPENAI_API_KEY environment variable must be set")

    if os.getenv("LLM") is None:
        raise ValueError("LLM environment variable must be set")

    agent = IDigBioAgent()
    app = build_agent_app(agent)
    return app
