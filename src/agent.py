from importlib.resources import files
from typing import Any, override

import dotenv
import langchain.agents
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint
from langchain.agents.middleware import after_model
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel
from starlette.applications import Starlette

from app_config import settings, reload
from tools.context import current_context
from tools.count_occurrence_records import count_occurrence_records
from tools.find_media_records import find_media_records
from tools.find_occurrence_records import find_occurrence_records
from util import set_llm_credentials, llm_access


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
        set_llm_credentials(metadata)
        current_context.set(context)

        # Run the graph
        langchain_agent = self._build_langchain_agent()
        await langchain_agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": request},
                ]
            }
        )

    def __init__(self):
        self.control_loop_prompt = files().joinpath("resources", "control_loop_prompt.md").read_text()

    def _build_langchain_agent(self):
        @after_model
        async def share_text_messages(state, runtime):
            message = state["messages"][-1]
            if isinstance(message, AIMessage) and isinstance(message.content, str):
                await current_context.get().reply(message.content)

        return langchain.agents.create_agent(
            model=ChatOpenAI(
                model=settings().llm,
                streaming=True,
                base_url=llm_access.get().base_url,
                api_key=llm_access.get().api_key,
                reasoning_effort=settings().reasoning_effort,
                temperature=0,
                model_kwargs={"tool_choice": "required"}
            ),
            tools=[
                find_occurrence_records,
                count_occurrence_records,
                find_media_records,
                abort,
                finish
            ],
            middleware=[share_text_messages],
            system_prompt=self.control_loop_prompt
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
    reload()
    agent = IDigBioAgent()
    app = build_agent_app(agent)
    return app
