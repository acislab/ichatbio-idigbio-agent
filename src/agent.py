import importlib.resources
from typing import override, Optional

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
from tools.find_media_records import find_media_records
from tools.find_occurrence_records import find_occurrence_records
from tools.count_occurrence_records import count_occurrence_records


class IDigBioAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="iDigBio Search",
            description="Searches for information in the iDigBio portal (https://idigbio.org).",
            documentation_url="https://github.com/acislab/ichatbio-idigbio-agent",
            icon='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAjVBMVEVHcExgjLhgjLhim4pgjLhgjLhgjLhgjLhgjLhgjLhgjLnMoy/WjCnWjCnWjClqqlFqqlFqqlHWjCnWjCnXjCfVpC5qqlFqqlFqqlFqqlHXjCjXjChqqlFqqlFqqlHVmSzWjChqqlHWjCnUuDPTuTPTuTPTuDPTuDPTuDPTuDPTuDPTuDPTuDPTuDPTuDN06hLOAAAAL3RSTlMAjeob/ms9z3muoRKP/6uV3F7ZekY3Mnf/yqDA9UbvKmOy8b9JaH/lW7WZ/8qpoeTYrfYAAAF3SURBVHgBbMtBDsAwCMTAhnhhyf8fXPVWKfg68vNrxUb5TGUVadCExXfCGt42fQSibiQOaQQekBUtg7nR4E5Q1I0lyIOIfWPi6FyUXxaoAseBGAZOmeIwOMtJuf3//y6VzrLAOICLIEnyv6+00ab9XTXI02oDss5bQQFAVEkxc8Zqu1qdz/CenLBedP0F4GFUo57MhPNhf0BHs5xt6D3RAsVmYt4UNSY0Y3vqAi2LF3MnPUxsM1aZ2WgAfUdLWyDhiTwmHXmYDHNRDReo1Hk/z56kCMisY1Em5zbPuN4giESVfaXgkdo/bqBZmYLrfb6DJLUUs50BTkMbtgWFx/x43m8XJ/sfJQcAKU1cMhe83s/P5/sA4FxvL/jFpowlmYi/NsgYCWAQBILprXlB9Iiu8P/nxTIT3ZId4OZmD2eaWOMvZZXd5ETeMNavPw8agKz7LrHJipPS3GUktdLN4XTWAj1w2y7rCLKhFuU6WIYQ1OtEEen52XsBMysktejZSV8AAAAASUVORK5CYII=',
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
        params: Optional[BaseModel],
    ):
        """
        Executes a LangChain agent graph with `request` as input. The agent does not produce text responses directly,
        but must do so by calling tools. Only tools send response messages back iChatBio.
        """

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
            model=ChatOpenAI(model="gpt-4.1", tool_choice="required"),
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
    agent = IDigBioAgent()
    app = build_agent_app(agent)
    return app
