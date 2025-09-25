from typing import override, Optional

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard
from pydantic import BaseModel
from starlette.applications import Starlette

from .entrypoints import (
    find_occurrence_records,
    find_media_records,
    count_occurrence_records,
)


class IDigBioAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="iDigBio Search",
            description="Searches for information in the iDigBio portal (https://idigbio.org).",
            icon=None,
            entrypoints=[
                find_occurrence_records.entrypoint,
                find_media_records.entrypoint,
                count_occurrence_records.entrypoint,
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
        match entrypoint:
            case find_occurrence_records.entrypoint.id:
                await find_occurrence_records.run(context, request)
            case find_media_records.entrypoint.id:
                await find_media_records.run(context, request)
            case count_occurrence_records.entrypoint.id:
                await count_occurrence_records.run(context, request)
            case _:
                raise ValueError()


def create_app() -> Starlette:
    agent = IDigBioAgent()
    app = build_agent_app(agent)
    return app
