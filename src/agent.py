from typing import override, Optional

import dotenv
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard
from pydantic import BaseModel
from starlette.applications import Starlette

from entrypoints import (
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
            icon='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAMAAABF0y+mAAAAjVBMVEVHcExgjLhgjLhim4pgjLhgjLhgjLhgjLhgjLhgjLhgjLnMoy/WjCnWjCnWjClqqlFqqlFqqlHWjCnWjCnXjCfVpC5qqlFqqlFqqlFqqlHXjCjXjChqqlFqqlFqqlHVmSzWjChqqlHWjCnUuDPTuTPTuTPTuDPTuDPTuDPTuDPTuDPTuDPTuDPTuDPTuDN06hLOAAAAL3RSTlMAjeob/ms9z3muoRKP/6uV3F7ZekY3Mnf/yqDA9UbvKmOy8b9JaH/lW7WZ/8qpoeTYrfYAAAF3SURBVHgBbMtBDsAwCMTAhnhhyf8fXPVWKfg68vNrxUb5TGUVadCExXfCGt42fQSibiQOaQQekBUtg7nR4E5Q1I0lyIOIfWPi6FyUXxaoAseBGAZOmeIwOMtJuf3//y6VzrLAOICLIEnyv6+00ab9XTXI02oDss5bQQFAVEkxc8Zqu1qdz/CenLBedP0F4GFUo57MhPNhf0BHs5xt6D3RAsVmYt4UNSY0Y3vqAi2LF3MnPUxsM1aZ2WgAfUdLWyDhiTwmHXmYDHNRDReo1Hk/z56kCMisY1Em5zbPuN4giESVfaXgkdo/bqBZmYLrfb6DJLUUs50BTkMbtgWFx/x43m8XJ/sfJQcAKU1cMhe83s/P5/sA4FxvL/jFpowlmYi/NsgYCWAQBILprXlB9Iiu8P/nxTIT3ZId4OZmD2eaWOMvZZXd5ETeMNavPw8agKz7LrHJipPS3GUktdLN4XTWAj1w2y7rCLKhFuU6WIYQ1OtEEen52XsBMysktejZSV8AAAAASUVORK5CYII=',
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
    dotenv.load_dotenv()
    agent = IDigBioAgent()
    app = build_agent_app(agent)
    return app
