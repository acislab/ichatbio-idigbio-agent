import importlib.resources

import instructor
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentEntrypoint
from instructor import AsyncInstructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import Field, BaseModel
from tenacity import AsyncRetrying

from util import AIGenerationException, StopOnTerminalErrorOrMaxAttempts
from ..schema import IDigBioMediaApiParameters
from ..util import query_idigbio_api, make_idigbio_api_url

# This description helps iChatBio understand when to call this entrypoint
description = """\
Searches iDigBio for media records (like images and audio). Returns the total number of media records that were found,
a URL to access the raw results returned by the iDigBio media API, and a URL to view the results in the iDigBio Search
Portal. Also displays an interactive media gallery to the user.
"""

# This gets included in the agent card
entrypoint = AgentEntrypoint(
    id="find_media_records",
    description=description,
    parameters=None
)

NUM_PREVIEW_URLS = 5


async def run(context: ResponseContext, request: str):
    """
    Executes this specific entrypoint. See description above. This function yields a sequence of messages that are
    returned one-by-one to iChatBio in response to the request, logging the retrieval process in real time. Any records
    retrieved from the iDigBio API are packaged as an JSON artifact that iChatBio can interact with.
    """
    async with context.begin_process("Searching iDigBio media records") as process:
        process: IChatBioAgentProcess
        await process.log("Generating search parameters for the iDigBio's media records API")
        try:
            params, artifact_description = await _generate_records_search_parameters(request)
        except AIGenerationException as e:
            await process.log(e.message)
            return

        await process.log(f"Generated search parameters", data=params)

        api_query_url = make_idigbio_api_url("/v2/search/media", params)
        await process.log(f"Sending a POST request to the iDigBio media records API at {api_query_url}")

        response_code, success, response_data = query_idigbio_api("/v2/search/media", params)

        if success:
            await process.log(f"Response code: {response_code}")
        else:
            await process.log(f"Response code: {response_code} - something went wrong!")
            return

        matching_count = response_data.get("itemCount", 0)
        record_count = len(response_data.get("items", []))

        await context.reply(
            f"The API query returned {record_count} out of {matching_count} matching media records in iDigBio using the"
            f" URL {api_query_url}"
        )

        if record_count > 0:
            preview_items = [item.get("indexTerms", {})
                             for item in response_data.get("items")
                             if item.get("indexTerms", {}).get("accessuri")][:NUM_PREVIEW_URLS]

            if len(preview_items) > 0:
                await process.log(
                    f"Summary of first {len(preview_items)} media records:",
                    data={"__table": _make_record_previews(preview_items)}
                )

            await process.create_artifact(
                mimetype="application/json",
                description=artifact_description,
                uris=[api_query_url],
                metadata={
                    "data_source": "iDigBio",
                    "retrieved_record_count": record_count,
                    "total_matching_count": matching_count
                }
            )
            await context.reply(
                "I showed the user an interactive media gallery that shows images matching the search parameters"
                " above. Notes:\n"
                "- To show the user images, you may use the access URLs above.\n"
                "- To see more of the retrieved media URLs, you will have to extract them from the"
                " artifact above.\n"
                "- UUIDs for associated specimen/occurrence records in iDigBio are found in the artifact record data at"
                " indexTerms.records"
            )
        else:
            await context.reply(f"I didn't find any matching media records in iDigBio, so no images will be shown.")


class LLMResponseModel(BaseModel):
    plan: str = Field(description="A brief explanation of what API parameters you plan to use")
    search_parameters: IDigBioMediaApiParameters = Field()
    artifact_description: str = Field(
        description="A concise characterization of the retrieved media record data",
        examples=["Image media records of Rattus rattus",
                  "Media records modified in 2025"])


async def _generate_records_search_parameters(request: str) -> (dict, str):
    try:
        client: AsyncInstructor = instructor.from_openai(AsyncOpenAI())
        result = await client.chat.completions.create(
            model="gpt-4.1-unfiltered",
            temperature=0,
            response_model=LLMResponseModel,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": request}
            ],
            max_retries=AsyncRetrying(stop=StopOnTerminalErrorOrMaxAttempts(3))
        )
    except InstructorRetryException as e:
        raise AIGenerationException(e)

    generation = result.model_dump(exclude_none=True, by_alias=True)
    return generation["search_parameters"], generation["artifact_description"]


def _make_record_previews(records) -> list[dict[str, str]]:
    return [
        {
            "type": record.get("type"),
            "format": record.get("format"),
            "accessuri": record.get("accessuri"),
            "Online view": (f"[View in iDigBio](https://portal.idigbio.org/portal/mediarecords/{record["uuid"]})"
                            if "uuid" in record else None)
        } for record in records
    ]


SYSTEM_PROMPT_TEMPLATE = """
You translate user requests into parameters for the iDigBio media search API.

# Query format

Here is a description of how iDigBio queries are formatted:

[BEGIN QUERY FORMAT DOC]

{query_format_doc}

[END QUERY FORMAT DOC]

# rq examples

{rq_examples_doc}

# mq examples

{mq_examples_doc}

"""


def get_system_prompt():
    query_format_doc = importlib.resources.files().joinpath("..", "resources", "records_query_format.md").read_text()
    rq_examples_doc = importlib.resources.files().joinpath("..", "resources",
                                                           "occurrence_records_examples.md").read_text()
    mq_examples_doc = importlib.resources.files().joinpath("..", "resources", "media_records_examples.md").read_text()

    return SYSTEM_PROMPT_TEMPLATE.format(
        query_format_doc=query_format_doc,
        rq_examples_doc=rq_examples_doc,
        mq_examples_doc=mq_examples_doc
    ).strip()
