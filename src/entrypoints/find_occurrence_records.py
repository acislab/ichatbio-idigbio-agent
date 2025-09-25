import functools
import importlib.resources
from typing import Optional

import instructor
from ichatbio.agent_response import IChatBioAgentProcess, ResponseContext
from ichatbio.types import AgentEntrypoint
from instructor import AsyncInstructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import Field, BaseModel
from tenacity import AsyncRetrying

from prompt import make_system_prompt
from schema import IDBRecordsQuerySchema, IDigBioRecordsApiParameters
from util import (
    query_idigbio_api,
    make_idigbio_api_url,
    make_idigbio_portal_url,
    AIGenerationException,
    StopOnTerminalErrorOrMaxAttempts,
)

# This description helps iChatBio understand when to call this entrypoint
description = """\
Searches for species occurrence records using the iDigBio Portal or the iDigBio records API. Returns the total number 
of records that were found, the URL used to call the iDigBio Records API to perform the search, and a URL to view the 
results in the iDigBio Search Portal.
"""

# This gets included in the agent card
entrypoint = AgentEntrypoint(
    id="find_occurrence_records", description=description, parameters=None
)


async def run(context: ResponseContext, request: str):
    """
    Executes this specific entrypoint. See description above. This function yields a sequence of messages that are
    returned one-by-one to iChatBio in response to the request, logging the retrieval process in real time. Any records
    retrieved from the iDigBio API are packaged as an JSON artifact that iChatBio can interact with.
    """
    async with context.begin_process("Searching iDigBio occurrence records") as process:
        process: IChatBioAgentProcess

        await process.log(
            "Generating search parameters for iDigBio's occurrence records API"
        )
        try:
            plan, params, artifact_description = (
                await _generate_records_search_parameters(request)
            )
        except AIGenerationException as e:
            await process.log(e.message)
            return

        if params is None:
            await process.log(
                f"Failed to generate appropriate search parameters. Reason: {plan}"
            )
            return

        await process.log(f"Generated search parameters", data=params)

        api_query_url = make_idigbio_api_url("/v2/search/records", params)
        await process.log(
            f"Sending a POST request to the iDigBio occurrence records API at {api_query_url}"
        )

        response_code, success, response_data = query_idigbio_api(
            "/v2/search/records", params
        )

        if success:
            await process.log(f"Response code: {response_code}")
        else:
            await process.log(f"Response code: {response_code} - something went wrong!")
            return

        matching_count = response_data.get("itemCount", 0)
        record_count = len(response_data.get("items", []))

        await context.reply(
            f"The API query returned {record_count} out of {matching_count} matching records in iDigBio using the URL"
            f" {api_query_url}"
        )

        portal_url = make_idigbio_portal_url(params)
        await process.log(
            f"[View {record_count} out of {matching_count} matching records]({api_query_url})"
            f" | [Show in iDigBio portal]({portal_url})"
        )

        if record_count > 0:
            await context.reply(
                f"The records can be viewed in the iDigBio portal at {portal_url}. The portal shows the records in an"
                f" interactive list and plots them on a map. The raw records returned returned by the API can be found"
                f" at {api_query_url}"
            )
            await process.create_artifact(
                mimetype="application/json",
                description=artifact_description,
                uris=[api_query_url],
                metadata={
                    "data_source": "iDigBio",
                    "portal_url": portal_url,
                    "retrieved_record_count": record_count,
                    "total_matching_count": matching_count,
                },
            )


class LLMResponseModel(BaseModel):
    plan: str = Field(
        description="A brief explanation of what API parameters you plan to use. Or, if you are unable to fulfill the user's request using the available API parameters, provide a brief explanation for why you cannot retrieve the requested records."
    )
    search_parameters: Optional[IDigBioRecordsApiParameters] = Field(
        None,
        description="The search parameters to use to produce the requested records. If you are unable to fulfill the user's request using the available API parameters, leave this field unset to abort.",
    )
    artifact_description: Optional[str] = Field(
        None,
        description="A concise characterization of the retrieved occurrence record data, if any.",
    )


async def _generate_records_search_parameters(request: str) -> (dict, str):
    try:
        client: AsyncInstructor = instructor.from_openai(AsyncOpenAI())
        result = await client.chat.completions.create(
            model="gpt-4.1-unfiltered",
            temperature=0,
            response_model=LLMResponseModel,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": request},
            ],
            max_retries=AsyncRetrying(stop=StopOnTerminalErrorOrMaxAttempts(3)),
        )
    except InstructorRetryException as e:
        raise AIGenerationException(e)

    generation = result.model_dump(exclude_none=True, by_alias=True)
    return (
        generation.get("plan"),
        generation.get("search_parameters"),
        generation.get("artifact_description"),
    )


SYSTEM_PROMPT_TEMPLATE = """
You translate user requests into parameters for the iDigBio record search API.

# Query format

Here is a description of how iDigBio queries are formatted:

[BEGIN QUERY FORMAT DOC]

{query_format_doc}

[END QUERY FORMAT DOC]

# Tips

- Searching by lists performs an OR operation. For example, a search for "genus":["Ursus","Puffinus"] will return Ursus
records and ALSO Puffinus records, it will NOT return co-occurrences of Ursus and Puffinus.

"""


@functools.cache
def get_system_prompt():
    query_format_doc = (
        importlib.resources.files()
        .joinpath("..", "resources", "records_query_format.md")
        .read_text()
    )

    examples = {
        "Homo sapiens": LLMResponseModel(
            plan="The name doesn't have authority specified, so I will search by genus and specificepithet instead of scientificname",
            search_parameters=IDigBioRecordsApiParameters(
                rq=IDBRecordsQuerySchema(genus="Homo", specificepithet="sapiens")
            ),
            artifact_description="Occurrence records for the species Homo sapiens in iDigBio",
        ),
        "Only Homo sapiens Linnaeus, 1758": LLMResponseModel(
            plan="The name name includes authority information, so I will search by scientificname",
            search_parameters=IDigBioRecordsApiParameters(
                rq=IDBRecordsQuerySchema(scientificname="Homo sapiens Linnaeus, 1758")
            ),
            artifact_description='Occurrence records for the species "Homo sapiens Linnaeus, 1758" in iDigBio',
        ),
        'Scientific name "this is fake but use it anyway"': LLMResponseModel(
            plan="The request placed a scientific name in quotes, so I will search by scientificname for an exact match",
            search_parameters=IDigBioRecordsApiParameters(
                rq=IDBRecordsQuerySchema(
                    scientificname="this is fake but use it anyway"
                )
            ),
            artifact_description='Occurrence records for the species "this is fake but use it anyway" in iDigBio',
        ),
        "kingdom must be specified": LLMResponseModel(
            plan='To find records that have the kingdom field, I need to search by kingdom for {"type": "exists"}',
            search_parameters=IDigBioRecordsApiParameters(
                rq=IDBRecordsQuerySchema(kingdom={"type": "exists"})
            ),
            artifact_description="Occurrence records with the kingdom field specified in iDigBio",
        ),
        "Records with no collector specified": LLMResponseModel(
            plan='To find records with no collector field, I need to search by collector for {"type": "missing"}',
            search_parameters=IDigBioRecordsApiParameters(
                rq=IDBRecordsQuerySchema(collector={"type": "missing"})
            ),
            artifact_description="Occurrence records with no collector specified in iDigBio",
        ),
        "Homo sapiens and Rattus rattus in North America and Australia": LLMResponseModel(
            plan="The request concerns two species (Homo sapiens and Rattus rattus) in two continents (North America and Australia), so I wlll search using the scientificnmae and continent fields, specifying the search values using list syntax.",
            search_parameters=IDigBioRecordsApiParameters(
                rq=IDBRecordsQuerySchema(
                    scientificname=["Homo sapiens", "Rattus rattus"],
                    continent=["North America", "Australia"],
                )
            ),
            artifact_description="Occurrence records of Homo sapiens and Rattus rattus in North America and Australia in iDigBio",
        ),
    }

    return make_system_prompt(
        SYSTEM_PROMPT_TEMPLATE.format(query_format_doc=query_format_doc),
        examples,
    )
