import http.client
import importlib.resources

import instructor
import requests
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentEntrypoint
from instructor import AsyncInstructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying

import util
from schema import IDigBioSummaryApiParameters
from util import AIGenerationException, StopOnTerminalErrorOrMaxAttempts

description = """\
Counts the total number of records in iDigBio matching the user's search criteria. Also breaks the count down by a 
specified field (default: scientific name) to build top-N lists or to find unique record field values that were 
matched. Counts can be broken down by any of iDigBio's query fields, such as "country" or "collector". Does NOT count
the total number of unique values that were matched.

Here are some examples of building top-N lists:
- List the 10 species that have the most records in a country
- List the 5 countries that have the most records of a species
- List the 3 collectors who have recorded the most occurrences of a species

Here are some examples of finding unique values in matching records:
- List the continents that a species occurs in
- List different scientific names that have the same genus and specific epithet (e.g., scientific names with 
different authors)

Also returns the URL used to collect records counts from the iDigBio Summary API.
"""

# This gets included in the agent card
entrypoint = AgentEntrypoint(
    id="count_occurrence_records", description=description, parameters=None
)

DEFAULT_COUNT_TO_SHOW = 10
MAX_COUNT_TO_SHOW = 25
MAX_COUNT = 5000


async def run(context: ResponseContext, request: str):
    async with context.begin_process("Requesting iDigBio statistics") as process:
        process: IChatBioAgentProcess

        await process.log("Generating search parameters for species occurrences")
        try:
            params, artifact_description = await _generate_records_summary_parameters(
                request
            )
        except AIGenerationException as e:
            await process.log(e.message)
            return

        # Call the API

        json_params = params.model_dump(exclude_none=True, by_alias=True)
        top_fields = params.top_fields

        await process.log(f"Generated search parameters", data=json_params)

        url_params = util.url_encode_params(json_params)
        full_summary_api_url = (
            f"https://search.idigbio.org/v2/summary/top/records?{url_params}"
        )

        await process.log(
            f"Sending a GET request to the iDigBio records summary API at {full_summary_api_url}"
        )

        if params.count is None:
            params.count = 0

        response_code, success, total_record_count, top_counts = _query_summary_api(
            full_summary_api_url
        )

        if success:
            await process.log(f"Response code: {response_code}")
        else:
            await process.log(f"Response code: {response_code} - something went wrong!")
            return

        total_unique_count = len(top_counts.get(top_fields, []))

        await context.reply(
            f'The API query found {total_unique_count} unique "{top_fields}" values across {total_record_count}'
            " matching records in iDigBio"
        )
        await process.log(
            f'[View summary of {total_unique_count} unique "{top_fields}" values across {total_record_count} records]({full_summary_api_url})'
        )

        if total_record_count > 0:
            if total_unique_count >= MAX_COUNT:
                await context.reply(
                    f"Warning: Maximum count reached! iDigBio's Summary API can not return more than {MAX_COUNT} unique"
                    f" values. There are probably more than that. Consider narrowing your search parameters if you need"
                    f" exact counts."
                )

            # Show a preview of the top counts

            if params.count is None:
                preview_count = DEFAULT_COUNT_TO_SHOW
            elif params.count > MAX_COUNT_TO_SHOW or params.count == 0:
                preview_count = MAX_COUNT_TO_SHOW
            else:
                preview_count = params.count

            top_field = [x for x in top_counts if x != "itemCount"][0]
            counts_table = {
                k: v["itemCount"]
                for k, v in list(top_counts[top_field].items())[:preview_count]
            }

            await process.log(
                f'Record counts for the top {preview_count} out of {total_unique_count} unique "{top_fields}" values',
                data={"__table": counts_table},
            )
            await process.create_artifact(
                mimetype="application/json",
                description=artifact_description,
                uris=[full_summary_api_url],
                metadata={
                    "data_source": "iDigBio",
                    "total_record_count": total_record_count,
                    "total_unique_count": total_unique_count,
                },
            )


def _query_summary_api(query_url: str) -> (int, dict):
    response = requests.get(query_url)
    item_count = response.json()["itemCount"]
    code = (
        f"{response.status_code} {http.client.responses.get(response.status_code, '')}"
    )
    return code, response.ok, item_count, response.json()


class LLMResponseModel(BaseModel):
    plan: str = Field(
        description="A brief explanation of what API parameters you plan to use"
    )
    search_parameters: IDigBioSummaryApiParameters = Field()
    artifact_description: str = Field(
        description="A concise characterization of the retrieved occurrence record statistics",
        examples=[
            "Per-country record counts for species Rattus rattus",
            "Per-species record counts for records created in 2025",
        ],
    )


FIELD_REPLACEMENTS = {
    "collector": "collector.keyword",
    "locality": "locality.keyword",
    "highertaxon": "highertaxon.keyword",
}


async def _generate_records_summary_parameters(
    request: str,
) -> (IDigBioSummaryApiParameters, str):
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

    # Some fields are indexed by word instead of full text. This is not useful for many fields. Use the
    # keyword versions of these fields instead.
    top_fields = result.search_parameters.top_fields
    if type(top_fields) is str:
        result.search_parameters.top_fields = FIELD_REPLACEMENTS.get(
            top_fields, top_fields
        )
    elif type(top_fields) is list:
        result.search_parameters.top_fields = [
            FIELD_REPLACEMENTS.get(field, field) for field in top_fields
        ]

    return result.search_parameters, result.artifact_description


SYSTEM_PROMPT_TEMPLATE = """\
You translate user requests into parameters for the iDigBio records summary API.

# Query format

Here is a description of how iDigBio queries are formatted:

[BEGIN QUERY FORMAT DOC]

{query_format_doc}

[END QUERY FORMAT DOC]

# General rq object examples

{examples_doc}

# Full examples

{specific_examples}
"""

SPECIFIC_EXAMPLES = """\
## Example

Request: "Count number of species of Aves"
Response: {
    "rq": {"class": "Aves", "taxonrank": "species"},
    "top_fields": "scientificname"
}
"""


def get_system_prompt():
    query_format_doc = (
        importlib.resources.files()
        .joinpath("..", "resources", "records_query_format.md")
        .read_text()
    )
    examples_doc = (
        importlib.resources.files()
        .joinpath("..", "resources", "occurrence_records_examples.md")
        .read_text()
    )

    return SYSTEM_PROMPT_TEMPLATE.format(
        query_format_doc=query_format_doc,
        examples_doc=examples_doc,
        specific_examples=SPECIFIC_EXAMPLES,
    ).strip()
