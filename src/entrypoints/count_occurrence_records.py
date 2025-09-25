import http.client
import importlib.resources

import requests
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentEntrypoint

import util
from prompt import make_system_prompt
from schema import IDigBioSummaryApiParameters, IDBRecordsQuerySchema
from util import (
    AIGenerationException,
    make_llm_response_model,
)

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

MAX_COUNT = 5000


LLMResponseModel = make_llm_response_model(IDigBioSummaryApiParameters)


async def run(context: ResponseContext, request: str):
    async with context.begin_process("Requesting iDigBio statistics") as process:
        process: IChatBioAgentProcess

        await process.log("Generating search parameters for species occurrences")
        try:
            plan, params, artifact_description = await util.generate_search_parameters(
                request, get_system_prompt(), LLMResponseModel
            )
        except AIGenerationException as e:
            await process.log(e.message)
            return

        if params is None:
            await process.log(
                f"Failed to generate appropriate search parameters. Reason: {plan}"
            )
            return

        # Call the API

        json_params = params.model_dump(exclude_none=True, by_alias=True)
        top_fields = remap_top_fields(params.top_fields)

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
                await process.log(
                    f"Warning: Maximum count reached! iDigBio's Summary API can not return more than {MAX_COUNT} unique"
                    f" values. There are probably more than that. Consider narrowing your search parameters if you need"
                    f" exact counts."
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


FIELD_REPLACEMENTS = {
    "collector": "collector.keyword",
    "locality": "locality.keyword",
    "highertaxon": "highertaxon.keyword",
}


def remap_top_fields(top_fields):
    # Some fields are indexed by word instead of full text. This is not useful for many fields. Use the
    # keyword versions of these fields instead.
    if type(top_fields) is str:
        top_fields = FIELD_REPLACEMENTS.get(top_fields, top_fields)
    elif type(top_fields) is list:
        top_fields = [FIELD_REPLACEMENTS.get(field, field) for field in top_fields]

    return top_fields


SYSTEM_PROMPT_TEMPLATE = """\
You translate user requests into parameters for the iDigBio records summary API.

# Query format

Here is a description of how iDigBio queries are formatted:

[BEGIN QUERY FORMAT DOC]

{query_format_doc}

[END QUERY FORMAT DOC]

"""


def get_system_prompt():
    query_format_doc = (
        importlib.resources.files()
        .joinpath("..", "resources", "records_query_format.md")
        .read_text()
    )

    examples = {
        "Number of species of Aves": LLMResponseModel(
            plan='Aves is a taxonomic class, so I will search by class. The request wants the number of unique species in Aves, so I will use "scientificname" as top_fields. Becayse scientificname can also match ranks besides species, so I will also limit the taxonrank (the rank of the scientific name in each record) to species',
            search_parameters=IDigBioSummaryApiParameters(
                rq=IDBRecordsQuerySchema(class_="Aves", taxonrank="species"),
                top_fields="scientificname",
            ),
            artifact_description="Per-species record counts for class Aves",
            search_parameters_fully_match_the_request=True,
        ),
        "Number of families of Aves": LLMResponseModel(
            plan='Aves is a taxonomic class, so I will search by class. The request wants the number of unique families in Aves, so I will use "families" as top_fields.',
            search_parameters=IDigBioSummaryApiParameters(
                rq=IDBRecordsQuerySchema(class_="Aves", taxonrank="species"),
                top_fields="scientificname",
            ),
            artifact_description="Per-family record counts for class Aves",
            search_parameters_fully_match_the_request=True,
        ),
        "Count Ursus arctos in each state in Australia": LLMResponseModel(
            plan='The name Ursus arctos doesn\'t have authority specified, so I will search by genus and specificepithet instead of scientificname. I will limit the search to the country Australia and set top_fields to "stateprovince" to break down record counts by state.',
            search_parameters=IDigBioSummaryApiParameters(
                rq=IDBRecordsQuerySchema(
                    genus="Ursus", specificepithet="arctos", country="Australia"
                ),
                top_fields="stateprovince",
            ),
            artifact_description="Per-state record counts for Ursus arctos in Australia",
            search_parameters_fully_match_the_request=True,
        ),
    }

    return make_system_prompt(
        SYSTEM_PROMPT_TEMPLATE.format(query_format_doc=query_format_doc),
        examples,
    )
