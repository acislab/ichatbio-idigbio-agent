import importlib.resources

from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentEntrypoint

from prompt import make_system_prompt
from schema import IDigBioMediaApiParameters, IDBRecordsQuerySchema, IDBMediaQuerySchema
from util import (
    AIGenerationException,
    query_idigbio_api,
    make_idigbio_api_url,
    generate_search_parameters,
    make_llm_response_model,
)

# This description helps iChatBio understand when to call this entrypoint
description = """\
Searches iDigBio for media records (like images and audio). Returns the total number of media records that were found,
a URL to access the raw results returned by the iDigBio media API, and a URL to view the results in the iDigBio Search
Portal. Also displays an interactive media gallery to the user.
"""

# This gets included in the agent card
entrypoint = AgentEntrypoint(
    id="find_media_records", description=description, parameters=None
)


LLMResponseModel = make_llm_response_model(IDigBioMediaApiParameters)


async def run(context: ResponseContext, request: str):
    """
    Executes this specific entrypoint. See description above. This function yields a sequence of messages that are
    returned one-by-one to iChatBio in response to the request, logging the retrieval process in real time. Any records
    retrieved from the iDigBio API are packaged as an JSON artifact that iChatBio can interact with.
    """
    async with context.begin_process("Searching iDigBio media records") as process:
        process: IChatBioAgentProcess
        await process.log(
            "Generating search parameters for the iDigBio's media records API"
        )
        try:
            plan, params, artifact_description = await generate_search_parameters(
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

        json_params = params.model_dump(exclude_none=True, by_alias=True)
        await process.log(f"Generated search parameters", data=json_params)

        api_query_url = make_idigbio_api_url("/v2/search/media", json_params)
        await process.log(
            f"Sending a POST request to the iDigBio media records API at {api_query_url}"
        )

        response_code, success, response_data = query_idigbio_api(
            "/v2/search/media", json_params
        )

        if not success:
            await process.log(f"Response code: {response_code} - something went wrong!")
            return

        matching_count = response_data.get("itemCount", 0)
        record_count = len(response_data.get("items", []))

        await context.reply(
            f"The API query returned {record_count} out of {matching_count} matching media records in iDigBio using the"
            f" URL {api_query_url}"
        )

        if record_count > 0:
            await process.create_artifact(
                mimetype="application/json",
                description=artifact_description,
                uris=[api_query_url],
                metadata={
                    "data_source": "iDigBio",
                    "retrieved_record_count": record_count,
                    "total_matching_count": matching_count,
                },
            )
            await context.reply(
                "Tips:\n"
                "- Image URLs can be found in the artifact record data at items[].indexterms.accessuri\n"
                "- UUIDs for associated specimen/occurrence records in iDigBio are found in the artifact record data at"
                " items[].indexTerms.records"
                "- The web pages for individual media records follow the pattern https://portal.idigbio.org/portal/mediarecords/[UUID] using the UUIDs found in the artifact record data at items[].uuid."
            )


SYSTEM_PROMPT_TEMPLATE = """
You translate user requests into parameters for the iDigBio media search API.

# Query format

Here is a description of how iDigBio queries are formatted:

[BEGIN QUERY FORMAT DOC]

{query_format_doc}

[END QUERY FORMAT DOC]

# Tips

- Searching by lists performs an OR operation. For example, a search for "genus":["Ursus","Puffinus"] will return Ursus
records and ALSO Puffinus records, it will NOT return co-occurrences of Ursus and Puffinus.

- Do not choose search parameters that only partially fulfill the user's request. Instead, you should abort (don't set any search parameters) and explain why.

"""


def get_system_prompt():
    query_format_doc = (
        importlib.resources.files()
        .joinpath("..", "resources", "records_query_format.md")
        .read_text()
    )

    examples = {
        "Homo sapiens": LLMResponseModel(
            plan="The request only specifies occurrence-related information, I will search using rq fields. The name doesn't have authority specified, so I will search by genus and specificepithet instead of scientificname",
            search_parameters=IDigBioMediaApiParameters(
                rq=IDBRecordsQuerySchema(genus="Homo", specificepithet="sapiens")
            ),
            artifact_description="Occurrence records for the species Homo sapiens",
            search_parameters_fully_match_the_request=True,
        ),
        "Audio of Homo sapiens": LLMResponseModel(
            plan='To filter for audio media I need to use the mq field and search by mediatype. The mediatype for audio is "sounds". The request doesn\'t specify an authority for the name Homo sapiens, so I will search by genus and specificepithet instead of scientificname',
            search_parameters=IDigBioMediaApiParameters(
                rq=IDBRecordsQuerySchema(genus="Homo", specificepithet="sapiens"),
                mq=IDBMediaQuerySchema(mediatype="sounds"),
            ),
            artifact_description="Occurrence records for the species Homo sapiens",
            search_parameters_fully_match_the_request=True,
        ),
        "Pictures of Rattus rattus in Taiwan": LLMResponseModel(
            plan='To filter for picture media I need to use the mq field and search by mediatype. The mediatype for pictures is "images". To filter by species, I need to use the rq field. The request doesn\'t specify an authority for the name Rattus rattus, so I will search by genus and specificepithet instead of scientificname',
            search_parameters=IDigBioMediaApiParameters(
                rq=IDBRecordsQuerySchema(genus="Homo", specificepithet="sapiens")
            ),
            artifact_description="Occurrence records for the species Homo sapiens",
            search_parameters_fully_match_the_request=True,
        ),
        "Blurry images in Canada": LLMResponseModel.model_construct(
            plan="There are no search parameters for image quality, so I should abort.",
            search_parameters=None,
            artifact_description=None,
            search_parameters_fully_match_the_request=False,
        ),
        "Images of blue plants": LLMResponseModel.model_construct(
            plan="There are no search parameters for color or other image features, so I should abort.",
            search_parameters=None,
            artifact_description=None,
            search_parameters_fully_match_the_request=False,
        ),
    }

    return make_system_prompt(
        SYSTEM_PROMPT_TEMPLATE.format(query_format_doc=query_format_doc),
        examples,
    )
