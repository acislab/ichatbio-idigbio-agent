import pydantic

system_prompt_template = """
{preamble}

# Query format

Here is a description of how iDigBio queries are formatted:

[BEGIN QUERY FORMAT DOC]

{query_format_doc}

[END QUERY FORMAT DOC]

# Examples

{examples}

# Tips

- When searching for records at the species level with binomial names (e.g. Homo sapiens), ALWAYS prefer to use "genus" and "specificepithet" instead of "scientificname". ONLY use "scientificname" if the user specifically requests it.

- Searching by lists performs an OR operation. For example, a search for "genus":["Ursus","Puffinus"] will return Ursus
records and ALSO Puffinus records, it will NOT return co-occurrences of Ursus and Puffinus.

- The iDigBio API can NOT perform searches that relate records to each other. For example, it cannot retrieve records that are near other records unless the locations of those records can be specified as search parameters.

- Do not choose search parameters that only partially fulfill the user's request. Instead, you should abort (don't set any search parameters) and explain why.
"""

example_template = """\
## Example {i}

User: {request}

You: {response}\
"""


def make_system_prompt(
    preamble: str, query_format_doc: str, examples: dict[str, pydantic.BaseModel]
):
    return system_prompt_template.format(
        preamble=preamble.strip(),
        query_format_doc=query_format_doc.strip(),
        examples="\n\n".join(
            (
                example_template.format(
                    i=i,
                    request=request,
                    response=response.model_dump_json(
                        exclude_none=True, exclude_unset=True, by_alias=True
                    ),
                )
                for i, (request, response) in enumerate(examples.items())
            )
        ),
    ).strip()
