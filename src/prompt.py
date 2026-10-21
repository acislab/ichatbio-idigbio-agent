import pydantic

system_prompt_template = """
{preamble}

# Query format

Here is a description of how iDigBio queries are formatted:

[BEGIN QUERY FORMAT DOC]

{query_format_doc}

[END QUERY FORMAT DOC]

# Tips

{tips}

# Examples

{examples}
"""

example_template = """\
## Example {i}

User: {request}

You: {response}\
"""


def make_system_prompt(
    preamble: str, query_format_doc: str, tips: str, examples: dict[str, pydantic.BaseModel]
):
    return system_prompt_template.format(
        preamble=preamble.strip(),
        query_format_doc=query_format_doc.strip(),
        tips=tips,
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
