import pydantic

example_template = """\
# Example {i}

User: {request}

You: {response}\
"""


def make_system_prompt(prelude: str, examples: dict[str, pydantic.BaseModel]):
    return (
        prelude
        + "\n\n".join(
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
        ).strip()
    )
