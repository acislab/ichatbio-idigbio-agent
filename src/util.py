import http.client
import json
from typing import Sized, Union, Type, Optional, Self

import instructor
import requests
from instructor import AsyncInstructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import Field
from pydantic.functional_validators import model_validator
from pydantic_core import ValidationError
from tenacity import AsyncRetrying
from tenacity import RetryCallState
from tenacity.stop import stop_base


def _get_terminal_validation_error(e: Exception):
    if isinstance(e, ValidationError):
        for error in e.errors():
            if error.get("ctx", {}).get("terminal", False):
                return error
    return None


class AIGenerationException(Exception):
    def __init__(self, e: InstructorRetryException):
        messages = []
        terminal_error = _get_terminal_validation_error(e)
        if terminal_error:
            messages.append(f"Error: {terminal_error['msg']}")
        else:
            messages.append(
                f"Error: AI failed to generate valid output after {e.n_attempts} attempts."
            )

        self.message = "\n\n".join(messages)


class StopOnTerminalErrorOrMaxAttempts(stop_base):
    """Stop when a bad value is encountered."""

    def __init__(self, max_attempts: int):
        self.max_attempts = max_attempts

    def __call__(self, retry_state: RetryCallState) -> bool:
        exception = retry_state.outcome.exception()
        if _get_terminal_validation_error(exception):
            return True
        else:
            return retry_state.attempt_number >= self.max_attempts


def url_encode_inner(x):
    if type(x) == dict:
        return (
            "{" + ",".join([f'"{k}":{url_encode_inner(v)}' for k, v in x.items()]) + "}"
        )
    elif type(x) == list:
        return "[" + ",".join([url_encode_inner(v) for v in x]) + "]"
    elif type(x) == str:
        return f'"{x}"'
    elif type(x) == int:
        return str(x)
    else:
        return f'"{str(x)}"'


def url_encode_params(d: dict) -> str:
    d = sanitize_json(d)
    return percent_encode(
        "&".join([f"{k}={url_encode_inner(v)}" for k, v in d.items()])
    )


PERCENT_ENCODING = [("{", "%7B"), ("}", "%7D"), ('"', "%22"), (" ", "%20")]


def percent_encode(s: str):
    for codec in PERCENT_ENCODING:
        s = s.replace(codec[0], codec[1])
    return s


JSON = Union[dict, list, str, int, float]


def sanitize_json(data: JSON) -> JSON:
    match data:
        case dict():
            return {k: sanitize_json(v) for k, v in data.items() if not _is_empty(v)}
        case list():
            return [sanitize_json(v) for v in data if not _is_empty(v)]
        case int() | float():
            return data
        case _:
            return str(data)


def _is_empty(data):
    return len(data) == 0 if isinstance(data, Sized) else False


def query_idigbio_api(endpoint: str, params: dict) -> (str, bool, dict):
    params = sanitize_json(params)
    api_url = make_idigbio_api_url(endpoint)
    response = requests.post(api_url, json=params)
    code = (
        f"{response.status_code} {http.client.responses.get(response.status_code, '')}"
    )
    data = response.json() if response.ok else None
    return code, response.ok, data


def query_idigbio_data_api(params) -> (str, bool, dict):
    sanitized_query = sanitize_json(params.get("rq", {}))
    api_params = {"rq": json.dumps(sanitized_query), "email": params.get("email", "")}
    response = requests.post("https://api.idigbio.org/v2/download", data=api_params)
    code = (
        f"{response.status_code} {http.client.responses.get(response.status_code, '')}"
    )
    return code, response.ok, response.json()


def make_idigbio_portal_url(params: dict = None):
    url_params = "" if params is None else "?" + url_encode_params(params)
    return f"https://portal.idigbio.org/portal/search{url_params}"


def make_idigbio_api_url(endpoint: str, params: dict = None) -> str:
    url_params = "" if params is None else "?" + url_encode_params(params)
    return f"https://search.idigbio.org{endpoint}{url_params}"


def make_idigbio_download_url(params: dict = None):
    url_params = "" if params is None else "?" + url_encode_params(params)
    return f"https://api.idigbio.org/v2/download{url_params}"


def make_llm_response_model(search_parameters_model: Type[BaseModel]):
    class LLMResponseModel(BaseModel):
        plan: str = Field(
            description="A brief explanation of what API parameters you plan to use. Or, if you are unable to fulfill the user's request using the available API parameters, provide a brief explanation for why you cannot retrieve the requested records."
        )
        search_parameters: Optional[search_parameters_model] = Field(
            None,
            description="The search parameters to use to produce the requested media records. If you are unable to fulfill the user's request using the available API parameters, leave this field unset to abort.",
        )
        artifact_description: Optional[str] = Field(
            None,
            description="A concise characterization of the retrieved occurrence record data, if any.",
        )
        search_parameters_fully_match_the_request: bool = Field(
            description="Whether or not the chosen search_parameters completely fulfill the request. It is unacceptable for the search parameters to only partially match the request.",
        )

        @model_validator(mode="after")
        def validate_model(self) -> Self:
            if (
                self.search_parameters is not None
                and not self.search_parameters_fully_match_the_request
            ):
                raise ValueError(
                    "The selected search parameters do not fully match the request. Either try again or abort."
                )

            return self

    return LLMResponseModel


async def generate_search_parameters(
    request: str, system_prompt: str, llm_response_model: BaseModel
) -> (str, BaseModel, str):

    try:
        client: AsyncInstructor = instructor.from_openai(AsyncOpenAI())
        result = await client.chat.completions.create(
            model="gpt-4.1-unfiltered",
            temperature=0,
            response_model=llm_response_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ],
            max_retries=AsyncRetrying(stop=StopOnTerminalErrorOrMaxAttempts(3)),
        )
    except InstructorRetryException as e:
        raise AIGenerationException(e)

    return result.plan, result.search_parameters, result.artifact_description
