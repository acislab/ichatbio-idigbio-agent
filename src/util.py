import http.client
import json
import os
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Self,
    Sized,
    Type,
    TypeVar,
    Union,
    cast,
)
from contextvars import ContextVar
import dotenv
import instructor
import requests
from instructor import AsyncInstructor
from instructor.core import InstructorRetryException
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import Field
from pydantic.functional_validators import model_validator
from pydantic_core import ValidationError
from tenacity import AsyncRetrying
from tenacity import RetryCallState
from tenacity.stop import stop_base


dotenv.load_dotenv()

langfuse = get_client()


def update_llm_credentials(metadata: dict[str, Any] | None):
    if metadata is None:
        return
temporary_llm_key: ContextVar[str | None] = ContextVar(
    "temporary_llm_key",
    default=None,
)


def update_llm_credentials(metadata: dict[str, Any] | None):
    match metadata:
        case {"https://ichatbio.org/a2a/v1": {"temporary_llm_key": llm_key}}:
            temporary_llm_key.set(llm_key)
        case _:
            temporary_llm_key.set(None)


def get_llm_client_kwargs() -> dict[str, str]:
    metadata_llm_key = temporary_llm_key.get()
    use_proxy = os.getenv("USE_LLM_PROXY") == "true" or metadata_llm_key is not None

    if use_proxy:
        assert metadata_llm_key is not None, "Temporary LLM key is required for proxy mode"
        proxy_base_url = os.getenv("PROXY_OPENAI_BASE_URL")
        assert proxy_base_url is not None, "PROXY_OPENAI_BASE_URL environment variable must be set"
        return {"api_key": metadata_llm_key, "base_url": proxy_base_url}

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    assert openai_api_key is not None, "OPENAI_API_KEY environment variable must be set"
    assert openai_base_url is not None, "OPENAI_BASE_URL environment variable must be set"
    return {"api_key": openai_api_key, "base_url": openai_base_url}


def _jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, list):
        return [_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if hasattr(value, "dict"):
        return value.dict()

    return value


def _set_trace_io(input_data: Any, output_data: Any, span: Any | None = None) -> None:
    """
    Best-effort trace-level input/output update.

    The LangChain callback captures the graph/model/tool internals. This helper
    just keeps the top-level request trace summary useful.
    """
    if span is not None:
        try:
            span.set_trace_io(input=input_data, output=output_data)
            return
        except AttributeError:
            pass

    try:
        langfuse.set_current_trace_io(input=input_data, output=output_data)
    except AttributeError:
        pass


def flush_langfuse() -> None:
    langfuse.flush()


TRunResult = TypeVar("TRunResult")


async def run_with_langfuse_agent_trace(
    *,
    request: str,
    entrypoint: str,
    params: Optional[BaseModel],
    operation: Callable[[dict[str, Any]], Awaitable[TRunResult]],
) -> TRunResult:
    """
    Wrap the top-level IDigBioAgent.run(...) call in a Langfuse span and pass
    Langfuse's LangChain CallbackHandler into the LangChain invocation.

    This keeps agent.py minimal while ensuring Langfuse is integrated with
    LangChain itself via config={"callbacks": [CallbackHandler()]}.
    """
    env = os.getenv("APP_ENV", "beta")
    version = os.getenv("APP_VERSION", "dev")

    trace_input = {
        "request": request,
        "entrypoint": entrypoint,
        "params": _jsonable(params),
    }

    trace_metadata = {
        "agent": "idigbio",
        "entrypoint": entrypoint,
        "env": env,
    }

    tags = ["ichatbio", "idigbio", "agent-run", entrypoint]

    with langfuse.start_as_current_observation(
        as_type="span",
        name="idigbio_agent_run",
        input=trace_input,
        metadata=trace_metadata,
    ) as span:
        try:
            with propagate_attributes(
                user_id="ichatbio_user",
                session_id=f"idigbio:{entrypoint}",
                trace_name="idigbio_agent_run",
                tags=tags,
                metadata=trace_metadata,
                version=version,
            ):
                langfuse_handler = CallbackHandler()

                langchain_config: dict[str, Any] = {
                    "callbacks": [langfuse_handler],
                    "run_name": f"idigbio:{entrypoint}",
                    "metadata": {
                        "agent": "idigbio",
                        "entrypoint": entrypoint,
                        "env": env,
                        "langfuse_tags": tags,
                        "langfuse_user_id": "ichatbio_user",
                        "langfuse_session_id": f"idigbio:{entrypoint}",
                    },
                }

                result = await operation(langchain_config)

            output = {
                "status": "completed",
                "entrypoint": entrypoint,
            }

            span.update(output=output)
            _set_trace_io(trace_input, output, span=span)

            return result

        except Exception as e:
            error_output = {
                "status": "error",
                "entrypoint": entrypoint,
                "error_type": type(e).__name__,
                "error": str(e),
            }

            span.update(
                level="ERROR",
                status_message=str(e),
                output=error_output,
            )
            _set_trace_io(trace_input, error_output, span=span)

            raise

        finally:
            if os.getenv("LANGFUSE_FLUSH_EACH_REQUEST", "false").lower() == "true":
                flush_langfuse()


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
            "{"
            + ",".join([f'"{k}":{url_encode_inner(v)}' for k, v in x.items()])
            + "}"
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
    d = cast(dict, sanitize_json(d))
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


def query_idigbio_api(endpoint: str, params: dict) -> tuple[str, bool, dict | None]:
    params = cast(dict, sanitize_json(params))
    api_url = make_idigbio_api_url(endpoint)
    response = requests.post(api_url, json=params)
    code = (
        f"{response.status_code} {http.client.responses.get(response.status_code, '')}"
    )
    data = response.json() if response.ok else None
    return code, response.ok, data


def query_idigbio_data_api(params) -> tuple[str, bool, dict]:
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


TModel = TypeVar("TModel", bound=BaseModel)


def make_llm_response_model(
    search_parameters_model: Type[TModel],
    validation_callback: Callable[[TModel], None] = None,
):
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
        warnings: Optional[str] = Field(
            None,
            description="If there is any reason why the search parameters might not fully match the request, or for the results of the search to not fully complete the request, explain why here.",
        )
        retry: bool = Field(
            description="Set True if you would be able to fix the warnings by setting different search parameters."
        )

        @model_validator(mode="after")
        def validate_model(self) -> Self:
            if self.search_parameters:
                if not self.artifact_description:
                    raise ValueError(
                        "artifact_description must be provided when search_parameters is present"
                    )

                if self.retry and self.warnings:
                    raise ValueError(
                        "I need to try again to address the following: " + self.warnings
                    )

                if validation_callback:
                    validation_callback(self.search_parameters)

            return self

    return LLMResponseModel


UModel = TypeVar("UModel", bound=BaseModel)


async def generate_search_parameters(
    request: str, system_prompt: str, llm_response_model: UModel
) -> tuple[str, UModel, str, str]:
    try:
        client: AsyncInstructor = instructor.from_openai(
            AsyncOpenAI(**get_llm_client_kwargs())
        )
        result = await client.chat.completions.create(
            model=os.getenv("LLM"),
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

    return (
        result.plan,
        result.search_parameters,
        result.artifact_description,
        result.warnings or "",
    )