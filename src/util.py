from __future__ import annotations

import asyncio
import http.client
import json
import os
from enum import Enum
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Self,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import instructor
import requests
from instructor import AsyncInstructor
from instructor.exceptions import InstructorRetryException
from langfuse import get_client, propagate_attributes
from langfuse.openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from pydantic.functional_validators import model_validator
from pydantic_core import ValidationError
from tenacity import AsyncRetrying, RetryCallState
from tenacity.stop import stop_base


# ---------------------------------------------------------------------------
# Langfuse configuration
# ---------------------------------------------------------------------------
# Prefer setting these in the deployment environment or through dotenv before
# importing this module.
#
# Example:
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-f4ceb77c-8d27-45c9-ab49-c22dec93e357"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-a59c1ba4-8772-4c98-9029-fe8b8340ac93"
os.environ["LANGFUSE_BASE_URL"] = "http://10.13.45.225:3000"#
# Do not create the OpenAI client at import time. The lazy getter below avoids
# import-time failures in tests/API startup.

os.environ["LANGFUSE_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED"] = "true"

langfuse = get_client()

# Lazily-created centralized Langfuse/OpenAI/Instructor client.
#
# Important:
# - AsyncOpenAI must come from langfuse.openai, not openai.
# - Instructor patches that Langfuse-wrapped client.
# - The client is created lazily so importing util.py does not require
#   OPENAI_API_KEY to already be present.
client: AsyncInstructor | None = None

# Simple rate limiter for feedback classification.
sem = asyncio.Semaphore(5)


def get_instructor_client() -> AsyncInstructor:
    """
    Lazily create the Langfuse-wrapped Instructor client.

    This avoids requiring OPENAI_API_KEY at import time. The API key only needs
    to exist by the time an LLM function is actually called.
    """
    global client

    if client is None:
        openai_client = AsyncOpenAI()

        # Explicit TOOLS mode keeps structured extraction behavior predictable.
        # If your installed Instructor version does not support patch() with an
        # async OpenAI client, the fallback preserves compatibility with apatch().
        try:
            client = cast(
                AsyncInstructor,
                instructor.patch(
                    openai_client,
                    mode=instructor.Mode.TOOLS,
                ),
            )
        except TypeError:
            client = instructor.apatch(openai_client)

    return client


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    """
    Convert Pydantic models, enums, tuples, lists, and dicts to values that are
    safe to send to Langfuse as input/output/metadata.
    """
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


def _current_trace_id() -> str | None:
    """
    Compatibility helper. Langfuse exposes current trace IDs from the active
    OpenTelemetry context. If unavailable, return None rather than breaking
    the request path.
    """
    try:
        return langfuse.get_current_trace_id()
    except Exception:
        return None


def _current_observation_id() -> str | None:
    """
    Compatibility helper. Langfuse exposes current observation IDs from the
    active OpenTelemetry context. If unavailable, return None rather than
    breaking the request path.
    """
    try:
        return langfuse.get_current_observation_id()
    except Exception:
        return None


def _set_trace_io(input_data: Any, output_data: Any, span: Any | None = None) -> None:
    """
    Best-effort trace-level IO update.

    This helps when the API path creates a root observation above this function
    and Langfuse's trace summary would otherwise appear blank even though the
    child span has input/output.
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


def _create_score(
    *,
    trace_id: str,
    name: str,
    value: float,
    observation_id: str | None = None,
):
    """
    Create a Langfuse score while remaining tolerant of small SDK API changes.
    """
    try:
        kwargs: dict[str, Any] = {
            "trace_id": trace_id,
            "name": name,
            "value": value,
        }

        if observation_id:
            kwargs["observation_id"] = observation_id

        return langfuse.create_score(**kwargs)

    except AttributeError:
        # Compatibility with newer score namespaces if create_score() is absent.
        payload: dict[str, Any] = {
            "traceId": trace_id,
            "name": name,
            "value": value,
        }

        if observation_id:
            payload["observationId"] = observation_id

        return langfuse.score.create(**payload)


def flush_langfuse():
    """
    Call this from API shutdown hooks, serverless cleanup, CLI jobs, or tests.
    Avoid flushing in every API request unless you explicitly need synchronous
    delivery before returning.
    """
    langfuse.flush()


# ---------------------------------------------------------------------------
# Validation / retry handling
# ---------------------------------------------------------------------------

def _get_terminal_validation_error(e: Exception | None):
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
                f"Error: AI failed to generate valid output after "
                f"{e.n_attempts} attempts."
            )

        self.message = "\n\n".join(messages)
        super().__init__(self.message)


class StopOnTerminalErrorOrMaxAttempts(stop_base):
    """Stop when a terminal validation error is encountered or max attempts is hit."""

    def __init__(self, max_attempts: int):
        self.max_attempts = max_attempts

    def __call__(self, retry_state: RetryCallState) -> bool:
        exception = (
            retry_state.outcome.exception()
            if retry_state.outcome is not None
            else None
        )

        if _get_terminal_validation_error(exception):
            return True

        return retry_state.attempt_number >= self.max_attempts


# ---------------------------------------------------------------------------
# URL / JSON helpers
# ---------------------------------------------------------------------------

JSON = Union[dict, list, str, int, float]


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


# ---------------------------------------------------------------------------
# iDigBio API helpers
# ---------------------------------------------------------------------------

def query_idigbio_api(endpoint: str, params: dict) -> tuple[str, bool, dict | None]:
    params = cast(dict, sanitize_json(params))
    api_url = make_idigbio_api_url(endpoint)
    response = requests.post(api_url, json=params)

    code = (
        f"{response.status_code} "
        f"{http.client.responses.get(response.status_code, '')}"
    )

    data = response.json() if response.ok else None
    return code, response.ok, data


def query_idigbio_data_api(params) -> tuple[str, bool, dict]:
    sanitized_query = sanitize_json(params.get("rq", {}))

    api_params = {
        "rq": json.dumps(sanitized_query),
        "email": params.get("email", ""),
    }

    response = requests.post("https://api.idigbio.org/v2/download", data=api_params)

    code = (
        f"{response.status_code} "
        f"{http.client.responses.get(response.status_code, '')}"
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


# ---------------------------------------------------------------------------
# LLM response model factory
# ---------------------------------------------------------------------------

TModel = TypeVar("TModel", bound=BaseModel)


def make_llm_response_model(
    search_parameters_model: Type[TModel],
    validation_callback: Callable[[TModel], None] | None = None,
):
    class LLMResponseModel(BaseModel):
        plan: str = Field(
            description=(
                "A brief explanation of what API parameters you plan to use. "
                "Or, if you are unable to fulfill the user's request using the "
                "available API parameters, provide a brief explanation for why "
                "you cannot retrieve the requested records."
            )
        )

        search_parameters: Optional[search_parameters_model] = Field(
            None,
            description=(
                "The search parameters to use to produce the requested media "
                "records. If you are unable to fulfill the user's request using "
                "the available API parameters, leave this field unset to abort."
            ),
        )

        artifact_description: Optional[str] = Field(
            None,
            description=(
                "A concise characterization of the retrieved occurrence record "
                "data, if any."
            ),
        )

        warnings: Optional[str] = Field(
            None,
            description=(
                "If there is any reason why the search parameters might not "
                "fully match the request, or for the results of the search to "
                "not fully complete the request, explain why here."
            ),
        )

        retry: bool = Field(
            description=(
                "Set True if you would be able to fix the warnings by setting "
                "different search parameters."
            )
        )

        @model_validator(mode="after")
        def validate_model(self) -> Self:
            if self.search_parameters:
                if not self.artifact_description:
                    raise ValueError(
                        "artifact_description must be provided when "
                        "search_parameters is present"
                    )

                if self.retry and self.warnings:
                    raise ValueError(
                        "I need to try again to address the following: "
                        + self.warnings
                    )

                if validation_callback:
                    validation_callback(self.search_parameters)

            return self

    return LLMResponseModel


UModel = TypeVar("UModel", bound=BaseModel)


# ---------------------------------------------------------------------------
# Feedback classification
# ---------------------------------------------------------------------------

class FeedbackType(Enum):
    PRAISE = "PRAISE"
    SUGGESTION = "SUGGESTION"
    BUG = "BUG"
    QUESTION = "QUESTION"


class FeedbackClassification(BaseModel):
    feedback_text: str = Field(...)
    classification: List[FeedbackType] = Field(
        description="Predicted categories for the feedback"
    )
    relevance_score: float = Field(
        default=0.0,
        description=(
            "Score of the query evaluating its relevance to the business "
            "between 0.0 and 1.0"
        ),
    )

    @field_validator("classification", mode="before")
    def validate_classification(cls, v):
        if not isinstance(v, list):
            v = [v]
        return v


def score_relevance(
    trace_id: str,
    observation_id: str | None,
    relevance_score: float,
):
    """
    Score the relevance of a feedback query in Langfuse.
    """
    _create_score(
        trace_id=trace_id,
        observation_id=observation_id,
        name="feedback-relevance",
        value=relevance_score,
    )


async def classify_feedback(
    feedback: str,
) -> tuple[str, FeedbackClassification, str | None]:
    """
    Classify customer feedback into categories and evaluate relevance.

    Return shape intentionally mirrors the previous Langfuse example pattern:
        feedback, classification_response, observation_id

    The relevance score is now also written to Langfuse from inside this helper
    when a trace ID is available.
    """
    span_input = {
        "feedback": feedback,
    }

    with propagate_attributes(
        user_id="ichatbio_user",
        session_id="session_ichatbio",
        trace_name="classify_feedback",
        tags=["ichatbio", "feedback-classification"],
        metadata={
            "agent": "ichatbio",
            "operation": "classify_feedback",
            "env": "beta",
        },
        version="1.0",
    ):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="classify_feedback",
            input=span_input,
        ) as span:
            try:
                async with sem:
                    llm_client = get_instructor_client()

                    response = await llm_client.chat.completions.create(
                        model="gpt-4o",
                        response_model=FeedbackClassification,
                        max_retries=2,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "Classify and score this feedback: "
                                    f"{feedback}"
                                ),
                            },
                        ],
                    )

                categories = [c.value for c in response.classification]

                output = {
                    "feedback_text": response.feedback_text,
                    "classification": categories,
                    "relevance_score": response.relevance_score,
                }

                span.update(
                    output=output,
                    metadata={
                        "classification": categories,
                        "relevance_score": response.relevance_score,
                    },
                )

                _set_trace_io(span_input, output, span=span)

                trace_id = getattr(span, "trace_id", None) or _current_trace_id()
                observation_id = getattr(span, "id", None) or _current_observation_id()

                if trace_id:
                    score_relevance(
                        trace_id=trace_id,
                        observation_id=observation_id,
                        relevance_score=response.relevance_score,
                    )

                    # Optional: make categories queryable as individual boolean scores.
                    for category in categories:
                        _create_score(
                            trace_id=trace_id,
                            observation_id=observation_id,
                            name=f"feedback-category-{category.lower()}",
                            value=1.0,
                        )

                return feedback, response, observation_id

            except Exception as e:
                span.update(
                    level="ERROR",
                    status_message=str(e),
                    output={"error": str(e)},
                )
                raise


# ---------------------------------------------------------------------------
# Search parameter generation
# ---------------------------------------------------------------------------

async def generate_search_parameters(
    request: str,
    system_prompt: str,
    llm_response_model: UModel,
) -> tuple[str, UModel, str, str]:
    """
    Generate iDigBio search parameters using the shared Langfuse-wrapped
    Instructor client.

    This function no longer relies on @observe. It creates an explicit active
    Langfuse span and updates input/output manually, which avoids blank
    input/output fields when called through API layers.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]

    span_input = {
        "request": request,
        "messages": messages,
        "response_model": getattr(
            llm_response_model,
            "__name__",
            str(llm_response_model),
        ),
    }

    with propagate_attributes(
        user_id="ichatbio_user",
        session_id="session_ichatbio",
        trace_name="generate_search_parameters",
        tags=["ichatbio", "search-parameters"],
        metadata={
            "agent": "ichatbio",
            "operation": "generate_search_parameters",
            "experiment": "variant_a",
            "env": "beta",
        },
        version="1.0",
    ):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="generate_search_parameters",
            input=span_input,
        ) as span:
            try:
                llm_client = get_instructor_client()

                result = await llm_client.chat.completions.create(
                    model="gpt-4.1",
                    temperature=0,
                    response_model=llm_response_model,
                    messages=messages,
                    max_retries=AsyncRetrying(
                        stop=StopOnTerminalErrorOrMaxAttempts(3)
                    ),
                )

                output = {
                    "plan": result.plan,
                    "search_parameters": _jsonable(result.search_parameters),
                    "artifact_description": result.artifact_description,
                    "warnings": result.warnings or "",
                }

                span.update(output=output)

                _set_trace_io(span_input, output, span=span)

                return (
                    result.plan,
                    result.search_parameters,
                    result.artifact_description,
                    result.warnings or "",
                )

            except InstructorRetryException as e:
                span.update(
                    level="ERROR",
                    status_message=str(e),
                    output={"error": str(e)},
                )

                _set_trace_io(
                    span_input,
                    {"error": str(e)},
                    span=span,
                )

                raise AIGenerationException(e) from e

            except Exception as e:
                span.update(
                    level="ERROR",
                    status_message=str(e),
                    output={"error": str(e)},
                )

                _set_trace_io(
                    span_input,
                    {"error": str(e)},
                    span=span,
                )

                raise

from typing import Awaitable

async def run_with_langfuse_agent_trace(
    *,
    request: str,
    entrypoint: str,
    params: Optional[BaseModel],
    operation: Callable[[], Awaitable[None]],
) -> None:
    """
    Wrap the top-level agent.run(...) call in a Langfuse root span.

    This is the shared boundary used by both:
      - tests that call IDigBioAgent.run(...) directly
      - the real API path that eventually calls IDigBioAgent.run(...)
    """
    trace_input = {
        "request": request,
        "entrypoint": entrypoint,
        "params": _jsonable(params),
    }

    with langfuse.start_as_current_observation(
        as_type="span",
        name="idigbio_agent_run",
        input=trace_input,
        metadata={
            "agent": "idigbio",
            "entrypoint": entrypoint,
            "env": "beta",
        },
    ) as span:
        try:
            with propagate_attributes(
                user_id="ichatbio_user",
                session_id="session_ichatbio",
                trace_name="idigbio_agent_run",
                tags=["ichatbio", "idigbio", "agent-run"],
                metadata={
                    "agent": "idigbio",
                    "entrypoint": entrypoint,
                    "env": "beta",
                },
                version="1.0",
            ):
                await operation()

            output = {
                "status": "completed",
                "entrypoint": entrypoint,
            }

            span.update(output=output)

            try:
                span.set_trace_io(input=trace_input, output=output)
            except AttributeError:
                langfuse.set_current_trace_io(input=trace_input, output=output)

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

            try:
                span.set_trace_io(input=trace_input, output=error_output)
            except AttributeError:
                langfuse.set_current_trace_io(
                    input=trace_input,
                    output=error_output,
                )

            raise

        finally:
            flush_langfuse()
