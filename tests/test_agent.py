import dotenv
import pytest
import pytest_asyncio
from ichatbio.agent_response import ArtifactResponse

from src.agent import IDigBioAgent


@pytest_asyncio.fixture()
def agent():
    dotenv.load_dotenv()
    return IDigBioAgent()


@pytest.mark.asyncio
async def test_find_occurrence_records(agent, context, messages):
    await agent.run(
        context, "Find records of Rattus rattus", "find_occurrence_records", None
    )

    artifact = next((m for m in messages if isinstance(m, ArtifactResponse)), None)
    assert artifact
    assert artifact.metadata["retrieved_record_count"] > 0


@pytest.mark.asyncio
async def test_abort_on_unsupported_search(agent, context, messages):
    await agent.run(
        context,
        "Find Rattus rattus occurrences near Naja naja occurrences",
        "find_occurrence_records",
        None,
    )

    assert not any((isinstance(m, ArtifactResponse) for m in messages))


@pytest.mark.asyncio
async def test_count_occurrence_records(agent, context, messages):
    await agent.run(
        context, "Find countries with Rattus rattus", "count_occurrence_records", None
    )

    artifact = next((m for m in messages if isinstance(m, ArtifactResponse)), None)
    assert artifact
    assert artifact.metadata["total_record_count"] > 0


@pytest.mark.asyncio
async def test_count_species(agent, context, messages):
    await agent.run(
        context, "How many bird species in Colombia?", "count_occurrence_records", None
    )

    artifact = next((m for m in messages if isinstance(m, ArtifactResponse)), None)
    assert artifact
    assert artifact.uris[0] == (
        "https://search.idigbio.org/v2/summary/top/records?"
        "top_fields=%22scientificname%22&count=5000&rq=%7B%22class%22:%22Aves%22,%22country%22:%22Colombia%22,%22taxonrank%22:%22species%22%7D"
    )
    assert artifact.metadata["total_record_count"] > 0
