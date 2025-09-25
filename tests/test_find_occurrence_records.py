import pytest
from ichatbio.agent_response import ArtifactResponse


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
