import pytest
from ichatbio.agent_response import ArtifactResponse, ProcessBeginResponse


@pytest.mark.asyncio
async def test_find_occurrence_records(agent, context, messages):
    await agent.run(
        context, "Find records of Rattus rattus", "find_occurrence_records", None
    )

    assert messages[0] == ProcessBeginResponse(summary="Searching iDigBio occurrence records")
    artifacts = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert artifacts
    artifact = artifacts[0]
    assert artifact
    assert artifact.metadata["retrieved_record_count"] > 0
    assert len(artifacts) == 1


@pytest.mark.asyncio
async def test_abort_on_unsupported_search(agent, context, messages):
    await agent.run(
        context,
        "Find Rattus rattus occurrences near Naja naja occurrences",
        "find_occurrence_records",
        None,
    )

    assert not [m for m in messages if isinstance(m, ArtifactResponse)]
