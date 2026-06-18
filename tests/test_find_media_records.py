import pytest
from ichatbio.agent_response import ArtifactResponse, ProcessBeginResponse


@pytest.mark.asyncio
async def test_find_media_records(agent, context, messages):
    await agent.run(context, "Find pictures of Rattus rattus", "find_media_records")

    assert messages[0] == ProcessBeginResponse(summary="Searching iDigBio media records")
    artifacts = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert artifacts
    artifact = artifacts[0]
    assert artifact
    assert artifact.metadata["retrieved_record_count"] > 0

    # Make sure the agent is outputting links to view records in iDigBio
    assert "https://portal.idigbio.org/portal/mediarecords/" in str(messages)

    assert len(artifacts) == 1


@pytest.mark.asyncio
async def test_abort_on_unsupported_proximity_search(agent, context, messages):
    await agent.run(
        context,
        "Find media for Rattus rattus occurrences near Naja naja occurrences",
        "find_media_records"
    )

    assert not any((isinstance(m, ArtifactResponse) for m in messages))


@pytest.mark.asyncio
async def test_abort_on_unsupported_semantics_search(agent, context, messages):
    await agent.run(
        context,
        "Find pictures of blue butterflies",
        "find_media_records"
    )

    assert not any((isinstance(m, ArtifactResponse) for m in messages))
