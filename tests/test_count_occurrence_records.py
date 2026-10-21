import pytest
from ichatbio.agent_response import ArtifactResponse, ProcessBeginResponse

import tools.count_occurrence_records


@pytest.mark.asyncio
async def test_count_occurrence_records(agent, context, messages):
    await agent.run(context, "Find countries with Rattus rattus", "")

    assert messages[0] == ProcessBeginResponse(summary="Requesting iDigBio statistics")
    artifacts = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert artifacts
    artifact = artifacts[-1]
    assert artifact
    assert artifact.metadata["total_record_count"] > 0
    assert len(artifacts) == 1


@pytest.mark.asyncio
async def test_count_species(agent, context, messages):
    await agent.run(context, "How many bird species in Colombia?", "")

    assert messages[0] == ProcessBeginResponse(summary="Requesting iDigBio statistics")
    artifacts = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert artifacts
    artifact = artifacts[-1]
    assert artifact
    assert artifact.uris[0] == (
        "https://search.idigbio.org/v2/summary/top/records?"
        "top_fields=%22scientificname%22&count=5000&rq=%7B%22class%22:%22Aves%22,%22country%22:%22Colombia%22,%22taxonrank%22:%22species%22%7D"
    )
    assert artifact.metadata["total_record_count"] > 0
    assert len(artifacts) == 1


@pytest.mark.asyncio
async def test_two_top_fields(agent, context, messages):
    await agent.run(context,
                    "Count bird species in Australia using \"stateprovince\" and \"scientifcname\" as top fields", "")

    assert messages[0] == ProcessBeginResponse(summary="Requesting iDigBio statistics")
    artifacts = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert artifacts
    artifact = artifacts[-1]
    assert artifact
    assert artifact.uris[0] == (
        "https://search.idigbio.org/v2/summary/top/records?top_fields=[%22stateprovince%22,%22scientificname%22]&count=5000&rq=%7B%22class%22:%22Aves%22,%22country%22:%22Australia%22%7D"
    )
    assert artifact.metadata["total_record_count"] > 0
    assert len(artifacts) == 1


@pytest.mark.asyncio
async def test_remap_keyword_top_fields(agent, context, messages):
    await agent.run(context, "Who has collected the most bird records?", "")

    assert messages[0] == ProcessBeginResponse(summary="Requesting iDigBio statistics")
    artifacts = [m for m in messages if isinstance(m, ArtifactResponse)]
    assert artifacts
    artifact = artifacts[-1]
    assert artifact
    assert artifact.uris[0] == (
        "https://search.idigbio.org/v2/summary/top/records?top_fields=[%22stateprovince%22,%22scientificname%22]&count=5000&rq=%7B%22class%22:%22Aves%22,%22country%22:%22Australia%22%7D"
    )
    assert artifact.metadata["total_record_count"] > 0
    assert len(artifacts) == 1


def test_get_counts():
    top_counts = {
        "stateprovince": {
            "new south wales": {
                "itemCount": 102943,
                "scientificname": {}
            },
            "queensland": {
                "itemCount": 98768,
                "scientificname": {
                    "trichoglossus haematodus": {
                        "itemCount": 685
                    },
                    "poephila cincta": {
                        "itemCount": 680
                    },
                }
            }
        }
    }
    counts = tools.count_occurrence_records.get_number_of_retrieved_counts(
        ["stateprovince", "scientificname"], top_counts
    )
    assert counts == 2
