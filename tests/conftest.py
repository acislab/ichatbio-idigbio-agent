import dotenv
import pytest_asyncio

from agent import IDigBioAgent


@pytest_asyncio.fixture()
def agent():
    dotenv.load_dotenv()
    return IDigBioAgent()
