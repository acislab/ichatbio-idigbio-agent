import dotenv
import langchain_core.globals
import pytest_asyncio

from agent import IDigBioAgent


@pytest_asyncio.fixture()
def agent():
    langchain_core.globals.set_debug(True)
    dotenv.load_dotenv()
    return IDigBioAgent()
