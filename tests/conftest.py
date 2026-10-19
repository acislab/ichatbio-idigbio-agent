import dotenv
import langchain_core.globals
import pytest_asyncio

import app_config
from agent import IDigBioAgent


@pytest_asyncio.fixture()
def agent():
    langchain_core.globals.set_debug(True)
    dotenv.load_dotenv()
    app_config.reload()
    return IDigBioAgent()
