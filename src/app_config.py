from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    llm_base_url: str | None = Field(None, alias="OPENAI_BASE_URL")
    llm: str = Field("gpt-5.6-terra")
    reasoning_effort: str = Field("high")
    use_llm_proxy: bool = Field(True)

def settings() -> Settings:
    return Settings.settings_cached()

def reload():
    Settings.settings_clear_cache()