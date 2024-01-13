import os
import openai_connectors.chat_connector

def get_llm_service_api_key() -> str:
    return os.getenv("ANYSCALE_API_KEY")

def get_llm_service_base_url() -> str:
    return os.getenv("ANYSCALE_BASE_URL")

def get_ll_model() -> str:
    return openai_connectors.chat_connector.OpenAIModels.ZEPHYR_7B.value