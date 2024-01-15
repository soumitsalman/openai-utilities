import os

def get_llm_service_api_key() -> str:
    return os.getenv("LLM_SERVICE_API_KEY")

def get_llm_service_base_url() -> str:
    return os.getenv("LLM_SERVICE_BASE_URL")

def get_llm_service_model() -> str:
    return os.getenv("LLM_SERVICE_MODEL")

def get_llm_service_organization() -> str:
    return os.getenv("LLM_SERVICE_ORGANIZATION")