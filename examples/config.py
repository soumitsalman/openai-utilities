import os

# I am using environment variables. That does not mean, you have to as well. Feel free to pass literals directly

def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY")

def get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL")

def get_chat_model() -> str:
    return os.getenv("OPENAI_CHAT_MODEL")

def get_embeddings_model() -> str:
    return os.getenv("OPENAI_EMBEDDINGS_MODEL")
    # return "thenlper/gte-large"

def get_org_id() -> str:
    return os.getenv("OPENAI_ORG_ID")