import tiktoken
from icecream import ic
from functools import reduce

# these are the currently supported context window limit (token limit of the entire chat thread) in each model
# this can change in future and will need updating
CONTEXT_WINDOW = {
    # this is based on https://app.endpoints.anyscale.com/docs
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
    "mistralai/Mistral-7B-Instruct-v0.1": 16384,
    "HuggingFaceH4/zephyr-7b-beta": 16384,
    "codellama/CodeLlama-34b-Instruct-hf": 16384,
    "meta-llama/Llama-2-13b-chat-hf": 4096,   
    "thenlper/gte-large": 512, #embedding

    # this is based on https://platform.openai.com/docs/models/overview
    "gpt-4-1106-preview": 128000,
    "gpt-3.5-turbo-1106": 16385,
    "text-embedding-ada-002": 8192 # embedding model
}

# in addition to context window each message also has a max allowed size
MESSAGE_TOKEN_LIMIT = {
    # this is based on https://app.endpoints.anyscale.com/docs
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 1024,
    "mistralai/Mistral-7B-Instruct-v0.1": 1024,
    "HuggingFaceH4/zephyr-7b-beta": 512,
    "codellama/CodeLlama-34b-Instruct-hf": 1024,
    "meta-llama/Llama-2-13b-chat-hf": 512,    
    "thenlper/gte-large": 512, # embedding model
        
    # this is based on https://platform.openai.com/docs/models/overview
    "gpt-4-1106-preview": 4096,
    "gpt-3.5-turbo-1106": 4096,
    "text-embedding-ada-002": 2047 # embedding model
}

# every user message follows <|start|>{role/name}\n{content}<|end|>\n
# every reply is primed with <|start|>assistant<|message|>
# depending on the model and existance of name field this amounts 3 or 4.
MSG_PADDING_BUFFER_TOKENS = 4

# this lambda is used multiple times in the following codes
ADD_FUNC = lambda x, y: x + y

# counts the number of tokens in a string
def count_tokens(content: str, model: str) -> int:    
    try:
        encoding = tiktoken.get_encoding(model)
    except:
        # cl100k_base is default encoding model
        encoding = tiktoken.get_encoding("cl100k_base") 
    tokens = encoding.encode(content)
    return len(tokens)

# counts the number of token for 1 message
def count_tokens_for_message(message, model) -> int:
    return MSG_PADDING_BUFFER_TOKENS + reduce(
        ADD_FUNC,
        [count_tokens(value, model) for value in message.values()]
    )

# counts the number of tokens in an entire thread of messages
def count_tokens_for_messages(messages, model) -> int:
    return reduce(
        ADD_FUNC,
        [count_tokens_for_message(msg, model) for msg in messages]
    )

# splits the content into multiple message contents if the content exceeds the max token limit of a message for that model
def split_content(content: str, model: str) -> list[str]:
    token_count = count_tokens(content, model)
    if token_count + MSG_PADDING_BUFFER_TOKENS < MESSAGE_TOKEN_LIMIT[model]:
        # the content is not exceeding the limit so keep it the way it is
        return [content]
    else:        
        # there would be roughly split_count number of messages in a sequence (rounding up the division result)
        split_count = ((token_count + MSG_PADDING_BUFFER_TOKENS) // MESSAGE_TOKEN_LIMIT[model]) + 1
        # now do a rough split based on whitespace
        words = content.split()
        word_count_per_split = len(words) // split_count
        # combine the words with space
        combined_strings = [" ".join(words[i:i+word_count_per_split]) for i in range(0, len(words), word_count_per_split)]        
        return combined_strings
