import re
import tiktoken
from icecream import ic
from functools import reduce
from transformers import AutoTokenizer

# every user message follows <|start|>{role/name}\n{content}<|end|>\n
# every reply is primed with <|start|>assistant<|message|>
# depending on the model and existance of name field this amounts 3 or 4.
_CHAT_MESSAGE_PADDING_TOKENS = 0

# these are the currently supported context window limit (token limit of the entire chat thread) in each model
# this can change in future and will need updating
CONTEXT_WINDOW = {
    # this is based on https://app.endpoints.anyscale.com/docs
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
    "mistralai/Mistral-7B-Instruct-v0.1": 16384,
    "HuggingFaceH4/zephyr-7b-beta": 16384,
    "codellama/CodeLlama-34b-Instruct-hf": 16384,
    "meta-llama/Llama-2-13b-chat-hf": 4096,

    # this is based on https://platform.openai.com/docs/models/overview
    "gpt-4-1106-preview": 128000,
    "gpt-3.5-turbo-1106": 16385
}

# in addition to context window each message also has a max allowed size
MESSAGE_TOKEN_LIMIT = {
    # this is based on https://app.endpoints.anyscale.com/docs
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 2048, 
    "mistralai/Mistral-7B-Instruct-v0.1": 2048,
    "HuggingFaceH4/zephyr-7b-beta": 512, 
    "codellama/CodeLlama-34b-Instruct-hf": 1024, 
    "meta-llama/Llama-2-13b-chat-hf": 512,    
    "thenlper/gte-large": 512,  # embedding model
        
    # this is based on https://platform.openai.com/docs/models/overview
    "gpt-4-1106-preview": 4096, 
    "gpt-3.5-turbo-1106": 4096, 
    "text-embedding-ada-002": 1024 # embedding model. this is actually 8191, i am cutting this down to retain quality of content
}

# this lambda is used multiple times in the following codes
_add_func = lambda x, y: x + y

# counts the number of tokens in a string
def count_tokens(text: str, model: str) -> int: 
    try: # this works for chatgpt/openai.com models
        return len(tiktoken.encoding_for_model(model).encode(text))
    except: # else try with the open source models from hugging face (hosted by anyscale)
        return len(AutoTokenizer.from_pretrained(model).tokenize(text))

# counts the number of token for 1 message
def count_tokens_for_message(message, model) -> int:
    return _CHAT_MESSAGE_PADDING_TOKENS + reduce(
        _add_func,
        [count_tokens(value, model) for value in message.values()]
    )

# counts the number of tokens in an entire thread of messages
def count_tokens_for_messages(messages, model) -> int:
    return reduce(
        _add_func,
        [count_tokens_for_message(msg, model) for msg in messages]
    )

# truncates the content to the message limit of the model
def truncate_text(text: str, model: str) -> str:  
    try: # this works for chatgpt/openai.com models
        encoding = tiktoken.encoding_for_model(model)
        return encoding.decode(encoding.encode(text)[:MESSAGE_TOKEN_LIMIT[model]])
    except: # else try with the open source models from hugging face (hosted by anyscale)
        tokenizer = AutoTokenizer.from_pretrained(model)
        return tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:MESSAGE_TOKEN_LIMIT[model]])

# natural language delimeter sequence:
# - section: "\n\n" (2 or more \n)
# - line: "\n" (1 line break)
# - sentence: ". " (dot and space) 
# for codes this would be different
NATURAL_LANGUAGE_DELIMITERS = ["\n\n", "\n", ". ", "? ", "! "]

# binary split the text based on delimiter sequence and then returned the padded text
# if no metadata_func is defined it will assume that the message is for chat. if metadata_func == None it will dead with the original text
# NOTE: gte-large has a known issue where it will NOT retain the semantics of the text given and will return tokens joined by " "
def split_content(text: str, model: str, delimiter_sequence = NATURAL_LANGUAGE_DELIMITERS, metadata_func = None) -> list[str]:
    text = text.strip() # remove leading and trailing whitespaces. By themselves they dont mean anything
    if not text: # if there is no content left after strip return
        return []

    content = metadata_func(text) if metadata_func != None else text    
    # the whole content is higher than the token limit.
    # so try to split in equal chucks based on delimeter sequence 
    if count_tokens(content, model) > MESSAGE_TOKEN_LIMIT[model]:
        for delimiter in delimiter_sequence:  
            # splitting in 2 nearly equal sized parts help retain chunks of equal size and hence as much context as possible for both chunks     
            # Note: send the text and NOT the whole padded content, because the idea is that each chunk needs to have padded metadata for reserving the content 
            # Note: this will do a rough cut in the middle                
            chunks = _split_in_half(text, model, delimiter)
            # there are 2 chunks. process them recursively
            if len(chunks) > 1:
                return reduce(_add_func, [split_content(text = c, model = model, delimiter_sequence = delimiter_sequence, metadata_func=metadata_func) for c in chunks])                
            # or else there is only 1 chunk and so move to the next delimeter to split it even further
            
    # either the padded content was less than the limit or 
    # we tried chunking and its not going to get any smaller. so just truncate the content and return
    # this can happen if the padding content or a sentence is too large. you get what you get!
    return [truncate_text(text = content, model = model)] 

def _split_in_half(text: str, model: str, delimiter):    
    # scrape out the empty strings, they are not going to value anyway
    chunks = [c for c in text.split(delimiter) if c]
    # its already split in half or cannot be split any more so just return what you got
    if len(chunks) <=2:
        return chunks
    else:
        halfway = count_tokens(text, model) >> 1
        # this way there will always be at least 1 item on each side and no side will be empty
        # in corner cases halfway token point can be somewhere in the first item or the last item
        # if it is in the first item then the loop will break at i == 1 and result will be chunks[0] & chunks[1 --> end]
        # if the halfway point is on the last item the loop will break i = len - 1 and the result will be chunks[0-(len-1)] & chunks[(len-1)]
        for i in range(1, len(chunks)):
            # the left side token count is over the limit anyway
            if count_tokens(delimiter.join(chunks[:i]), model) >= halfway:
                break
        return [delimiter.join(chunks[:i]), delimiter.join(chunks[i:])]   
    