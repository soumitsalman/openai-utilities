from enum import Enum
import openai
from .tokenutils import count_tokens_for_messages, count_tokens_for_message, split_content, MESSAGE_TOKEN_LIMIT, CONTEXT_WINDOW
from .retryutils import retry_after_random_wait
from icecream import ic

# the currently supported models in this code
class OpenAIModels(Enum):
    MISTRAL_8X_7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1"
    ZEPHYR_7B = "HuggingFaceH4/zephyr-7b-beta"
    CODELLAMA_34B = "codellama/CodeLlama-34b-Instruct-hf"
    METALLAMA_13B = "meta-llama/Llama-2-13b-chat-hf"
    GPT4_1106 = "gpt-4-1106-preview"
    GPT3_5_TURBO = "gpt-3.5-turbo-1106"

# internal utility function for creating a message format for openai client
def create_message(role: str, msg: str, name: str = None) -> dict[str, str]:
    msg = {"role": role, "content": msg}
    if name != None:
        msg["name"] = name
    return msg

# wrapper on top of python openai SDK/Driver which is used by both openai and anyscale
# this class manages a conversational thread with some built in error handling for exceeding context window and rate limiting
class OpenAIChatSession:

    # chat_model cannot be None
    # instruction can be None
    # service_api_key cannot be None
    # service_org_id is primarily for openai hosted services
    # service_url is primarily for anyscale hosted services
    def __init__(
            self, 
            model: str,
            instruction: str,            
            service_api_key: str, 
            service_org_id: str = None, 
            service_url: str = None):
        
        self.openai_client = openai.OpenAI(api_key=service_api_key, organization=service_org_id, base_url=service_url)
        self.model = model
        self.instruction = instruction
        self.thread = []
        self.reset_window()

    # this function resets the context window.
    # this is a private utility function for the class itself
    # ext_thread is the initialization value
    def reset_window(self, ext_thread = None):
        self.thread = []
        if self.instruction != None:
            self.thread = [create_message("system", self.instruction)]        
        if ext_thread != None:
            self.thread = self.thread + ext_thread
        return self.thread

    # the retry decorator here is for dealing with rate limit. This number should change with different services
    @retry_after_random_wait(min_wait=61, max_wait=240, retry_count=5, errors=(openai.RateLimitError))
    # this is a private utility function
    def get_chat_completion(self, ext_thread):
        # TODO: fix it for function call
        resp = self.openai_client.chat.completions.create(
            model = self.model,
            messages = ext_thread,
            temperature = 0.7, # TODO: this should become configurable
            seed = 10000 # random number to keep the response consistent
        )
        return resp.choices[0].message
    
    # this is a private utility function aimed to summarize the conversation in a thread
    # this function is used iteratively by run_thread compress the content in the thread
    def create_summary_message(self, messages):
        SUMMARY_MESSAGE_INSTRUCTION = f"create a summary of this conversation in less than {MESSAGE_TOKEN_LIMIT[self.model]} tokens. Prefix the response with the word CONTEXT SUMMARY:"
        # instruction message is not important for creating summary
        messages = messages + [create_message("user", SUMMARY_MESSAGE_INSTRUCTION)]
        return create_message("assistant", self.get_chat_completion(messages).content)

    # internal utility function for creating a summerized thread from the messages
    # the summarization of the conversation would be represented as a user message
    def create_summerized_thread(self, messages):
        group = []
        group_token_count = 0        
        token_limit = CONTEXT_WINDOW[self.model] - MESSAGE_TOKEN_LIMIT[self.model]
        
        for msg in messages:
            msg_token_count = count_tokens_for_message(msg, self.model)
            if msg_token_count + group_token_count >= token_limit:
                # get a summarized message since it passed the limit
                summary = self.create_summary_message(group)
                group = [summary]
                group_token_count += count_tokens_for_message(summary, self.model)
            # keep adding the new message to group
            group.append(msg)
            group_token_count += msg_token_count

        return group

    # splits large messages and adds to the thread so that openai api doesnt die
    def add_to_thread(self, content, name = None):      
        self.thread = self.thread + [create_message("user", msg, name) for msg in split_content(content, self.model)]
        return self.thread

    # passes the entire existing thread to the service for running.
    # this takes care of the context window if it gets bigger. 
    # it summarizes the existing content and creates 1 message for context
    def run_thread(self):
        # checking if the current thread exceeds the context window
        window_token_count = count_tokens_for_messages(self.thread, self.model)
        # leave some room for response so minus the message token limit
        if window_token_count >= CONTEXT_WINDOW[self.model] - MESSAGE_TOKEN_LIMIT[self.model]:
            # this needs summarization            
            self.thread = self.reset_window(self.create_summerized_thread(self.thread))            

        resp = self.get_chat_completion(self.thread)
        # response messages can be added to the context window since if the window was larger it would failed anyway
        self.thread.append(create_message(resp.role, resp.content))
        return resp.content 

    
    