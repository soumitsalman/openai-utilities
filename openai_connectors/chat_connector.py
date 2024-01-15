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

    # model: the language model that would be used for this chat session. This cannot be None. Make sure this is supported by the service provider. model value can be changed later
    # api_key: the llm service provider's api_key e.g. API KEY or personal access token for anyscale.com of openai.com public endpoints. this cannot be none
    # organization: optional identifier for tenant of the api_key defined by the llm service provider.
    # base_url: the endpoint of the llm service. None value will default to openai.com (https://api.openai.com/v1)
    # instructions: optional instructions if you want the chat bot to function a certain way. This str or list[str]. Setting instructions value is specially important if you want the output format to be specific to your preference e.g. JSON blob
    # json_mode: If you want the output message text to be formatted as a json blob. Note that you would also include an instruction text saying so e.g. provide all outputs in JSON.
    # schema: if json_mode is set to True you can optionally specify the schema. This needs to be formatted as defined by https://json-schema.org/
    # BUG: anyscale endpoints cannot seem to support more than 1 system message. So instructions has to be 1 str.
    def __init__(
            self, 
            model: str,                       
            api_key: str, 
            organization: str = None, 
            base_url: str = None,
            instructions: str = None,
            json_mode: bool = False,
            schema = None):
        
        self.openai_client = openai.OpenAI(api_key=api_key, organization=organization, base_url=base_url)
        self.model = model
        self.instructions = instructions
        if json_mode:
            self.response_format = {"type": "json_object"}
            if schema != None:
                self.response_format["schema"] = schema
            self.temperature = 0.1 # to keep output consistent
        else:
            self.response_format = None
            self.temperature = 0.75 # adding some default entropy to answers.
        self.thread = []
        self.reset_window()

    # this function resets the context window.
    # this is a private utility function for the class itself
    # ext_thread is the initialization value
    def reset_window(self, ext_thread = None):
        self.thread = []
        if isinstance(self.instructions, str):        
            self.thread = [create_message("system", self.instructions)]
        elif isinstance(self.instructions, list):
            self.thread = [create_message("system", inst) for inst in self.instructions]       
        if ext_thread != None:
            self.thread = self.thread + ext_thread
        return self.thread

    def update_model(self, new_model: str):
        if MESSAGE_TOKEN_LIMIT[new_model] < MESSAGE_TOKEN_LIMIT[self.model]:
            # the new model has a lower token limit so adjust all the messages in a thread
            new_thread = []
            for msg in self.thread:
                new_thread = new_thread + [create_message(msg['role'], msg_fraction, msg.get('name')) for msg_fraction in split_content(msg["content"], new_model)]
            self.thread = new_thread
        # context window will be taken care of next time when run_thread runs         
        self.model = new_model
    
    # this is a private utility function
    def get_chat_completion(self, ext_thread):
        # TODO: fix it for function call
        if self.response_format == None:
            resp = self.openai_client.chat.completions.create(
                model = self.model,
                messages = ext_thread,
                temperature = self.temperature,
                seed = 10000 # a random number to keep the response consistent through out the context window
            )
        else:
            resp = self.openai_client.chat.completions.create(
                model = self.model,
                messages = ext_thread,
                temperature = self.temperature,
                seed = 10000, # a random number to keep the response consistent
                response_format=self.response_format
            ) 
        
        return resp.choices[0].message
    
    # this is a private utility function aimed to summarize the conversation in a thread
    # this function is used iteratively by run_thread compress the content in the thread
    def create_summary_message(self, messages):
        SUMMARY_MESSAGE_INSTRUCTION = f"create a summary of this conversation in less than {MESSAGE_TOKEN_LIMIT[self.model]} tokens. Prefix the response with the word CURRENT CONTEXT:"
        # existing system messages are not important for creating summary
        messages = messages + [create_message("user", SUMMARY_MESSAGE_INSTRUCTION)]
        # save the context as a user message
        # TODO: check if it is better to save it as system message
        return create_message("user", self.get_chat_completion(messages).content)

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
    def run_thread(self, json_mode = False):
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

    
    