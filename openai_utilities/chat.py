from enum import Enum
import os
import openai
from .tokenutils import count_tokens_for_messages, count_tokens_for_message, split_content, MESSAGE_TOKEN_LIMIT, CONTEXT_WINDOW
from icecream import ic
from functools import reduce

# the currently supported models in this code
class ChatModels(Enum):
    MISTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
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

# internal lambda function for adding 2 items. this is used in reduce functions
_add = lambda a, b: a+b

# CONTEXT WINDOW MANAGEMENT:
# option 1: run and forget -- no context
# option 2: sliding window -- shave of a chunk from the top/old messages
# option 3. summarize from top --> creates a summary of the old messages and saves it at the top. mimics openai web app
# option 4. summarize from bottom --> summarizes content based on the most recent conversation and goes up till summary is done. Generally good fit for instructional bots where you want the bot to act more based on recent messages

# no context  --> all messages other than the system messages will be dumped 
# past messages and responses are not saved. Generally used along with JSON mode as a post-clean-up-function
def empty_context_window(thread, model):
    # this will keep going the system messages and dump the rest
    return [sys_msg for sys_msg in thread if (sys_msg['role'] == "system")]       

# sliding window: slides the thread and discards user/assistant messages from the top to fit the context window. 
# Generally good fit for group chats where the conversation content evolves too greatly for the initial messages have any weighting
# Good fit as pre-clean-up-func
# this is also assuming that all system messages are bundled up at the top
def slide_context_window(thread, model):
    # check to see if there is enough room for a large response. if there is just return what there is in the thread now
    if count_tokens_for_messages(thread, model) <= CONTEXT_WINDOW[model] - MESSAGE_TOKEN_LIMIT[model]:
        return thread
    # or else shave MESSAGE_TOKEN_LMIT worth of messages from the top

    # this is for retaining the system messages for later
    sys_messages = [sys_msg for sys_msg in thread if (sys_msg['role'] == "system")]
    token_count = 0
    for i in range(len(thread)):
        if thread[i]['role'] != "system": # count token only if it is 
            token_count += count_tokens_for_message(thread[i], model)
            if token_count >= MESSAGE_TOKEN_LIMIT[model]:
                # cut from current iterator and return
                break

    return sys_messages + thread[i:]
       
# temporarily disabling it since the implementation is wrong
# the summarization of the conversation would be represented as a user message
# def create_summerized_thread(chat_agent):
#     group = []
#     group_token_count = 0        
#     token_limit = CONTEXT_WINDOW[chat_agent.model] - MESSAGE_TOKEN_LIMIT[chat_agent.model]
        
#     for msg in chat_agent.thread:
#         msg_token_count = count_tokens_for_message(msg, chat_agent.model)
#         if msg_token_count + group_token_count >= token_limit:
#             # get a summarized message since it passed the limit
#             summary = create_summary_message(chat_agent, group)
#             group = [summary]
#             group_token_count += count_tokens_for_message(summary, chat_agent.model)
#         # keep adding the new message to group
#         group.append(msg)
#         group_token_count += msg_token_count

#     return group

# this is a private utility function aimed to summarize the conversation in a thread
# this function is used iteratively by run_thread compress the content in the thread
# def create_summary_message(chat_agent, messages):
#     SUMMARY_MESSAGE_INSTRUCTION = f"create a summary of this conversation in less than {MESSAGE_TOKEN_LIMIT[chat_agent.model]} tokens. Prefix the response with the word CURRENT CONTEXT:"
#     # existing system messages are not important for creating summary
#     messages = messages + [create_message("user", SUMMARY_MESSAGE_INSTRUCTION)]
#     # save the context as a user message
#     # TODO: check if it is better to save it as system message
#     return create_message("user", chat_agent._run_thread(messages).content)


# wrapper on top of python openai SDK/Driver which is used by both openai and anyscale
# this class manages a conversational thread with some built in error handling for exceeding context window and rate limiting
class ChatAgent:
    # model: the language model that would be used for this chat session. Make sure this is supported by the service provider. model value can be changed later.
    # api_key: the llm service provider's api_key e.g. API KEY or personal access token for anyscale.com of openai.com public endpoints. this cannot be none
    # organization: optional identifier for tenant of the api_key defined by the llm service provider.
    # base_url: the endpoint of the llm service. None value will default to openai.com (https://api.openai.com/v1)
    # instructions: optional instructions if you want the chat bot to function a certain way. This str or list[str]. Setting instructions value is specially important if you want the output format to be specific to your preference e.g. JSON blob
    # json_mode: If you want the output message text to be formatted as a json blob. Note that you would also include an instruction text saying so e.g. provide all outputs in JSON.
    # schema: if json_mode is set to True you can optionally specify the schema. This needs to be formatted as defined by https://json-schema.org/
    # BUG: anyscale endpoints cannot seem to support more than 1 system message. So instructions has to be 1 str.
    # pre_run_cleanup_func/post_run_cleanup_func: this is used for managing context window for stopping it from overflowing or cleaning up old messages for adding recency bisas
    # If no context window management is done, the thread will at somepoint overflow the context window and subsequent calls will fail
    def __init__(
            self, 
            model: str = os.getenv("OPENAI_CHAT_MODEL"),                       
            api_key: str = None, 
            organization: str = None, 
            base_url: str = None,
            instructions: str = None,
            # TODO: add context management scheme 
            # context_scheme = DEFAULT,
            json_mode: bool = False,
            schema = None,
            pre_run_cleanup_func = None,
            post_run_cleanup_func = None):
        
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
        # self.thread = []
        self.pre_run_cleanup = pre_run_cleanup_func
        self.post_run_cleanup = post_run_cleanup_func
        self._init_window()

    def __call__(self, message: str = None):
        if message != None:
            self.add_message(message)
        return self.get_response()

    # this function resets the context window.
    # this is a private utility function for the class itself
    # ext_thread is the initialization value
    def _init_window(self):
        self.thread = []
        if isinstance(self.instructions, str):        
            self.thread = [create_message("system", self.instructions)]
        elif isinstance(self.instructions, list):
            self.thread = [create_message("system", inst) for inst in self.instructions]       
        return self.thread
    
    # this is a private utility function
    def _run_thread(self, message_thread):
        # TODO: fix it for function call
        if self.response_format == None:
            resp = self.openai_client.chat.completions.create(
                model = self.model,
                messages = message_thread,
                temperature = self.temperature,
                response_format=self.response_format,
                seed = 10000 # a random number to keep the response consistent through out the context window
            )
        else:
            resp = self.openai_client.chat.completions.create(
                model = self.model,
                messages = message_thread,
                temperature = self.temperature,
                seed = 10000, # a random number to keep the response consistent
                response_format=self.response_format
            ) 
        
        return resp.choices[0].message
    
    # splits large messages and adds to the thread so that openai api doesnt die
    def add_message(self, user_message, name = None):      
        self.thread = self.thread + [create_message("user", chunk, name) for chunk in split_content(user_message, self.model)]
        return self.thread

    # passes the entire existing thread to the service for running.
    # this takes care of the context window if it gets bigger. 
    # it summarizes the existing content and creates 1 message for context
    def get_response(self):
        # checking if the current thread exceeds the context window
        run_cleanup = lambda cleanup_func: cleanup_func(self.thread, self.model) if cleanup_func else self.thread

        ic("Pre cleanup thread length: ", len(self.thread))  
        self.thread = run_cleanup(self.pre_run_cleanup)  
        ic("Pre call thread length: ", len(self.thread))   
        resp = self._run_thread(self.thread)        
        self.thread.append(create_message(resp.role, resp.content))
        ic("Post call thread length: ", len(self.thread))  
        self.thread = run_cleanup(self.post_run_cleanup)
        ic("Post cleanup thread length: ", len(self.thread))  

        return resp.content 

    
    