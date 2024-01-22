## OPENAI UTILITIES
This is a privately maintained wrapper library on top of openai python Driver/SDK. Currently openai python Driver/SDK can be used to access both OpenAI and Anyscale public endpoints. This wrapper provides abstraction to access the models hosted by both of the endpoints. It currently uses the openai.OpenAI().chat.completions to make synchronous calls. It automatically handles rate limit error, context window resizing and large message splitting to fit within message token limit.

## Developer Info:
- name: Soumit Salman Rahman (personal)
- github: https://github.com/soumitsalman/
- email: soumitsr@gmail.com
- linkedIn: https://www.linkedin.com/in/soumitsrahman/

## Features:
- retryutils.py: contains functions that can be used as decorators for to deal with rate limiting errors
- tokenutils.py: contains functions for counting tokens and splitting messages based on the models token size. 
- chat.py: contains OpenAIChatSession class provides a chat session/thread management wrapper that deals with rate limit error, context window resizing and large message splitting under the hood
    - allows for JSON mode (set at the begining of the session initiation)
    - every chat session has ability to customize context window management
- embeddings.py: wrapper for client.embeddings.create function call. It has function to chunk large text into smaller pieces, create embeddings and vectors search

### Missing Features:
- [ ] Function callback

### Known Bugs:
anyscale.com endpoints cannot support more than 1 system message in a thread. However, openai.com endpoints are able to support multiple system messages.

## Usage Documentation:
Look at `examples/openai_example.py` for usage.

## Contribution Guideline:
Feel free to make contributions to the code. There is currently no test automation or github workflow set up. So #YOLO. If you want more wrapped capabilities feel free to reach out.

## Disclaimer:
This is privately maintained by the developer. The developer as no association to openai.org or anyscale.com on a commercial level.