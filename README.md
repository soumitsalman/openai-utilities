## OPENAI UTILITIES
This is a privately maintained wrapper library on top of openai python Driver/SDK. Currently openai python Driver/SDK can be used to access both OpenAI and Anyscale public endpoints. This wrapper provides abstraction to access the models hosted by both of the endpoints. It currently uses the openai.OpenAI().chat.completions to make synchronous calls. It automatically handles rate limit error, context window resizing and large message splitting to fit within message token limit.

## Developer Info:
- name: Soumit Salman Rahman (personal)
- github: https://github.com/soumitsalman/
- email: soumitsr@gmail.com
- linkedIn: https://www.linkedin.com/in/soumitsrahman/

## Features:
- tokenutils.py: contains functions for counting tokens and splitting messages based on the models token size. 
- chat_connector.py: contains OpenAIChatSession class provides a chat session/thread management wrapper that deals with rate limit error, context window resizing and large message splitting under the hood
    - allows for JSON mode (set at the begining of the session initiation)
    - allows model change in the middle of a thread. Automatically splits the existing thread to adjust of reduced message token limit if the new model's limit is lower

### Missing Features:
- [ ] Function callback
- [ ] File upload for indexing
- [ ] Text-to-File upload for indexing

### Known Bugs:
anyscale.com endpoints cannot support more than 1 system message in a thread. However, openai.com endpoints are able to support multiple system messages.

## Usage Documentation:
Look at `examples/openai_example.py` for usage.

## Contribution Guideline:
Feel free to make contributions to the code. There is currently no test automation or github workflow set up. So #YOLO. If you want more wrapped capabilities feel free to reach out.

## Disclaimer:
This is privately maintained by the developer. The developer as no association to openai.org or anyscale.com on a commercial level.