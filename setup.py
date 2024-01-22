from setuptools import setup, find_packages


setup(
    name='openai-utilities',
    version='0.2.5',
    description='Wrapper on top of openai python SDK primarily to interface with openai and anyscale public endpoints.',
    long_description='This is a privately maintained wrapper library on top of openai python Driver/SDK. Currently openai python Driver/SDK can be used to access both OpenAI and Anyscale public endpoints. This wrapper provides abstraction to access the models hosted by both of the endpoints. It currently uses the openai.OpenAI().chat.completions to make synchronous calls. It automatically handles rate limit error, context window resizing and large message splitting to fit within message token limit. Look at https://github.com/soumitsalman/openai-utilities for more info.',
    long_description_content_type='text/markdown',
    keywords = 'openai anyscale tiktoken chatgpt chatbot',
    url='https://github.com/soumitsalman/openai-utilities',
    author='Soumit Salman Rahman',
    author_email='soumitsr@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'openai',
        'tiktoken',
        'icecream',
        'numpy',
        'transformers',
        'scipy'
    ],
    zip_safe=False
)
