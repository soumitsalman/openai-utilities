from setuptools import setup, find_packages

setup(
    name='openai-utilities',
    version='0.1.0',
    description='Wrapper on top of openai python SDK/Driver primarily to interface with openai and anyscale public endpoints. This automatically takes care of rate limiting, context window limits and message size limits',
    url='https://github.com/soumitsalman/openai-utilities',
    author='Soumit Salman Rahman',
    author_email='soumitsr@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'openai',
        'tiktoken',
        'icecream'
    ],
    zip_safe=False
)
