import openai
from . import config
from openai_utilities.chat import ChatAgent, create_message, slide_context_window, empty_context_window
from openai_utilities.embeddings import EmbeddingAgent
from openai_utilities.retryutils import retry_after_random_wait
from openai_utilities.tokenutils import count_tokens, count_tokens_for_messages, split_content
from icecream import ic
from functools import reduce
import json

# testing tokencounts
def example_count_tokens():
    contents = [
        "que pasa calabaza",
        "how are you feeling today",
        "tell me something about bugs bunny",
        "who created it",
        "what how does it relate to tom and jerry",
        "what how does it relate to road runner"
    ]

    # count token for 1 string
    ic(count_tokens(contents[0], config.get_chat_model()))    
    # count tokens for multiple messages with role information in it
    ic(count_tokens_for_messages([create_message("user", msg) for msg in contents], config.get_chat_model()))

# testing splitting of 1 large message into multiple smaller messages
def example_split_content():
    msg_seed = """
Once upon a time, in a quaint village nestled between rolling hills and a sparkling river, there lived a young and curious girl named Amelia. Amelia was known throughout the village for her insatiable thirst for knowledge and her adventurous spirit. She spent her days exploring the forest, studying the stars, and talking to the animals that roamed the woods. But there was one thing that fascinated her above all else: a mysterious old oak tree that stood at the edge of the forest.
This oak tree was unlike any other tree in the village. Its bark was gnarled and ancient, and its branches stretched high into the sky, as if they were trying to reach the heavens themselves. The villagers often whispered about the tree, saying that it held a secret, a hidden treasure that would bring unimaginable wealth and power to anyone who could unlock its mysteries.
Amelia, being the curious soul that she was, couldn't resist the allure of the old oak tree. She would often sneak away from her chores and spend hours sitting beneath its branches, running her fingers over the rough bark and listening to the whispers of the wind that rustled through its leaves. She knew that the tree held a secret, and she was determined to uncover it.
One warm summer's day, as Amelia sat beneath the oak tree, she noticed something unusual. A small, glowing keyhole had appeared on the trunk of the tree, right at eye level. It was as if the tree had chosen her to reveal its secret. With trembling hands, Amelia reached into her pocket and pulled out a tiny key that had been passed down through her family for generations. It was said to be a key to a long-forgotten treasure.
Amelia carefully inserted the key into the keyhole, and to her amazement, the tree began to shudder and shake. Its branches quivered, and the ground beneath her feet trembled. Then, with a deafening crack, the tree split open, revealing a hidden passage leading deep underground.
Without hesitation, Amelia descended into the darkness, guided only by the soft glow of her key. She found herself in a vast underground chamber filled with glittering jewels, piles of gold, and ancient artifacts beyond imagination. It was the legendary treasure of the old oak tree.
Amelia couldn't believe her eyes. She had uncovered the village's most coveted secret, but she knew that the treasure wasn't meant for her alone. With a generous heart, she shared her discovery with the entire village, and they all rejoiced in their newfound wealth.
But Amelia's true treasure wasn't the gold or jewels; it was the knowledge that had led her to the old oak tree in the first place. She had learned that sometimes, the greatest treasures in life are not material possessions, but the mysteries waiting to be unraveled, the adventures waiting to be undertaken, and the love and generosity that bind us all together.
And so, the village prospered not just from their newfound riches, but from the lessons learned from the curious girl who had uncovered the secret of the old oak tree. Amelia's spirit of adventure and thirst for knowledge inspired generations to come, reminding them that there is magic in the world for those who dare to seek it.
And they all lived happily ever after, in a village where curiosity was celebrated, and the old oak tree continued to whisper its secrets to those who were willing to listen."""
    large_content = " ".join([msg_seed] * 20)
    ic(len(split_content(large_content, config.get_chat_model())))

# testing chat session
# the retry decorator here is for dealing with rate limit. This number should change with different services
@retry_after_random_wait(min_wait=121, max_wait=600, retry_count=5, errors=(openai.RateLimitError))
def example_chat_session(): 
    messages = [f"generate python code for printing {i} to {i+10}." for i in range(0, 100, 10)]

    session = ChatAgent(
        model = config.get_chat_model(), 
        instructions="you are a verbose programming helper", 
        api_key=config.get_api_key(),
        base_url=config.get_base_url(), # this is primarily for anyscale apis
        pre_run_cleanup_func=slide_context_window
    )

    for msg in messages:
        # adds user message with optional name parameter for multi user chat
        session.add_message(msg, name = "random_name")
        # gets the response and prints
        ic(session.get_response())

@retry_after_random_wait(min_wait=121, max_wait=600, retry_count=5, errors=(openai.RateLimitError))
def example_chat_session_json_mode(): 
    session = ChatAgent(
        # Currently JSON mode is supported by the following models
        # anyscale.com --> "mistralai/Mistral-7B-Instruct-v0.1"
        # anyscale.com --> "mistralai/Mixtral-8x7B-Instruct-v0.1"
        # openai.com --> gpt-4-1106-preview
        # openai.com --> gpt-3.5-turbo-1106
        model = config.get_chat_model(),          
        api_key=config.get_api_key(),
        base_url=config.get_base_url(), # this is primarily for anyscale apis
        instructions="You read chatlogs between multiple people create responses based on user prompt. You provide all your output in JSON format",
        json_mode=True, # schema is optional
        post_run_cleanup_func=empty_context_window
    )

    messages = [
        "Determine how many people are in the chat, a summary of what they are talking about and who talks the most. Provide the output in a JSON format",
        """Johnnie Walker: Hey everyone, how's it going?
        Jim Bean: Hey Johnnie! Not too shabby, just another day at the office. How about you?
        Gin Aviator: Hey there, folks! I'm fantastic, as usual, and I've got so much to share!
        Gin Aviator: You won't believe the hilarious meme I saw today. It had me laughing for ages!
        Gin Aviator: Also, I tried this new recipe last night, and it was to die for. It's all about that culinary adventure, you know?
        Jim Bean: That sounds great, Gin! I'm down for an action movie.
        Jose Cuervo: ¡Hola a todos! Estoy muy bien, disfrutando del fin de semana.
        Jose Cuervo: Claro, Jim. Este fin de semana planeo relajarme y ver una buena película.
        Jose Cuervo: Sería genial si te unieras, Johnnie.
        Jim Bean: So, Johnnie, have any plans for the weekend?
        Gin Aviator: Oh, speaking of the weekend, I heard about this awesome art exhibition happening downtown!
        Gin Aviator: We should totally check it out after the movie. I'm all about embracing culture and creativity.
        Gin Aviator: And don't get me started on the latest tech gadgets I've been eyeing. It's like Christmas came early!
        Jim Bean: That sounds exciting, Gin! I'm down for an action movie.
        Jose Cuervo: Yo también, una película de acción suena emocionante.
        Gin Aviator: Awesome! I'll look up the movie options and showtimes. Johnnie, are you up for some action?
        Gin Aviator: By the way, speaking of movies, did you guys see that new sci-fi trailer? Mind-blowing stuff, I tell you!
        Gin Aviator: So, who's up for some popcorn and cinematic adventures?
        Johnnie Walker: Late afternoon works for me.
        Jim Bean: Yeah, Johnnie, you should join us. It'll be a blast!
        Jose Cuervo: Estoy disponible en la tarde, cualquier hora está bien para mí.
        Gin Aviator: Great! I'll look for action movie showtimes in the late afternoon. And after the movie, we'll head to that art exhibition, alright?
        Gin Aviator: And Johnnie, don't worry, we'll make sure you have a fantastic time too!"""
    ]
    # add the contents in the thread
    for msg in messages:
        session.add_message(msg)
    
    # run the thread
    ic(json.loads(session.get_response()))

# disabling this since the update-model function has been turned off
# @retry_after_random_wait(min_wait=61, max_wait=240, retry_count=5, errors=(openai.RateLimitError))
# def example_update_model():
#     messages = [f"generate python code for printing {i} to {i+10}." for i in range(0, 100, 10)]

#     session = ChatAgent(
#         model = "mistralai/Mistral-7B-Instruct-v0.1", 
#         instructions="you are a verbose programming helper", 
#         api_key=config.get_api_key(),
#         base_url=config.get_base_url() # this is primarily for anyscale apis
#     )

#     ic(session.model)
#     for msg in messages[:len(messages)//2]:
#         # adds user message with optional name parameter for multi user chat
#         session.add_to_thread(msg)
#         # gets the response and prints
#         ic(session.run_thread())

#     # change modle
#     session.update_model("HuggingFaceH4/zephyr-7b-beta")

#     ic(session.model)
#     for msg in messages[len(messages)//2:]:
#         # adds user message with optional name parameter for multi user chat
#         session.add_to_thread(msg, name = "random_name")
#         # gets the response and prints
#         ic(session.run_thread())

@retry_after_random_wait(min_wait=121, max_wait=600, retry_count=5, errors=(openai.RateLimitError))
def example_embeddings_small_text():
    # EMBEDDING small text with no metadata
    embed = EmbeddingAgent(
        model = config.get_embeddings_model(), 
        api_key=config.get_api_key(), 
        base_url=config.get_base_url())    
    vectors = embed.create("how dem apples?")
    ic(len(vectors), vectors[0], vectors[-1])

@retry_after_random_wait(min_wait=121, max_wait=600, retry_count=5, errors=(openai.RateLimitError))
def example_embeddings_large_text():
    embed = EmbeddingAgent(
        model = config.get_embeddings_model(), 
        api_key=config.get_api_key(), 
        base_url=config.get_base_url())
    
    content = {
        "title": "Back-End & Web Development Trends For 2024",
        "description": "The ever-shifting landscape of digital innovation can feel like a relentless race, a whirlwind of challenges and opportunities. Your pains as a developer are real \u2014 the pressure to deliver\u2026",
        "authors": "Kostya Stepanov",
        "url": "https://medium.com/ux-planet/back-end-web-development-trends-for-2024-04cc14bb43cb",
        "text": "Kostya Stepanov\n\n\u00b7\n\nFollow\n\nPublished in\n\nUX Planet\n\n\u00b7\n\n9 min read\n\n\u00b7\n\nOct 17, 2023\n\n--\n\nBy Mary Moore, copywriter at Shakuro\n\nThe ever-shifting landscape of digital innovation can feel like a relentless race, a whirlwind of challenges and opportunities. Your pains as a developer are real \u2014 the pressure to deliver cutting-edge products, stay competitive, and keep up with evolving user expectations can be overwhelming.\n\nBut what if we told you that there\u2019s a compass to navigate this complex terrain? What if there were insights that could not only ease your pains but also spark a wildfire of inspiration? Well, you\u2019re in luck because we\u2019re about to embark on a journey through the future trends of back-end and web development.\n\nIn this article, we will unveil the key trends that will define the year 2024, providing you with the tools and knowledge to stay ahead of the curve. Whether you\u2019re a seasoned developer striving for excellence or a product owner seeking to drive innovation, this is your roadmap to success.\n\nAI and machine learning integration\n\nArtificial Intelligence and machine learning are no longer buzzwords but powerful tools in the arsenal of developers. In back-end development, they play a pivotal role in automating tasks, analyzing vast datasets, and making data-driven decisions. Here\u2019s how you can use them to your advantage:\n\nCode generation: you can generate code snippets or even complete chunks, saving time and reducing the chances of human error. Check out tools like OpenAI\u2019s ChatGPT: they write code based on natural language descriptions.\n\nSecurity and code quality improvement: use AI-based code review tools to analyze codebases and identify potential bugs, security vulnerabilities, and quality issues. For example, DeepCode and CodeClimate help developers write more secure code.\n\nPersonalization: with Artificial Intelligence, you can analyze user behavior and preferences to deliver tailored content and product recommendations. This way, your web and mobile apps have higher user engagement and retention rates.\n\nPredictive analytics: with machine learning models, you forecast user actions. So you can create preventive measures to solve issues that may emerge.\n\nRecommendation engines: AI-driven recommendation systems suggest products, services, or content to clients based on their preferences and behavior. Use this trend to enhance user engagement and conversion rates.\n\nChatbots and virtual assistants: to level up your customer service, integrate AI-powered chatbots into your app or website. They can handle customer inquiries, providing instant support 24/7.\n\nServerless architecture\n\nServerless architecture is a trend in web development that will continue to expand in 2024. Often referred to as Function as a Service (FaaS), it eliminates the need for developers to manage servers. Instead, you can focus on writing code and deploying functions, enhancing scalability and cost-efficiency.\n\nThe serverless approach allows programs to operate on cloud-based servers. So you don\u2019t need to be concerned with server availability, capacity, or infrastructure management. AWS, Microsoft Azure Functions, Google Cloud Functions, and others offer such services. Furthermore, it is very cost-efficient, since the service cost is usually calculated depending on real resource utilization.\n\nYou can apply this development trend in most businesses for image identification, multimedia processing, chatbots and assistants, notification engines, IoT apps, data collecting, and so on.\n\nEdge computing\n\nThis emerging technology decentralizes data processing by moving it closer to the source. In web development, you can minimize latency and enhance real-time capabilities.\n\nReduced latency: since edge computing brings computation closer to the data source, it reduces the round-trip time between a user\u2019s request and the response. This significantly lowers latency, making web applications more responsive and improving the user experience. For real-time applications like online gaming, video streaming, and IoT interactions, lower latency is crucial.\n\nImproved performance: also, with this 2024 trend, your web applications have faster performance. Content delivery networks (CDNs) are a common implementation of edge computing that caches and serves content from edge servers. The approach decreases the load on the back-end servers and accelerates content delivery.\n\nBandwidth savings: less data is transferred to centralized points or cloud services. So you have significant bandwidth savings, especially in scenarios where large volumes of data are generated.\n\nReal-time data processing: edge nodes can process data in real time, making it ideal for applications that require immediate analysis and decision-making. For example, in IoT apps: sensors can process data at the edge to trigger actions or alerts without relying on centralized servers.\n\nZero Trust Architecture (ZTA)\n\nIt is a trendy cybersecurity approach that challenges the traditional perimeter-based security model. In a zero-trust model, organizations do not automatically trust any user or device, whether they are inside or outside the corporate network. Instead, it assumes that threats can come from both internal and external sources, and it verifies and validates every user and device attempting to access resources.\n\nHere are the key principles of this software development trend:\n\nVerify identity: people must authenticate their identity before gaining access to resources. This often involves multi-factor authentication (MFA) and strong verification methods.\n\nLeast privilege access: users get the least privilege necessary to perform their tasks. Access is restricted to only essential things, reducing the potential impact of a security breach.\n\nMicro-segmentation: the trend suggests segmenting the network at a granular level, allowing you to isolate and protect individual resources.\n\nData encryption: the encryption applies both in transit and at rest to protect the data from unauthorized access.\n\nNo implicit trust: apply the principle of \u201cnever trust, always verify\u201d, meaning that verification is required at every stage of access.\n\nInternet of Things\n\nThe Internet of Things is a rapidly growing software development trend. This is an interconnected network of physical devices that collect and exchange data over the Internet. These devices can range from simple sensors and actuators to complex industrial machinery and consumer appliances. Smart homes, robot vacuums, lightning, and conditioning \u2014 all of these features are a part of IoT.\n\nThere are approximately 15.14 billion connected IoT devices. They generate vast amounts of data, including device statuses and user interactions. You can create systems for your web or mobile app to ingest, process, and store this data efficiently.\n\nThis trend pairs well with cloud computing since the data is typically stored and processed in the cloud. You need to work with platforms like AWS, Azure, or Google Cloud to build scalable and reliable back-end systems for IoT applications.\n\nErgonomic keyboards\n\nWhile not directly related to back-end or web development trends, ergonomic keyboards are gaining attention among developers. They often spend long hours typing and coding, which can lead to discomfort and health issues if not properly managed. Ergonomic keyboards are designed with the user\u2019s comfort and health in mind.\n\nWith reduced discomfort and a more comfortable typing experience, your productivity increases. You also have reduced downtime due to discomfort-related breaks.\n\nPopular programming languages in 2024\n\nRust\n\nRust is gaining momentum as a robust and secure programming language. Its memory safety features make it ideal for systems in backend development that prioritize performance and security.\n\nMemory safety: Rust uses a strict ownership model and a borrow checker to prevent common memory-related bugs like null pointer dereferences and data races.\n\nConcurrency: there is built-in support for concurrency with its ownership and borrowing system, allowing you to write concurrent code without the risk of data races. This is important for building scalable and efficient web and back-end applications.\n\nWebAssembly support: Rust is gaining traction as a language for compiling to WebAssembly, where you can run code in web browsers at near-native speeds.\n\nJavaScript\n\nJavaScript has been a trend in software development for quite a long time. It continues to play a crucial role in the industry.\n\nWhile JavaScript was traditionally a front-end language, it has expanded its reach into back-end development as well. Node.js, a runtime environment for executing JavaScript server-side, has gained significant popularity. It allows you to use JavaScript on both the client and server sides of a web application, making it a full-stack language.\n\nAt the same time, JavaScript has a vast ecosystem of libraries, frameworks, and tools that simplify web development. For back-end development with Node.js, you can take advantage of frameworks like Express.js and NestJS. JavaScript is often used in serverless computing platforms like AWS Lambda, Azure Functions, and Google Cloud Functions.\n\nPython\n\nPython has long been a popular programming language for back-end software development. Python\u2019s clean and easy-to-read syntax makes it an excellent choice for developers, whether they are beginners or experienced programmers. This simplicity accelerates development and reduces the likelihood of errors. There is a rich ecosystem of libraries and packages that simplify web and back-end development.\n\nPython works well with another trend of 2024 \u2014 cloud platforms like AWS, Azure, and Google Cloud. For example, you can use libraries such as NumPy and Pandas for data-driven web applications in IoT, machine learning, and AI systems.\n\nPopular frameworks in 2024\n\nDjango\n\nDjango has been a trend in web development for a while. Its main goal is to make the development process faster and more efficient by providing a robust and flexible foundation.\n\nOne of Django\u2019s major strengths is its emphasis on rapid development. It follows the \u201cDon\u2019t Repeat Yourself\u201d (DRY) principle and provides a high-level, clean, and pragmatic design that allows you to build feature-rich web applications with less code and effort.\n\nMoreover, the framework includes an admin interface that is automatically generated depending on the data models defined in your application. With the interface, you manage your app\u2019s data easily, making it a valuable tool during development and for site administrators.\n\nNode.js\n\nIt is an open-source, cross-platform JavaScript runtime environment with which you can run JavaScript code on the server side. Node.js has gained significant popularity and has become a development trend in 2024.\n\nIt is known for its event-driven, non-blocking I/O model. It can handle a large number of concurrent connections efficiently, making it well-suited for building scalable and high-performance applications.\n\nAlso, Node.js allows you to use JavaScript not only for client-side web development but also for server-side programming. This unification of client-side and server-side code simplifies the whole process, as you re-use the same language and libraries on both ends.\n\nSvelte\n\nSvelte is a game-changer in web development. It compiles components into highly efficient JavaScript, resulting in faster load times and a smoother user experience.\n\nSvelte is often compared to React, another popular JavaScript framework. While React focuses on a virtual DOM, this one takes a different approach by compiling components into efficient JavaScript code during build, potentially leading to better performance.\n\nBy the way, Svelte has recently become the most admired JavaScript web framework in the StackOverflow industry survey.\n\nQwick\n\nIt is an open-source project featuring a modern JavaScript framework. Qwick optimizes web application performance, particularly focusing on speed and efficiency.\n\nThe framework is becoming a trend for its rapid page load times and efficient rendering approach, even for complex websites. Unlike traditional frameworks that require client-side hydration for interactivity, Qwik eliminates this step, further improving load times.\n\nUse the trends to your advantage\n\nThe world of back-end and web development is poised for exciting changes in 2024. From AI and ML integration to serverless architecture and edge computing, you have a lot to look forward to. Check out these trends and technologies now to harness their full potential and remain competitive in the ever-evolving digital landscape.\n\nOriginally published at https://shakuro.com"
    }
    add_metadata = lambda text: f"Title: {content['title']}\nAuthor: {content['authors']}\nContent: {text}"
    
    # chunk the text into smaller pieces before sending it for embeddings
    chunks = embed.chunk_text(text = content['text'], metadata_func=add_metadata)    
    vectors = [embed.create(text = chunk) for chunk in chunks]
    ic(len(chunks), len(vectors),len(vectors[0]), vectors[0][0], vectors[0][-1])

@retry_after_random_wait(min_wait=121, max_wait=600, retry_count=5, errors=(openai.RateLimitError))
def example_search_embeddings():
    items_to_search_in = _read_file("./examples/_embeddings_examples.json")
    ic(len(items_to_search_in))
    embedder = EmbeddingAgent()

    queries = [
        "what is the lastest update on corona virus", 
        "what do i need to do get promoted as poduct manager", 
        "how to attracting tech talent", 
        "PRAW is not working. what should I do" ]

    for q in queries:
        ic(f"====== QUERY: {q} ========\n")
        results = embedder.search(q, items_to_search_in, lambda item: item['vectors'], limit = 3)
        ic(">>> RESULT: >>>")
        [ic(res['text'][200: ]) for res in results]

def _read_file(filepath):
    with open(filepath, "r") as file:
        items = json.load(file)
    return items