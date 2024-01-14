import config
from openai_connectors.chat_connector import OpenAIChatSession, OpenAIModels, create_message
from openai_connectors.tokenutils import count_tokens, count_tokens_for_messages, split_content
from icecream import ic

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
    ic(count_tokens(contents[0], config.get_ll_model()))    
    # count tokens for multiple messages with role information in it
    ic(count_tokens_for_messages([create_message("user", msg) for msg in contents], config.get_ll_model()))

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
    ic(len(split_content(large_content, config.get_ll_model())))

# testing chat session
def example_chat_session():    
    messages = [f"generate python code for printing {i} to {i+10}." for i in range(0, 100, 10)]

    session = OpenAIChatSession(
        model = config.get_ll_model(), 
        instruction="you are a verbose programming helper", 
        service_api_key=config.get_llm_service_api_key(),
        service_url=config.get_llm_service_base_url() # this is primarily for anyscale apis
    )

    for msg in messages:
        # adds user message with optional name parameter for multi user chat
        session.add_to_thread(msg, name = "random_name")
        # gets the response and prints
        ic(session.run_thread())


if __name__ == "__main__":
    example_count_tokens()
    example_split_content()
    example_chat_session()