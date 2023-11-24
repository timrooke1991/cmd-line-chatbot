from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import (
    ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
)
from langchain.prompts import (
    HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
)

load_dotenv()

chat = ChatOpenAI(verbose=True)

memory = ConversationSummaryMemory(
    memory_key="messages", 
    return_messages=True,
    llm=chat
)

# This is a memory that stores the messages in a file
# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory("messages.json"),
#     memory_key="messages", 
#     return_messages=True
# )

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])
