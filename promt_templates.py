from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

promt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

formatted_prompt = promt_template.invoke({"topic": "soccer"})

print(formatted_prompt)


chat_message_template = ChatPromptTemplate(
    [
        ("system", "You are a helpfull assistant"),
        ("user", "Tell me about a joke about {topic}")
    ]
)

print(chat_message_template.invoke({"topic":"cat"}))


palce_holder = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])


messages_to_pass = [
    HumanMessage(content="What's the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="And what about Germany?")
]

place_holder_formatted = palce_holder.invoke({"msgs" : messages_to_pass})

print(place_holder_formatted)

llm = ChatOpenAI(temperature=0.5, model="gpt-4o")

response = llm.invoke(place_holder_formatted)

print(response.content)