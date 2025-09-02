from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(temperature=0.5, model="gpt-4o")


chat_message_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {count} jokes.")
    ]
)


chain = chat_message_template | llm | StrOutputParser()

result = chain.invoke({"topic":"soccer", "count":3})

print(result)