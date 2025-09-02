from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence,RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(temperature=0.5, model="gpt-4o")

positive_feadback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpfull assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feadback}")
    ]
)

negative_feadback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpfull assistant."),
        ("human", "Generate a thank you note for this negative feedback: {feedback}")
    ]
)

neutral_feadback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpfull assistant."),
        ("human", "Generate a thank you note for this neutral feedback: {feedback}")
    ]
)

esclatation_feadback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpfull assistant."),
        ("human", "Generate a message to  escalate to huamn agent note for this feedback: {feedback}")
    ]
)

classification_template = ChatPromptTemplate(
    [
        ("system", "You are a helpfull assistant."),
        ("human", "Classify this sentiment of this feedback as positive,negative,netural or escalate: {feedback}")
    ]
)

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feadback_template | llm | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feadback_template | llm | StrOutputParser()
    ),
    (
        lambda x: "netural" in x,
        neutral_feadback_template | llm | StrOutputParser()
    ),
    esclatation_feadback_template | llm | StrOutputParser()
)

classification_chain = classification_template | llm | StrOutputParser()

chain = classification_chain | branches


review = "I'm not sure about this product, can you tell me more about it's feature?"
result = chain.invoke({"feedback": review})

print(result)