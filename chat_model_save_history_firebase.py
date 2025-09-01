from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI
#from langchain.schema import AIMessage,HumanMessage,SystemMessage

load_dotenv()


PROJECT_NAME = "langchain-chat-model-history"
PROJECT_ID = "langchain-chat-model-history"
PROJECT_NUMBER = "404011607590"
SESSION_ID = "user1_session"
COLLECTION_NAME = "chat_history"

print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

model = ChatOpenAI(temperature=0.5, model="gpt-4o")

while True:
    human_input = input("You: ")
    if human_input.lower() == "exit":
        break
    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)

    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")