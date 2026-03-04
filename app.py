import streamlit as st
from Main import chatbot_response, load_database

# Page config
st.set_page_config(page_title="AI Health Assistant", layout="centered")

st.title("🩺 AI-Powered Smart Health Assistant")
st.write("Describe your symptoms and get possible health insights.")

# Load database
database = load_database()

# Create symptom list
symptom_list = []
for disease in database:
    symptom_list.extend(database[disease]["symptoms"])
symptom_list = list(set(symptom_list))

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your symptoms here...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    response = chatbot_response(user_input, database, symptom_list)

    # Show bot message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)