import json
import random
import re
import streamlit as st
from typing import Generator
from groq import Groq

# Load the JSON data
with open(r'C:\Users\USER\Desktop\Chatbot Project\data.json', 'r') as f:
    data = json.load(f)

# Initialize the Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Set Streamlit page configuration
st.set_page_config(page_icon="💬", layout="wide", page_title="EASTC Chatbot")

# Define the icon function
def icon(emoji: str):
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

# Display the icon
icon("🎓")

# Subheader
st.subheader("EASTC Chatbot")

# Initialize chat history and set a welcome message if it's the first time
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to the EASTC Chatbot! How can I assist you today?"}]

# Set default model and max tokens
model_option = "mixtral-8x7b-32768"
max_tokens = 32768

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍🎓'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Function to generate chat responses
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Function to search and retrieve data from the JSON knowledge base
def search_knowledge_base(prompt):
    for intent in data['intents']:
        for pattern in intent['patterns']:
            if re.fullmatch(pattern, prompt, re.IGNORECASE):
                response = random.choice(intent['responses'])
                if isinstance(response, dict) and response.get('type') == 'list':
                    return "\n".join([f"- {item}" for item in response['content']])
                return response

    for intent in data['intents']:
        for pattern in intent['patterns']:
            if re.search(pattern, prompt, re.IGNORECASE):
                response = random.choice(intent['responses'])
                if isinstance(response, dict) and response.get('type') == 'list':
                    return "\n".join([f"- {item}" for item in response['content']])
                return response

    return None

# Function to check if the response is related to EASTC
def is_response_related_to_eastc(response):
    keywords = ["Eastern Africa Statistical Training Center", "EASTC"]
    return any(keyword in response for keyword in keywords)

# Function to create a custom prompt
def create_custom_prompt(user_input: str) -> str:
    base_prompt = (
        "I want you to act as an admission officer at the Eastern Africa Statistical Training Center (EASTC). "
        "Your task is to help the user interact with our chatbot to have a smooth experience and get good information relating to our university. "
        "Please provide detailed, relevant, and helpful responses. Here is the user's input:"
    )
    return f"{base_prompt} {user_input}"

# Main chat input handling
if prompt := st.chat_input("Enter your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍🎓'):
        st.markdown(prompt)

    # Fetch response from the knowledge base
    response = search_knowledge_base(prompt)

    if response is None:
        try:
            custom_prompt = create_custom_prompt(prompt)
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ] + [{"role": "system", "content": custom_prompt}],
                max_tokens=max_tokens,
                stream=True
            )

            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = "".join(chat_responses_generator)

                # Ensure the response is related to EASTC
                if is_response_related_to_eastc(full_response):
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    fallback_response = "I apologize, but I couldn't find any specific information related to your question. Please visit the EASTC website or contact their admissions office directly for more information."
                    st.markdown(fallback_response)
                    st.session_state.messages.append({"role": "assistant", "content": fallback_response})

        except Exception as e:
            st.error(f"Error: {e}", icon="🚨")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
