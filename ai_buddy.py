# Streamlit → builds the web app.
import streamlit as st

# streamlit_chat → helper for chat-like UI.
from streamlit_chat import message

# ChatOpenAI → LLM wrapper (here configured for TogetherAI’s API).
from langchain_google_genai import ChatGoogleGenerativeAI

# ConversationChain → wraps an LLM into a chain that maintains dialogue.
from langchain.chains import LLMChain

# ConversationBufferWindowMemory → remembers a rolling window of messages (here, last k=3 turns).
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# This helps us build structured prompts with system & user role
from langchain.prompts import ChatPromptTemplate

# Import Json to be able to store the past history in the Json File
import json

# os → for environment variables (API keys, etc.).
import os

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# File to which we will store the history
HISTORY_FILE = "chat_history.json"

# method that saves the history to the file
def save_history(messages, filename=HISTORY_FILE):
    """Save chat history (list of dicts) to JSON file."""
    with open(filename, "w") as f:
        json.dump(messages, f, indent=4)

# Method to read from the history file and load it back to the chatbot
def load_history(filename=HISTORY_FILE):
    """Load chat history from JSON file if it exists."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None

def load_history(filename=HISTORY_FILE):
    """Load chat history from JSON file if it exists, safely."""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []  # return empty history if file is corrupted
    return []


# Create user interface
st.title("🗣️ AI Buddy - Ask me anything - I am a Multilingual Chat Machine⚡ ")
st.subheader("㈻ Simple Chat Interface for LLMs (MasterMind: Build Fast with AI)")

# Languages Supported by Gemini Model
SUPPORTED_LANGUAGES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bn": "Bengali",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "iw": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "ms": "Malay",
    "ml": "Malayalam",
    "mr": "Marathi",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pa": "Punjabi",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
}

# Select the language of choice
language_code = st.selectbox(
    "🌐 Choose your conversation language:",
    options=list(SUPPORTED_LANGUAGES.keys()),
    format_func=lambda code: SUPPORTED_LANGUAGES[code],
    index=list(SUPPORTED_LANGUAGES.keys()).index("en"),  # default English
)

#---------------------------------------------------------------------------------------------
# st.session_state is a dict-like object Streamlit uses to persist values across interactions.
#---------------------------------------------------------------------------------------------
# buffer_memory: stores recent conversation history (upto last 3 exchanges).
# Initialize session state variables
if 'buffer_memory' not in st.session_state.keys():
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# messages: keeps the full chat log for display in the UI, starting with a greeting from the assistant.
if "messages" not in st.session_state.keys():
    # Try to load from disk
    stored_history = load_history()
    if stored_history:
        st.session_state.messages = stored_history
    else:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you today?"}
        ]

#---------------------------------------------------
# Initialize ChatOpenAI LLM - We are not using now |
#---------------------------------------------------
# llm = ChatOpenAI(model_name="gpt-4o-mini")

#---------------------------------------------------
# Initialize Llama LLM - We are not using now      |
#---------------------------------------------------
# llm = ChatOpenAI(model = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
#                      openai_api_key = st.secrets["TOGETHER_API_KEY"] , ## use your key
#                      openai_api_base = "https://api.together.xyz/v1"
#
# )

#------------------------------------------------------
# Initialize Google Gemini LLM - We are not using now |
#------------------------------------------------------
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash",
                             google_api_key=st.secrets["GOOGLE_API_KEY"])
# (We are saving the GOOGLE_API_KEY in a streamlit settings)

#--------------------------------------------------------------------------------------------------------------------
# Dynamic system prompt with language choice
#--------------------------------------------------------------------------------------------------------------------
system_template = f"""
You are a helpful AI assistant. 
The user may type in any language, but you must always reply in {SUPPORTED_LANGUAGES[language_code]}.
If the user writes in English or another language, translate their question into {SUPPORTED_LANGUAGES[language_code]} 
before answering, and then respond in {SUPPORTED_LANGUAGES[language_code]}.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{input}")
])


#---------------------------------------------------
# Initialize Conversation Chain                    |
#---------------------------------------------------
# ConversationChain wraps the LLM and automatically integrates with the memory object to maintain context.
conversation = LLMChain(memory=st.session_state.buffer_memory, llm=llm, prompt=prompt)

#----------------------------------------------------------------------------------------------
# st.chat_input() displays a chat-style text input box.
# If the user types a question, it’s appended to st.session_state.messages with "role": "user".
#----------------------------------------------------------------------------------------------
if prompt_text := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    save_history(st.session_state.messages)  
    
#----------------------------------------------------------------------------------------------
# Iterates over stored messages and displays them in chat bubbles.
# Uses st.chat_message(role) for correct styling ("user" vs "assistant").
#----------------------------------------------------------------------------------------------
for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

#----------------------------------------------------------------------------------------------
# Following Section:
# ------------------
# Checks if the last message was from the user.
# If so:
# Displays an "assistant" bubble with a spinner (Thinking...).
# Calls the LLM (conversation.predict) to generate a reply, using:
# The prompt (latest user input).
# The buffer_memory (last 3 exchanges).
# Displays the response.
# Saves it back into st.session_state.messages so it shows up in history.
#----------------------------------------------------------------------------------------------

# ---------------------------
# Generate Assistant Response
# ---------------------------
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input=prompt_text)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.messages.append(message)
            save_history(st.session_state.messages)

#----------------------------------------------------------------------------------------------
# ⚡ So in summary:
# st.session_state.messages → drives the chat UI.
#
# ConversationBufferWindowMemory → controls how much past context the model sees.
#
# ConversationChain → wires the memory + LLM together for conversation.
#----------------------------------------------------------------------------------------------

# ----------------
# Important Update
# ----------------
# On startup → load chat history from chat_history.json (if it exists).
# Every time user/assistant adds a message → save updated history to file.
# Even if you restart the Streamlit app, the conversation continues where it left off.
