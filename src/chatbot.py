from time import sleep

import streamlit as st

from llm import LLM

st.title("Social media chatter")

llm = LLM("mistral")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    stream = llm.generate_response(prompt)
    response = st.chat_message("assistant").write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
    llm.add_message(response)
