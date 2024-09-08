import streamlit as st
from utils import print_messages

st.set_page_config(page_title="ChatGPT", page_icon=".")
st.title("ChatGPT")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if user_input := st.chat_input("메시지를 입력해 주세요."):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(("user", user_input))
    with st.chat_message("assistant"):
        msg = f"당신이 입력한 내용 : {user_input}"
        st.write(msg)
        st.session_state["messages"].append(("assistant", msg))
