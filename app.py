import streamlit as st
from langchain. chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from langchain.chains import LLMSummarizationCheckerChain
from langchain.memory import ConversationBufferMemory
import openai
from typing import Any, Dict, List

st.header("🤔ChatFACT")
st.subheader("get the summary of the text only with the facts")
                                  
def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

prompt = PromptTemplate(
    input_variables=["chat_history","input"], 
    template='Based on the following chat_history, Please reply to the question in format of markdown. history: {chat_history}. question: {input}'
)

user_input = st.text_area("You: ",placeholder = "一日1リットルの水素水でがんは確実に治ります。")
ask = st.button('ask',type='primary')
st.markdown("----")

class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

handler = SimpleStreamlitCallbackHandler()

if ask:
    res_box = st.empty()
    with st.spinner('typing...'):
        report = []
        chat = ChatOpenAI(streaming=True, temperature=0.9)
        conversation = LLMSummarizationCheckerChain.from_llm(
            llm=chat,
            max_checks=2,
            verbose=True
        )
        res = conversation.run(user_input) #, callbacks=[handler])
        st.markdown(res)
st.markdown("----")
