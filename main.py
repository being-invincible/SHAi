import streamlit as st
from openai import OpenAI
from glob import glob
from llama_index import download_loader, VectorStoreIndex, ServiceContext
from llama_index.vector_stores import MilvusVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext
from llama_index.llms import PaLM
from llama_index.embeddings import GooglePaLMEmbedding
from llama_index.memory import ChatMemoryBuffer
import numpy as np
from trulens_eval import TruLlama, Tru, Query, Feedback, feedback
import google.generativeai as palm
import os
import pickle

# 1. Set up the name of the collection to be created.
COLLECTION_NAME = 'hydroponics_knowledge_base'

# 2. Set up the dimension of the embeddings.
DIMENSION = 1536

# 3. Set the inference parameters
BATCH_SIZE = 128
TOP_K = 3

# 4. Set up the connection parameters for your Zilliz Cloud cluster.
URI = st.secrets['CLUSTER_ENDPOINT']

TOKEN = st.secrets['API_TOKEN']

# Palm API
palm_api_key = st.secrets['PALM_API_KEY']

palm.configure(api_key=palm_api_key)

models = [
    m
    for m in palm.list_models()
    if "generateText" in m.supported_generation_methods
]
model = models[0].name
print(model)

llm = PaLM(api_key=palm_api_key)

# .streamlit/secrets.toml
# # OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_API_KEY"]

# To load the data from pickle
        
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Query Engine
# query_engine = index.as_query_engine()

# # Truera Wrapper
# l = TruLlama(query_engine)

st.title("SHAi")
st.text("Sustainable Hydroponic AI")

# client = TruLlama(query_engine)

if "openai_model" not in st.session_state:
    # st.session_state["openai_model"] = "gpt-3.5-turbo"
    pass

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Hydroponics & I will try to answer it in a sustainable way possible!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the vectors from Zilliz – hang tight! This should take 30 - 50 seconds"):
        # Grab all markdown files and convert them using the reader
        docs = []
        if os.path.exists("docs.pkl"):
            with open("docs.pkl", "rb") as f:
                docs = pickle.load(f)
        
        # Push all markdown files into Zilliz Cloud
        vector_store = MilvusVectorStore(
            uri=URI, 
            token=TOKEN, 
            collection_name=COLLECTION_NAME, 
            similarity_metric="L2",
            dim=DIMENSION,
        )
        
        llm=PaLM(api_key=palm_api_key)
        
        # Service Context - PALM - Vertex AI
        embed_model = GooglePaLMEmbedding("models/embedding-gecko-001", api_key=palm_api_key)
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

        index = VectorStoreIndex.from_documents(
            documents=docs, 
            service_context=service_context,
            show_progress=True,
        )

        return index

index = load_data()

# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
chat_engine = index.as_chat_engine(chat_mode="context",
    memory=memory, verbose=True)

# Initialize Huggingface-based feedback function collection class:
hugs = feedback.Huggingface()
openai = feedback.OpenAI()
# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on_input_output()
# By default this will check language match on the main app input and main app
# output.

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
).aggregate(np.min)

feedbacks = [f_lang_match, f_qa_relevance, f_qs_relevance]

chat_engine = TruLlama(chat_engine, feedbacks=feedbacks, app_id="SHAi App")

tru = Tru()
tru.run_dashboard()

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history