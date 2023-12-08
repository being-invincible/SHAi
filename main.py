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
import google.generativeai as palm
import os
import pickle
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as OAI

tru = Tru()

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
# chat_engine = index.as_chat_engine(chat_mode="context",
#     memory=memory, verbose=True)

chat_engine = index.as_query_engine()

import numpy as np

# Initialize provider class
openai = OAI()

grounded = Groundedness(groundedness_provider=OAI())

# Define a groundedness feedback function
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
    TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

tru_query_engine_recorder = TruLlama(chat_engine,
    app_id='SHAi_App',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])

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
            response = chat_engine.query(prompt)
            st.write(response.response)
            with tru_query_engine_recorder as recording:
                chat_engine.query(prompt)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history