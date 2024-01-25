from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import tiktoken
import streamlit as st

load_dotenv()

OPENAI_API_KEY="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"
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def get_answer(query):
    index_path = "faiss_index"
    if os.path.exists(index_path):
        embeddings = OpenAIEmbeddings()
        loaded_vectorstore = FAISS.load_local(index_path, embeddings)
    else:
        loader = PyPDFLoader("/Users/shivammitter/Desktop/AI/Gen_AI/embeddings/vectorestore-in-memory/48lawsofpower.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, length_function=tiktoken_len, separators=["\n\n", "\n"]
        )
        doc_f = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        loaded_vectorstore = FAISS.from_documents(doc_f, embeddings)
        loaded_vectorstore.save_local(index_path)

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=loaded_vectorstore.as_retriever())

    result = qa.invoke({"query": query})
    return result.get("result", "No answer available.")


def main():
    st.set_page_config(
        page_title="Chatisstan",
        page_icon=":balance_scale:", 
        layout="wide",
    )
    
    title_html = """
        <style>
            .title-text {
                color: #FFD700;
                font-weight: bold;
                font-size: 36px;
            }
        .subtitle-text {
                color:  #00FF00; 
                font-size: 18px;
            }
        </style>
        <div class="title-text">The 48 Laws Of Power</div>
        <div class="subtitle-text">Hi, I'm well-versed in 'The 48 Laws of Power,' so if knowledge were power, consider me a walking powerhouse. Don't believe me? Go ahead, quiz me—I've got 48 reasons to impress!</div>
    """
    st.markdown(title_html, unsafe_allow_html=True)
    st.image("logo.png", width=100, caption="")
    st.write("LES GO")


    query = st.text_input("Test me, ask me anything from the book ")

    if st.button("Submit"):
        result = get_answer(query)

        st.subheader("Result:")
        with st.container():
            st.success(result)


if __name__ == "__main__":
    main()
