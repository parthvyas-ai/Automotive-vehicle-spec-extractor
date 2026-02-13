# importing dependencies
import os
import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template


# -------------------- CUSTOM EXTRACTION PROMPT --------------------

custom_template = """
You are an automotive service manual specification extraction engine.

Using ONLY the provided context, extract the specification requested in the user query.

Rules:
- Extract ALL matching records from ALL sections in the context.
- Return structured output as a JSON array.
- Each record must contain:
    - "Section" (give full name of section only)
    - "component"
    - "spec_type" (example: Torque etc)
    - "value"
    - "unit"
- Return ONLY valid JSON.
- If nothing is found, return [].
- Do NOT explain anything.
- Do NOT add extra text.

Context:
{context}

Question:
{question}

JSON Output:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=custom_template,
    input_variables=["context", "question"]
)


# -------------------- PDF TEXT EXTRACTION --------------------

def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


# -------------------- SECTION-BASED CHUNKING --------------------

def get_chunks(raw_text):
    sections = re.split(r'(SECTION\s+\d+[-\dA-Z]*:.*)', raw_text)

    chunks = []
    for i in range(1, len(sections), 2):
        section_header = sections[i].strip()
        section_content = sections[i + 1] if i + 1 < len(sections) else ""
        full_section = section_header + "\n" + section_content
        chunks.append(full_section.strip())

    return chunks


# -------------------- VECTOR STORE --------------------

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


# -------------------- CONVERSATION CHAIN --------------------

def get_conversationchain(vectorstore, api_key, model_name, temperature, k):

    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature
    )

    # Hybrid Retrieval (case-insensitive)
    all_docs = vectorstore.similarity_search("specifications", k=5000)

    filtered_texts = [
        doc.page_content for doc in all_docs
        if "specifications" in doc.page_content.lower()
    ]

    if filtered_texts:
        filtered_vectorstore = faiss.FAISS.from_texts(
            texts=filtered_texts,
            embedding=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        )
        retriever = filtered_vectorstore.as_retriever(search_kwargs={"k": k})
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
    )

    return conversation_chain


# -------------------- HANDLE USER QUESTION --------------------

def handle_question(question):
    response = st.session_state.conversation({
        'question': question,
        'chat_history': []
    })

    answer = response["answer"]

    st.write(user_template.replace("{{MSG}}", question),
             unsafe_allow_html=True)

    st.write(bot_template.replace("{{MSG}}", answer),
             unsafe_allow_html=True)


# -------------------- MAIN FUNCTION --------------------

def main():
    st.set_page_config(page_title="Automotive Vehicle Specification Extraction System",
                       page_icon="🚗")
    st.write(css, unsafe_allow_html=True)

    # Hide default Streamlit footer and replace it
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("🚗📝 Vehicle Specification Extraction System")

    st.markdown(
        "<p style='font-size:18px;'><b>How to use: First add all entries in left side bar, upload pdf then click Process and wait till you get message: 'Ready! You can now ask questions.'</b></p>",
        unsafe_allow_html=True
    )

    question = st.text_input("Ask a specification question from your service manual:")
    if question and st.session_state.conversation:
        handle_question(question)

    # -------------------- SIDEBAR --------------------

    with st.sidebar:
        st.subheader("🔑 OpenAI Settings")

        api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password"
        )

        model_name = st.selectbox(
            "Select GPT Model:",
            [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ]
        )

        st.markdown("---")
        st.subheader("⚙️ Parameters (keep default values for first use)")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )

        k = st.slider(
            "Top Similar Sections (k)",
            min_value=1,
            max_value=50,
            value=20,
            step=1
        )

        st.markdown("---")
        st.subheader("📄 Your Documents")

        docs = st.file_uploader(
            "Upload your Automotive Service Manual PDFs and click Process",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if not api_key:
                st.error("Please enter your OpenAI API key.")
                return

            if not docs:
                st.error("Please upload at least one PDF.")
                return

            with st.spinner("Processing..."):
                raw_text = get_pdf_text(docs)
                text_chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversationchain(
                    vectorstore,
                    api_key,
                    model_name,
                    temperature,
                    k
                )

                st.success("Ready! You can now ask questions..")

    st.markdown("<hr><center>Made with ❤️ for PREDII</center>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
