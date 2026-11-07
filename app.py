import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG-Based Document Q&A")
st.caption("Ask questions about your PDFs. Answers are grounded in retrieved document passages.")

@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=180,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=generator)
    return embeddings, llm

def extract_pdf_text(files):
    pages = []
    for file in files:
        reader = PdfReader(file)
        for page_no, page in enumerate(reader.pages, start=1):
            content = page.extract_text() or ""
            if content.strip():
                pages.append(f"[{file.name}, page {page_no}]\n{content}")
    return "\n\n".join(pages)

uploaded = st.file_uploader("Upload one or more text-based PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded:
    with st.spinner("Reading and indexing documents..."):
        raw_text = extract_pdf_text(uploaded)
        if not raw_text.strip():
            st.error("No extractable text found. Scanned PDFs need OCR before indexing.")
            st.stop()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.create_documents([raw_text])
        embeddings, llm = load_components()
        db = FAISS.from_documents(chunks, embeddings)
        st.session_state["rag_db"] = db
        st.session_state["rag_llm"] = llm
        st.success(f"Indexed {len(chunks)} text chunks from {len(uploaded)} PDF(s).")

question = st.text_input("Ask a question about your uploaded documents")
if st.button("Get answer", type="primary", disabled=not question.strip()):
    if "rag_db" not in st.session_state:
        st.warning("Upload and index PDFs first.")
    else:
        docs = st.session_state["rag_db"].similarity_search(question, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = PromptTemplate.from_template(
            "Answer the question using only the context below. "
            "If the answer is not present, say you cannot find it in the documents. "
            "Do not invent facts.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        chain = prompt | st.session_state["rag_llm"]
        with st.spinner("Generating an answer..."):
            answer = chain.invoke({"context": context, "question": question})
        st.subheader("Answer")
        st.write(answer)
        with st.expander("Retrieved source passages"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Passage {i}**")
                st.write(doc.page_content)
else:
    st.info("Upload PDFs to create a searchable knowledge base.")
