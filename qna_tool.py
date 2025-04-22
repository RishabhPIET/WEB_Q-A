import streamlit as st
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import uuid

# Initialize session state for storing content and index
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to scrape text content from a URL
def scrape_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts, styles, and navigation elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Extract text from paragraphs, headings, and articles
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article'])
        text = ' '.join([element.get_text(strip=True) for element in text_elements])
        
        # Clean text: remove extra spaces, newlines, and special characters
        text = re.sub(r'\s+', ' ', text).strip()
        return text if text else "No relevant content found."
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# Function to chunk text into smaller pieces for embedding
def chunk_text(text, max_length=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Function to add content to vector store
def add_to_vector_store(text, url):
    chunks = chunk_text(text)
    if not chunks:
        return False
    
    embeddings = st.session_state.model.encode(chunks)
    
    if st.session_state.index is None:
        dimension = embeddings.shape[1]
        st.session_state.index = faiss.IndexFlatL2(dimension)
        st.session_state.embeddings = embeddings
    else:
        st.session_state.embeddings = np.vstack([st.session_state.embeddings, embeddings])
    
    st.session_state.index.add(embeddings)
    st.session_state.documents.extend([(chunk, url) for chunk in chunks])
    return True

# Function to retrieve relevant content for a question
def retrieve_relevant_content(question, k=3):
    query_embedding = st.session_state.model.encode([question])
    if st.session_state.index is None:
        return []
    
    distances, indices = st.session_state.index.search(query_embedding, k)
    relevant_chunks = [(st.session_state.documents[idx][0], st.session_state.documents[idx][1]) for idx in indices[0]]
    return relevant_chunks

# Function to generate answer from retrieved content
def generate_answer(question, relevant_chunks):
    if not relevant_chunks:
        return "No relevant content available to answer the question."
    
    # Combine relevant chunks
    context = ' '.join([chunk for chunk, _ in relevant_chunks])
    if not context.strip():
        return "No relevant content available to answer the question."
    
    # Simple extractive answer generation: find sentences containing key terms
    sentences = re.split(r'(?<=[.!?])\s+', context)
    question_terms = set(question.lower().split())
    
    relevant_sentences = []
    for sentence in sentences:
        sentence_terms = set(sentence.lower().split())
        if question_terms.intersection(sentence_terms):
            relevant_sentences.append(sentence)
    
    if not relevant_sentences:
        return "No direct answer found in the content."
    
    # Return the most relevant sentence or a summary
    return ' '.join(relevant_sentences[:2])  # Limit to 2 sentences for conciseness

# Streamlit UI
st.title("Web Content Q&A Tool")
st.write("Enter URLs to scrape content and ask questions based on that content.")

# URL Input Section
st.subheader("Step 1: Input URLs")
url_input = st.text_input("Enter a URL (e.g., https://example.com):")
if st.button("Scrape URL"):
    if url_input:
        with st.spinner("Scraping content..."):
            content = scrape_url(url_input)
            if content.startswith("Error"):
                st.error(content)
            else:
                if add_to_vector_store(content, url_input):
                    st.success(f"Content from {url_input} successfully scraped and stored!")
                else:
                    st.error("Failed to process content. No relevant text found.")
    else:
        st.warning("Please enter a valid URL.")

# Display stored URLs
if st.session_state.documents:
    st.subheader("Stored URLs")
    unique_urls = list(set([url for _, url in st.session_state.documents]))
    for url in unique_urls:
        st.write(f"- {url}")

# Question Input Section
st.subheader("Step 2: Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if question:
        with st.spinner("Retrieving answer..."):
            relevant_chunks = retrieve_relevant_content(question)
            answer = generate_answer(question, relevant_chunks)
            st.subheader("Answer")
            st.write(answer)
            if relevant_chunks:
                st.subheader("Sources")
                for chunk, url in relevant_chunks:
                    st.write(f"- From {url}")
    else:
        st.warning("Please enter a question.")

# Clear stored data
if st.button("Clear Stored Content"):
    st.session_state.documents = []
    st.session_state.embeddings = None
    st.session_state.index = None
    st.success("Stored content cleared!")