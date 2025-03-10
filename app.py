from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from dotenv import load_dotenv
load_dotenv()
import os
import requests
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
import bs4
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set custom headers for web requests
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Initialize global variables for RAG system
rag_system = None
vector_store = None
initialized = False

BASE_URL = "https://brainlox.com"

# Improved course data loading with web scraping only
def extract_text(soup, selector, default_value="Not available"):
    """Extracts text based on a CSS selector."""
    elem = soup.select_one(selector)
    return elem.get_text(strip=True) if elem else default_value

def extract_list_items(soup):
    """Extracts price, lessons, duration, and total price from the course page."""
    items = soup.select("ul.info > li")

    price = "Not available"
    lessons = "Not available"
    duration = "Not available"
    total_price = "Not available"

    for item in items:
        text = item.get_text(strip=True)

        if "$" in text and price == "Not available":
            price = text.replace("$", "").strip()
        elif "hour" in text or "session" in text:
            duration = text.strip()
        elif text.isdigit():
            lessons = text.strip()
        elif "$" in text and price != "Not available":
            total_price = text.replace("$", "").strip()

    return price, lessons, duration, total_price

def scrape_course_page(course_url):
    """Scrapes course details from an individual course page."""
    try:
        response = requests.get(course_url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch course page: {course_url}")
            return "Price not available", "Lessons not available", "Duration not available", "Total price not available"

        soup = bs4.BeautifulSoup(response.text, "html.parser")
        return extract_list_items(soup)

    except Exception as e:
        logger.error(f"Error scraping course page {course_url}: {str(e)}")
        return "Price not available", "Lessons not available", "Duration not available", "Total price not available"

def load_course_data():
    """Scrapes Brainlox courses including title, description, price, lessons, duration, and total price."""
    try:
        logger.info("Scraping Brainlox course listing page...")
        response = requests.get(f"{BASE_URL}/courses/category/technical", headers=headers, timeout=15)

        if response.status_code != 200:
            logger.error(f"Failed to retrieve course data. HTTP Status: {response.status_code}")
            return []

        soup = bs4.BeautifulSoup(response.text, "html.parser")
        course_elements = soup.select(".single-courses-box")

        if not course_elements:
            logger.warning("No course elements found on the main page.")
            return []

        docs = []
        for course in course_elements:
            # Extract title
            title_elem = course.select_one(".courses-content h3 a")
            title = title_elem.text.strip() if title_elem else "Untitled Course"

            # Extract course URL
            course_url = f"{BASE_URL}{title_elem['href']}" if title_elem and title_elem.has_attr("href") else None

            # Extract description
            desc_elem = course.select_one(".courses-content p")
            description = desc_elem.text.strip() if desc_elem else "No description available"

            price, lessons, duration, total_price = "Not available", "Not available", "Not available", "Not available"

            if course_url:
                price, lessons, duration, total_price = scrape_course_page(course_url)

            # Create document
            content = f"{title} - {description}. Price: {price}. Lessons: {lessons}. Duration: {duration}. Total Price: {total_price}."

            docs.append(Document(
                page_content=content,
                metadata={"source": "brainlox.com", "title": title, "price": price, "lessons": lessons, "duration": duration, "total_price": total_price}
            ))

        logger.info(f"Successfully extracted {len(docs)} courses.")
        return docs

    except Exception as e:
        logger.error(f"Error loading course data: {str(e)}")
        return []

def recognize_intent(question):
    question = question.lower().strip()
    greetings = ["hi", "hello", "hey", "howdy"]
    return "greeting" if any(g in question for g in greetings) else "course_query"

# Process documents and create embeddings
def setup_rag_system():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Initialize vector store
    vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    
    # Check if we need to load data
    if vector_store._collection.count() == 0:
        logger.info("Vector store is empty. Loading initial data...")
        docs = load_course_data()
        if not docs:
            logger.error("No course data available.")
            return None, None
        process_documents(docs, vector_store)
    
    # Initialize LLM
    llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("API_KEY", ""))
    
    # Define improved RAG prompt
    custom_prompt_template = """You are a helpful assistant for Brainlox.com technical courses. 
Follow these steps:
1. If the user greets or makes small talk, respond politely and briefly
2. For course inquiries, use the context below to answer
3. If context is missing, say you don't know

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    
    # Define state for RAG application
    class State(TypedDict):
        question: str
        context: List
        answer: str
    
    # Define improved RAG steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"], k=10)
        return {"context": retrieved_docs}
    
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant for Brainlox courses."},
            {"role": "user", "content": prompt.format(question=state["question"], context=docs_content)}
        ]
        
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    # Compile RAG application
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    return graph, vector_store

# Process documents function with improved chunking
def process_documents(docs, vector_store):
    if not docs:
        return False
        
    logger.info("Processing documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    
    vector_store.add_documents(documents=all_splits)
    logger.info(f"Added {len(all_splits)} document chunks to vector store")
    return True

# Initialize the RAG system on startup
def initialize_rag_system():
    global rag_system, vector_store, initialized
    
    if not initialized:
        logger.info("Initializing RAG system...")
        os.environ["USER_AGENT"] = headers["User-Agent"]
        
        rag_system, vector_store = setup_rag_system()
        if rag_system is None:
            return False
        
        initialized = True
        return True
    return True


@app.route("/")
def home():
    return {"message": "Welcome to the Brainlox Course Assistant API!"}, 200

# API Resources
class HealthCheck(Resource):
    def get(self):
        return {"status": "ok", "initialized": initialized}

class RefreshCourseData(Resource):
    def post(self):
        global vector_store, rag_system, initialized
        
        try:
            if vector_store:
                # Get all document IDs first
                all_ids = vector_store._collection.get()["ids"]
                
                # Then delete using these IDs
                if all_ids:
                    vector_store._collection.delete(ids=all_ids)
                    logger.info(f"Vector store cleared successfully! Deleted {len(all_ids)} documents.")
                else:
                    logger.info("Vector store is already empty.")
            
            initialized = False
            success = initialize_rag_system()
            
            if success:
                return {"status": "success", "message": "Course data refreshed successfully"}
            else:
                return {"status": "error", "message": "Failed to refresh course data"}, 500
        except Exception as e:
            logger.error(f"Error refreshing course data: {str(e)}")
            return {"status": "error", "message": str(e)}, 500

class Conversation(Resource):
    def post(self):
        if not initialize_rag_system():
            return {"status": "error", "message": "RAG system initialization failed"}, 500
        
        try:
            # Force Flask to parse JSON even if Content-Type is missing
            data = request.get_json(force=True, silent=True)
        except Exception as e:
            return {"status": "error", "message": f"Invalid JSON: {str(e)}"}, 400
        
        # Check if message key is present
        if not data or "message" not in data:
            return {"status": "error", "message": "No message provided"}, 400
        
        user_message = data["message"]
        
        # Get conversation history if provided
        conversation_history = data.get("conversation_history", [])

        # Determine intent
        intent = recognize_intent(user_message)

        if intent == "greeting":
            answer = "Hello! I'm here to help with Brainlox technical courses. Ask me anything about available courses!"
        else:
            try:
                response = rag_system.invoke({"question": user_message})
                answer = response["answer"]
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return {"status": "error", "message": "Failed to generate response"}, 500

        # Add to history and return
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": answer})

        return {
            "status": "success",
            "answer": answer,
            "conversation_history": conversation_history
        }

# API Routes
api.add_resource(HealthCheck, '/api/health')
api.add_resource(RefreshCourseData, '/api/refresh')
api.add_resource(Conversation, '/api/chat')

# Run the initialization on startup
if __name__ == '__main__':
    initialize_rag_system()
    app.run(debug=True, host='0.0.0.0', port=5000)
