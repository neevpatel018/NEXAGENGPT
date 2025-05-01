import streamlit as st
from core.llm_engine import LLMEngine
from core.rag_engine import RAGEngine
from core.memory import ChatMemory
import os

# Initialize components
@st.cache_resource
def init_components():
    return {
        "llm": LLMEngine(),
        "rag": RAGEngine(),
        "memory": ChatMemory()
    }

def main():
    st.title("🤖 AI Assistant")
    
    # Initialize session state
    if "components" not in st.session_state:
        st.session_state.components = init_components()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📚 Document Management")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
        
        if uploaded_file:
            # Save the uploaded file
            os.makedirs("uploads", exist_ok=True)
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the document
            with st.spinner("Processing document..."):
                docs = st.session_state.components["rag"].load_documents(file_path)
                st.session_state.components["rag"].create_vector_store(docs)
                st.success("Document processed successfully!")
    
    # Chat interface
    st.header("💬 Chat")
    user_input = st.text_input("You:", key="user_input")
    
    if user_input:
        # Add user message to history
        st.session_state.components["memory"].add_message("user", user_input)
        
        # Get conversation history
        history = st.session_state.components["memory"].get_history()
        
        # Generate response
        with st.spinner("Thinking..."):
            response = st.session_state.components["llm"].generate_response(history)
            
            # Add assistant message to history
            st.session_state.components["memory"].add_message("assistant", response)
            
            # Display response
            st.text_area("Assistant:", value=response, height=200)

if __name__ == "__main__":
    main() 