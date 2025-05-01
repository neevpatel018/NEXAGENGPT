# 🤖 AI Assistant

A full-stack GPT-like AI agent with RAG capabilities, built using Streamlit, LangChain, and OpenAI's GPT-4.

## Quick Setup

1. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

## Features

- 💬 Chat interface with conversation history
- 📚 Document processing and RAG (Retrieval Augmented Generation)
- 🧠 GPT-4 powered responses
- 🔄 Persistent memory for context-aware conversations
- 📁 Support for PDF and text document uploads

## Prerequisites

- Python 3.8+
- OpenAI API key
- pip (Python package manager)

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── core/               # Core components
│   ├── llm_engine.py   # LLM interface
│   ├── rag_engine.py   # RAG implementation
│   └── memory.py       # Chat memory management
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Configuration

You can customize the following settings in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `CHROMA_DB_PATH`: Path to store the vector database
- `MODEL_NAME`: OpenAI model to use (default: gpt-4)
- `EMBEDDING_MODEL`: Embedding model to use (default: text-embedding-ada-002)
- `MAX_TOKENS`: Maximum tokens per response (default: 2000)
- `TEMPERATURE`: Response creativity (default: 0.7)

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.