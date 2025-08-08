# Customer Service Bot

This is a FastAPI-powered customer support agent that leverages **Retrieval-Augmented Generation (RAG)** to provide accurate and friendly responses to customer queries. It combines a pre-defined FAQ knowledge base with OpenAI's language model to deliver warm, emoji-filled answers, ensuring a delightful user experience.

## Features
- **FastAPI Backend**: Lightweight and fast API for handling customer queries.
- **RAG Implementation**: Uses FAISS for vector-based retrieval and HuggingFace embeddings to find relevant FAQ answers.
- **Friendly Persona**: Configurable agent persona with warm greetings and emoji-rich responses.
- **OpenAI Integration**: Powered by OpenAI's `gpt-4o-mini` model for natural and accurate responses.
- **Extensible FAQ**: Easily update the FAQ data to expand the knowledge base.

## Prerequisites
To run the bot locally, ensure you have the following installed:
- Python 3.8+
- pip for installing dependencies
- An OpenAI API key (stored in a `.env` file)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/haiderze/customer_service_bot.git
   cd customer_service_bot
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your-openai-api-key
   ```

## Usage
1. **Run the FastAPI Server**:
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the API**:
   - Open your browser or use a tool like Postman to visit:
     ```
     http://localhost:8000/
     ```
     This will display a welcome message from the bot.

3. **Ask a Question**:
   Use the `/ask` endpoint to query the bot:
   ```
   http://localhost:8000/ask?question=What%20are%20your%20opening%20hours%3F
   ```
   Example response:
   ```json
   {
     "question": "What are your opening hours?",
     "answer": "We're open from 9 AM to 9 PM every day! 🕘"
   }
   ```

## Example Questions
Try these questions to test the bot:
- "What are your opening hours?"
- "Do you offer home delivery?"
- "Where are your products manufactured?"
- "What is the meaning of life?" (This will trigger the "unknown answer" response)

## Project Structure
- `main.py`: Core application code, including FastAPI setup, RAG pipeline, and FAQ data.
- `.env`: Environment file for storing the OpenAI API key (not tracked in Git).
- `requirements.txt`: List of Python dependencies.

## Dependencies
Key libraries used:
- `fastapi`: For the API framework.
- `langchain`: For text splitting, embeddings, and vector store.
- `faiss-cpu`: For efficient similarity search.
- `huggingface_hub`: For pre-trained embeddings.
- `httpx`: For async HTTP requests to OpenAI.
- `python-dotenv`: For loading environment variables.

Install them using:
```bash
pip install fastapi uvicorn httpx python-dotenv langchain langchain-community faiss-cpu sentence-transformers
```

## Configuration
The bot's behavior can be customized in the `config` dictionary in `main.py`:
- `agent_name`: Name of the bot (default: configurable).
- `agent_persona`: Description of the bot's tone and style.
- `greeting_options`: List of welcome messages.
- `unknown_answer`: Fallback response for unknown queries.
- `chatgpt_model`: OpenAI model (default: `gpt-4o-mini`).
- `retriever_k`: Number of FAQ chunks to retrieve (default: 2).

## Adding New FAQs
To expand the knowledge base, modify the `faq_data` list in `main.py`:
```python
faq_data = [
    {"question": "Your new question?", "answer": "Your new answer!"},
    ...
]
```
The vector store will automatically update with the new FAQs when the server restarts.

## Limitations
- Requires an active internet connection for OpenAI API calls.
- Limited to the FAQ data provided; unhandled questions return a fallback response.
- Current embeddings model (`all-MiniLM-L6-v2`) is lightweight but may not handle complex queries as well as larger models.

## Future Improvements
- Add support for more dynamic knowledge bases (e.g., loading FAQs from a database).
- Implement caching for faster responses.