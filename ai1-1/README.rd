# NERD Agent - Document-Based AI Assistant

A sophisticated AI agent built with LangChain that analyzes PDF documents and provides intelligent question-answering capabilities. The agent specializes in processing academic materials and can generate test questions based on uploaded content.

## Features

- **PDF Document Processing**: Automatically extracts text from PDF files using PyPDF
- **Vector Search**: Uses FAISS vector store with OpenAI embeddings for semantic document retrieval
- **Intelligent Q&A**: Powered by GPT-4o for accurate, context-aware responses
- **Interactive Chat**: Command-line interface for continuous questioning
- **Test Generation**: Can create practice questions based on document content
- **Memory Management**: Handles context length limitations gracefully

## Prerequisites

- Python 3.11+
- OpenAI API key
- Virtual environment (recommended)

## Installation

1. **Clone the repository and navigate to the project directory**

2. **Create and activate a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install required dependencies**:
```bash
pip install -r requirements.txt
```

Required packages include:
- `langchain`
- `langchain-community`
- `langchain-openai`
- `faiss-cpu`
- `pypdf`
- `python-dotenv`
- `langgraph`

4. **Set up environment variables**:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Project Structure

```
nerd-agent/
├── .venv/                  # Virtual environment
├── docs/                   # PDF documents directory
│   ├── probability1.pdf    # Sample probability theory document
│   └── calc_lecture.pdf    # Sample calculus lecture notes
├── nerd-agent.ipynb       # Main Jupyter notebook
├── .env                   # Environment variables
└── requirements.txt       # Python dependencies
```

## Usage

### Basic Setup

1. **Place your PDF documents** in the `docs/` folder
2. **Open the Jupyter notebook**: `jupyter notebook nerd-agent.ipynb`
3. **Run the cells sequentially** to:
   - Extract text from PDFs
   - Create vector embeddings
   - Initialize the QA chain
   - Start the interactive chat

### Interactive Q&A

The agent provides a continuous chat interface:

```python
while True:
    query = input("Ask your question: ")
    
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.run(query)
    print("AI Agent:", response)
```

**Example interactions**:
- "What is the definition of sample space in probability?"
- "Explain linear approximation in calculus"
- "Generate 5 practice questions on probability theory"

### Key Components

#### 1. Document Processing
```python
def extract_text_pypdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text
```

#### 2. Vector Store Creation
```python
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever()
```

#### 3. QA Chain Setup
```python
llm = ChatOpenAI(model='gpt-4o', openai_api_key=os.getenv('OPENAI_API_KEY'))
qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever)
```

## Customization

### Custom Prompts

You can customize the agent's behavior by adding custom prompts:

```python
from langchain.prompts import PromptTemplate

custom_prompt = """You are a specialized tutor with 10 years of experience.
Answer questions based on the provided documents with precise explanations.

Context: {context}
Question: {question}
Answer:"""

prompt_template = PromptTemplate(
    template=custom_prompt,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_llm(
    llm=llm, 
    retriever=retriever,
    prompt=prompt_template
)
```

## Troubleshooting

### Common Issues

1. **Context Length Exceeded Error**:
   ```
   BadRequestError: This model's maximum context length is 8192 tokens
   ```
   **Solution**: The agent includes error handling to reset conversation context when limits are reached.

2. **Missing Dependencies**:
   ```
   NameError: name 'PromptTemplate' is not defined
   ```
   **Solution**: Ensure all imports are included:
   ```python
   from langchain.prompts import PromptTemplate
   ```

3. **API Key Issues**:
   **Solution**: Verify your `.env` file contains the correct OpenAI API key.

### Performance Tips

- **Document Size**: For large PDFs, consider chunking documents into smaller sections
- **Token Management**: Monitor token usage to avoid API limits
- **Vector Store**: Save and load FAISS indexes to avoid recomputing embeddings

## Example Documents

The repository includes sample academic documents:

- **probability1.pdf**: Covers probability theory fundamentals, sample spaces, and probability functions
- **calc_lecture.pdf**: MIT calculus lecture on linear and quadratic approximations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the Jupyter notebook comments
3. Create an issue in the repository

---

**Note**: This agent is designed for educational and research purposes. Ensure you have appropriate rights to process any PDF documents you upload.
