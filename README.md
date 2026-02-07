#🤖 AI Chatbot with Tool Integration

An intelligent conversational AI assistant powered by OpenAI's GPT models and LangGraph that can perform calculations, greet users, and engage in natural conversations with streaming responses.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- 💬 **Natural Conversation**: Engage in fluid, context-aware conversations
- 🧮 **Built-in Calculator**: Perform arithmetic calculations on the fly
- 👋 **Friendly Greetings**: Personalized hello messages
- ⚡ **Streaming Responses**: ChatGPT-style character-by-character output
- 🔧 **Extensible Tools**: Easy to add custom tools and capabilities
- 🎯 **Smart Tool Selection**: AI automatically chooses the right tool for each task

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-chatbot-tools.git
cd ai-chatbot-tools
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Run the chatbot**
```bash
python main.py
```

## 📦 Requirements

Create a `requirements.txt` file with:
```
langchain==0.1.0
langchain-openai==0.0.2
langchain-core==0.1.0
langgraph==0.0.20
python-dotenv==1.0.0
openai==1.3.0
```

## 🎮 Usage

### Basic Conversation
```
You: Hello!
Assistant: Hi! How can I help you today?

You: What is 25 + 17?
Assistant: [Tool executing...]
The sum of 25 and 17 is 42.

You: Say hello to John
Assistant: Hello John, I hope you are well today.

You: quit
```

### Example Interactions

**Calculator Tool:**
- "What is 5 + 3?"
- "Calculate 100 * 25"
- "Add 42 and 17"

**Greeting Tool:**
- "Say hello to Alice"
- "Greet Bob"
- "Say hi to Sarah"

**General Chat:**
- "Tell me a joke"
- "What's the weather like?"
- "Explain quantum computing"

## 🛠️ How It Works

### Architecture
```
User Input → Agent Node → Decision → Tool Node → Agent Node → Response
                ↑                         ↓
                └─────────────────────────┘
```

1. **Agent Node**: Processes user input and decides if tools are needed
2. **Tool Selection**: AI automatically chooses the appropriate tool
3. **Tool Execution**: Selected tool performs the task
4. **Response Generation**: AI formulates the final response
5. **Streaming Output**: Character-by-character display

### Tool System

The chatbot uses LangGraph to manage tool execution:
```python
@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations with numbers"""
    return f"The sum of {a} and {b} = {a+b}"

@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    return f"Hello {name}, I hope you are well today."
```

## 📋 Project Structure
```
ai-chatbot-tools/
│
├── main.py                # Main chatbot application
├── .env                   # Environment variables (not in repo)
├── .env.example          # Example env file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore file
```

## 📄 Complete Code Files

### `main.py`
```python
from typing import TypedDict, List, Annotated
import sys
import time

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()


@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations with numbers"""
    print("\n[Tool executing...]\n")
    return f"The sum of {a} and {b} = {a+b}"


@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    print("\n[Tool executing...]\n")
    return f"Hello {name}, I hope you are well today."


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def main():
    model = ChatOpenAI(temperature=0)

    tools = [calculator, say_hello]
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: AgentState):
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    app = graph.compile()

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        user_message = HumanMessage(content=user_input)

        print("Assistant: ", end="", flush=True)

        # Collect the final response
        final_content = ""
        for chunk in app.stream({"messages": [user_message]}, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                if node_name == "agent":
                    last_msg = node_output["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        if not (
                            hasattr(last_msg, "tool_calls") and last_msg.tool_calls
                        ):
                            final_content = last_msg.content

        # Print character by character with delay
        for char in final_content:
            print(char, end="", flush=True)
            time.sleep(0.03)  # Adjust for speed (0.03 = 30ms per character)

        print("\n")


if __name__ == "__main__":
    main()
```

### `.env.example`
```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
```

### `.gitignore`
```
# Environment variables
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### `requirements.txt`
```
langchain==0.1.0
langchain-openai==0.0.2
langchain-core==0.1.0
langgraph==0.0.20
python-dotenv==1.0.0
openai==1.3.0
```

## 🔧 Adding Custom Tools

Create new tools by defining functions with the `@tool` decorator:
```python
@tool
def weather_checker(city: str) -> str:
    """Get current weather for a city"""
    # Your implementation here
    return f"The weather in {city} is sunny!"

# Add to tools list
tools = [calculator, say_hello, weather_checker]
```

## ⚙️ Configuration

### Adjust Streaming Speed

In `main.py`, modify the delay:
```python
time.sleep(0.01)  # Very fast (like ChatGPT)
time.sleep(0.03)  # Medium speed (default)
time.sleep(0.05)  # Slower
time.sleep(0.1)   # Very slow
```

### Change AI Model
```python
model = ChatOpenAI(
    temperature=0,
    model="gpt-4",  # or "gpt-3.5-turbo", "gpt-4-turbo"
)
```

### Adjust Response Length
```python
model = ChatOpenAI(
    temperature=0,
    max_tokens=2000  # Increase for longer responses
)
```

## 🎯 Key Concepts

### LangGraph State Management
The chatbot uses `Annotated[List[BaseMessage], add_messages]` to automatically manage conversation history.

### Tool Binding
Tools are bound to the model so it knows when and how to use them:
```python
model_with_tools = model.bind_tools(tools)
```

### Conditional Edges
The graph routes between nodes based on whether tools are needed:
```python
graph.add_conditional_edges("agent", should_continue)
```

## 🔒 Security Notes

- Never commit your `.env` file or API keys to version control
- Keep your OpenAI API key secure
- Monitor API usage to avoid unexpected charges
- Validate user input in production environments

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingTool`)
3. Commit your changes (`git commit -m 'Add weather tool'`)
4. Push to the branch (`git push origin feature/AmazingTool`)
5. Open a Pull Request

## 🎯 Future Enhancements

- [ ] Web search integration
- [ ] Memory/conversation history persistence
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] GUI interface (Streamlit/Gradio)
- [ ] Custom tool marketplace
- [ ] Rate limiting and usage tracking
- [ ] Conversation export feature

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI](https://openai.com/)
- Inspired by modern conversational AI systems

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Tutorial](https://github.com/langchain-ai/langgraph)
- [OpenAI API Reference](https://platform.openai.com/docs/)

## ⚠️ Disclaimer

This is a demonstration project. For production use, implement proper error handling, input validation, rate limiting, and security measures.

---

Made with ❤️ and AI | Powered by LangChain & OpenAI