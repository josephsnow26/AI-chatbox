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
    print("Tool has been called.")
    return f"Hello {name}, i hope you are well today."


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
            time.sleep(0.03)  # 🔹 Adjust this for speed (0.03 = 30ms per character)

        print("\n")


if __name__ == "__main__":
    main()
