from typing import TypedDict, List

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[BaseMessage]


def main():
    model = ChatOpenAI(temperature=0)

    tools = []

    # 🔹 agent node (model lives here)
    def agent_node(state: AgentState):
        # call the model with all messages
        response = model.invoke(state["messages"])
        # return updated messages
        return {"messages": state["messages"] + [response]}

    # 🔹 tool node (tools live here)
    tool_node = ToolNode(tools)

    # 🔹 create graph
    graph = StateGraph(AgentState)

    # 🔹 plug nodes into graph
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    app = graph.compile()

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # create initial state with user message
        user_message = HumanMessage(content=user_input)
        state = {"messages": [user_message]}

        # call agent node directly
        new_state = agent_node(state)

        # print assistant response
        print("Assistant:", new_state["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()
