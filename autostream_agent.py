# AutoStream Conversational AI Agent
# Stack: LangChain + LangGraph

from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json

# ------------------ Knowledge Base ------------------
KB = {
    "pricing": {
        "Basic": "Basic Plan: $29/month, 10 videos/month, 720p resolution",
        "Pro": "Pro Plan: $79/month, Unlimited videos, 4K resolution, AI captions"
    },
    "policies": "No refunds after 7 days. 24/7 support only on Pro plan."
}

# ------------------ Mock Tool ------------------
def mock_lead_capture(name, email, platform):
    print(f"Lead captured successfully: {name}, {email}, {platform}")

# ------------------ State ------------------
class AgentState(TypedDict):
    history: list
    intent: Optional[str]
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]

# ------------------ LLM ------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ------------------ Nodes ------------------
def detect_intent(state: AgentState):
    last_msg = state["history"][-1]
    prompt = f"""
    Classify intent:
    greeting, product_inquiry, high_intent

    Message: {last_msg}
    """
    intent = llm.invoke(prompt).content.strip()
    state["intent"] = intent
    return state


def rag_answer(state: AgentState):
    query = state["history"][-1].lower()
    if "price" in query or "plan" in query:
        answer = f"{KB['pricing']['Basic']}\n{KB['pricing']['Pro']}"
    else:
        answer = KB['policies']
    state["history"].append(answer)
    print("Agent:", answer)
    return state


def lead_questions(state: AgentState):
    if not state.get("name"):
        print("Agent: May I know your name?")
    elif not state.get("email"):
        print("Agent: Please share your email.")
    elif not state.get("platform"):
        print("Agent: Which platform do you create content on?")
    return state


def capture_lead(state: AgentState):
    if state.get("name") and state.get("email") and state.get("platform"):
        mock_lead_capture(state['name'], state['email'], state['platform'])
    return state

# ------------------ Graph ------------------
builder = StateGraph(AgentState)

builder.add_node("intent", detect_intent)
builder.add_node("rag", rag_answer)
builder.add_node("lead", lead_questions)
builder.add_node("capture", capture_lead)

builder.set_entry_point("intent")

builder.add_conditional_edges(
    "intent",
    lambda s: s["intent"],
    {
        "greeting": "rag",
        "product_inquiry": "rag",
        "high_intent": "lead",
    },
)

builder.add_edge("lead", "capture")
builder.add_edge("rag", END)
builder.add_edge("capture", END)

app = builder.compile()

# ------------------ Run ------------------
state: AgentState = {
    "history": [],
    "intent": None,
    "name": None,
    "email": None,
    "platform": None,
}

print("AutoStream Agent Started (type 'exit' to quit)")

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    state["history"].append(user_input)
    state = app.invoke(state)
