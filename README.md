# AutoStream Social-to-Lead Agentic Workflow

Conversational AI agent for **ServiceHive ML Intern** assignment. Handles product queries, detects high-intent users, and captures leads using **LangGraph** stateful workflows.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org) [![LangChain](https://img.shields.io/badge/LangChain-Essential-brightgreen)](https://langchain.com) [![LangGraph](https://img.shields.io/badge/LangGraph-Stateful-orange)](https://langchain-ai.github.io/langgraph/)

## 🎯 Features Implemented

- ✅ **Intent Detection**: greeting, product_inquiry, high_intent  
- ✅ **RAG Knowledge Base**: Pricing plans, support policies
- ✅ **Lead Qualification**: Sequential flow (name → email → platform)
- ✅ **Tool Execution**: Mock lead capture API call
- ✅ **Stateful Memory**: LangGraph conversation persistence
- **State Schema:**
```python
class AgentState(TypedDict):
    history: list
    intent: Optional[str]
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream_agent

# 2. Setup environment
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt

# 3. Set OpenAI API key
$env:OPENAI_API_KEY="sk-your-key"

# 4. Run agent
python autostream_agent.py
User: Hi, tell me about your pricing
Agent: Basic: $29/mo (10 videos, 720p)
       Pro: $79/mo (Unlimited, 4K, AI captions)

User: I want Pro for my YouTube channel
Agent: May I know your name?
User: Aditi
Agent: Please share your email.
User: aditi@example.com
Agent: Which platform do you create content on?
User: YouTube
Agent: Lead captured: Aditi, aditi@example.com, YouTube ✓
