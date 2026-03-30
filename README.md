# Multi-Agent Supervisor and Swarm

Two multi-agent orchestration patterns for customer service built with LangGraph: a **supervisor pattern** where a central router dispatches to specialist agents, and a **swarm pattern** where agents hand off directly to each other using `Command`-based transfers. Both use `create_agent` with domain-specific tools and LangSmith tracing.

## Prerequisites

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/)
- [LangSmith API key](https://smith.langchain.com/)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |
| `LANGSMITH_API_KEY` | Yes | LangSmith API key for tracing and evals |
| `LANGSMITH_TRACING` | Yes | Set to `true` to enable tracing |

## Running

Run the supervisor pattern:

```bash
python supervisor.py
```

Run the swarm pattern:

```bash
python swarm.py
```

## Article

[[Multi-Agent Orchestration -- Supervisor and Swarm Patterns in LangGraph](#)
]([url](https://focused.io/lab/multi-agent-orchestration-in-langgraph-supervisor-vs-swarm-tradeoffs-and-architecture))
