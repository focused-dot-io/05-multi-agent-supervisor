"""Evaluation suite for multi-agent supervisor."""

from langsmith import Client, evaluate
from openevals.llm import create_llm_as_judge
from langchain_core.messages import HumanMessage, AIMessage

from supervisor import supervisor_graph

ls_client = Client()

dataset_name = "multi-agent-supervisor-evals"

try:
    dataset = ls_client.create_dataset(
        dataset_name=dataset_name,
        description="Multi-agent supervisor evaluation dataset",
    )
    ls_client.create_examples(
        dataset_id=dataset.id,
        inputs=[
            {"question": "I want to upgrade my plan to Enterprise."},
            {"question": "My SSO integration is broken, error code SAML-401."},
            {"question": "Can you waive my setup fee and also fix my SSO?"},
        ],
        outputs=[
            {"must_mention": ["plan", "upgrade"], "expected_agents": ["account"]},
            {"must_mention": ["SSO", "SAML"], "expected_agents": ["tech_support"]},
            {"must_mention": ["fee", "SSO"], "expected_agents": ["billing", "tech_support"]},
        ],
    )
    print(f"Created dataset: {dataset_name}")
except Exception:
    print(f"Dataset '{dataset_name}' already exists, reusing.")

QUALITY_PROMPT = """\
Customer query: {inputs}
Agent response: {outputs}

Rate 0.0-1.0 on completeness, accuracy, and appropriate routing.
Return ONLY: {{"score": <float>, "reasoning": "<explanation>"}}"""

quality_judge = create_llm_as_judge(
    prompt=QUALITY_PROMPT,
    model="anthropic:claude-sonnet-4-5-20250929",
    feedback_key="quality",
    continuous=True,
)


def coverage(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    text = outputs.get("response", "").lower()
    must_mention = reference_outputs.get("must_mention", [])
    hits = sum(1 for t in must_mention if t.lower() in text)
    return {"key": "coverage", "score": hits / len(must_mention) if must_mention else 1.0}


def agent_routing(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    notes = outputs.get("resolution_notes", [])
    notes_text = " ".join(notes).lower()
    expected = reference_outputs.get("expected_agents", [])
    if not expected:
        return {"key": "agent_routing", "score": 1.0}
    hits = sum(1 for a in expected if a.lower() in notes_text)
    return {"key": "agent_routing", "score": hits / len(expected)}


def target(inputs: dict) -> dict:
    result = supervisor_graph.invoke({
        "messages": [HumanMessage(content=inputs["question"])],
        "current_agent": "",
        "resolution_notes": [],
    })
    response = ""
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            response = msg.content
    return {
        "response": response,
        "resolution_notes": result.get("resolution_notes", []),
    }


if __name__ == "__main__":
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[quality_judge, coverage, agent_routing],
        experiment_prefix="multi-agent-supervisor-v1",
        max_concurrency=4,
    )
    print("\nEvaluation complete. Check LangSmith for results.")
