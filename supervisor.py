import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.agents import create_agent
from langsmith import traceable
from pydantic import BaseModel, Field

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)


class CustomerServiceState(MessagesState):
    current_agent: str
    resolution_notes: Annotated[list[str], operator.add]


# --- Tools ---

@tool
def lookup_billing_info(customer_id: str) -> str:
    """Look up billing information for a customer."""
    return (
        f"Customer {customer_id}: Enterprise plan, $2,400/mo, "
        f"next billing date 2026-03-01, payment method: invoice."
    )


@tool
def apply_discount(customer_id: str, discount_percent: int) -> str:
    """Apply a discount to a customer's account."""
    return f"Applied {discount_percent}% discount to customer {customer_id}."


@tool
def diagnose_sso(customer_id: str, error_code: str) -> str:
    """Diagnose SSO integration issues."""
    return (
        f"SSO diagnosis for {customer_id}: Error {error_code} indicates "
        f"SAML certificate expiration. Certificate expired 2026-02-04. "
        f"Resolution: regenerate SAML certificate in IdP and re-upload."
    )


@tool
def check_system_status(service: str) -> str:
    """Check the status of a service."""
    return f"Service {service}: operational, 99.97% uptime last 30 days."


@tool
def lookup_account_details(customer_id: str) -> str:
    """Look up account details and plan information."""
    return (
        f"Customer {customer_id}: Enterprise plan since 2024-06, "
        f"5 seats, primary contact: jane@example.com, "
        f"account manager: Sarah Chen."
    )


@tool
def update_plan(customer_id: str, new_plan: str) -> str:
    """Update a customer's plan."""
    return f"Plan updated for {customer_id}: now on {new_plan}."


# --- Specialist Agents ---

billing_agent = create_agent(
    llm,
    tools=[lookup_billing_info, apply_discount],
    system_prompt="You are a billing specialist. Help customers with invoices, "
           "payments, discounts, and plan pricing. Be precise with numbers. "
           "Customer ID is 'C-1042' unless otherwise specified.",
)

tech_agent = create_agent(
    llm,
    tools=[diagnose_sso, check_system_status],
    system_prompt="You are a technical support specialist. Help customers diagnose "
           "and resolve technical issues. Provide specific remediation steps. "
           "Customer ID is 'C-1042' unless otherwise specified.",
)

account_agent = create_agent(
    llm,
    tools=[lookup_account_details, update_plan],
    system_prompt="You are an account management specialist. Help customers with "
           "plan changes, upgrades, and account administration. "
           "Customer ID is 'C-1042' unless otherwise specified.",
)


# --- Supervisor ---

class RoutingDecision(BaseModel):
    next_agent: str = Field(
        description="The next agent to handle the request: "
                    "'billing', 'tech_support', 'account', or 'DONE'"
    )
    reasoning: str = Field(description="Why this agent was chosen")


routing_llm = llm.with_structured_output(RoutingDecision)


@traceable(name="supervisor", run_type="chain")
def supervisor(state: CustomerServiceState) -> dict:
    notes = "\n".join(state.get("resolution_notes", []))
    history_context = f"\n\nAlready resolved:\n{notes}" if notes else ""

    response = routing_llm.invoke([
        SystemMessage(
            content="You are a customer service supervisor. Analyze the "
                    "conversation and decide which specialist should handle "
                    "the next part of the request.\n\n"
                    "Available agents:\n"
                    "- billing: invoices, payments, discounts, pricing\n"
                    "- tech_support: technical issues, SSO, integrations, bugs\n"
                    "- account: plan changes, upgrades, account administration\n"
                    "- DONE: the customer's request has been fully addressed\n\n"
                    "Do NOT re-route to an agent that has already handled its "
                    "portion of the request." + history_context
        ),
        *state["messages"],
    ])
    return {"current_agent": response.next_agent}


# --- Specialist Wrappers ---

@traceable(name="billing_node", run_type="chain")
def billing_node(state: CustomerServiceState) -> dict:
    result = billing_agent.invoke({"messages": state["messages"]})
    return {
        "messages": result["messages"][-1:],
        "resolution_notes": [f"Billing: {result['messages'][-1].content[:200]}"],
    }


@traceable(name="tech_support_node", run_type="chain")
def tech_support_node(state: CustomerServiceState) -> dict:
    result = tech_agent.invoke({"messages": state["messages"]})
    return {
        "messages": result["messages"][-1:],
        "resolution_notes": [f"Tech Support: {result['messages'][-1].content[:200]}"],
    }


@traceable(name="account_node", run_type="chain")
def account_node(state: CustomerServiceState) -> dict:
    result = account_agent.invoke({"messages": state["messages"]})
    return {
        "messages": result["messages"][-1:],
        "resolution_notes": [f"Account: {result['messages'][-1].content[:200]}"],
    }


# --- Graph ---

def route_to_agent(state: CustomerServiceState) -> str:
    agent = state.get("current_agent", "DONE")
    if agent == "DONE":
        return "end"
    return agent


builder = StateGraph(CustomerServiceState)

builder.add_node("supervisor", supervisor)
builder.add_node("billing", billing_node)
builder.add_node("tech_support", tech_support_node)
builder.add_node("account", account_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "billing": "billing",
        "tech_support": "tech_support",
        "account": "account",
        "end": END,
    },
)

builder.add_edge("billing", "supervisor")
builder.add_edge("tech_support", "supervisor")
builder.add_edge("account", "supervisor")

supervisor_graph = builder.compile()


if __name__ == "__main__":
    result = supervisor_graph.invoke({
        "messages": [HumanMessage(
            content="I want to upgrade my plan, but first I need help fixing "
                    "my SSO — it's been broken since last Tuesday. "
                    "Also, can you waive the setup fee?"
        )],
        "current_agent": "",
        "resolution_notes": [],
    })

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"Agent: {msg.content[:200]}...")
            print()

    print("Resolution notes:")
    for note in result.get("resolution_notes", []):
        print(f"  - {note}")
