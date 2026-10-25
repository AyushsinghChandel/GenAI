from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from state import BlogState
from agents import get_llm, researcher_agent, writer_agent, editor_agent


MAX_REVISIONS = 3

### NOde - Edges 
### Defining the nodes

def researcher_node(state: BlogState) -> BlogState:
    """Researcher Agent generates the research outline based on the topic and audience."""
    llm = get_llm()
    agent = researcher_agent(llm, state.topic, state.audience, state.research_feedback)
    state.research = agent
    state.research_feedback = ""
    return state


def human_review_node(state: BlogState) -> BlogState:
    """Puse and ask the human to approve the research or send the feedback"""
    decision = interrupt({
        "state": "research_review",
        "research": state.research,
        "instructions": (
            "Reply with 'approve' to continue to writing",
            "or describe the changes you want to see in the research outline."
        )
    })
    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve", "yes", "ok", "approved", "process", "continue", ""] else "revise"
        feedback = "" if action == "approve" else text
    
    state.research_feedback = feedback
    return state


def writer_node(state: BlogState) -> BlogState:
    """Writer Agent produces the full draft blog (or revises it)"""
    llm = get_llm()
    draft_data = writer_agent(
        llm,
        topic=state.topic,
        audience=state.audience,
        research=state.research,
        feedback=state.draft_feedback,
    )
    state.draft = draft_data
    state.draft_feedback = ""
    state.revision_count += 1
    return state


def human_review_draft_node(state: BlogState) -> BlogState:
    """Puse and ask the human to approve the draft or send the feedback"""
    decision = interrupt({
        "state": "draft_review",
        "draft": state.draft,
        "instructions": (
            "Reply with 'approve' to continue to editor",
            "or describe the changes you want to see in the draft."
        )
    })
    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve", "yes", "ok", "approved", "process", "continue", ""] else "revise"
        feedback = "" if action == "approve" else text
    
    state.draft_feedback = feedback
    return state


def editor_node(state: BlogState) -> BlogState:
    llm = get_llm()
    final_blog = editor_agent(llm, state.topic, state.draft)
    state.final_blog = final_blog
    return state


#### Conditional Edges
def route_after_research_review(state: BlogState) -> Literal["researcher_node", "writer_node"]:
    if state.research_feedback:
        return "researcher_node"
    else:
        return "writer_node"



def route_after_draft_review(state: BlogState) -> Literal["writer_node", "editor_node"]:
    if state.draft_feedback and state.revision_count < MAX_REVISIONS:
        return "writer_node"
    return "editor_node"
    
    
### Build and complie the graph
def build_blog_graph():
    builder = StateGraph(BlogState)
    
    ### add nodes
    builder.add_node("researcher_node", researcher_node)
    builder.add_node("human_review_node", human_review_node)
    builder.add_node("writer_node", writer_node)
    builder.add_node("human_review_draft_node", human_review_draft_node)
    builder.add_node("editor_node", editor_node)
    
    ### add edges
    builder.add_edge(START, "researcher_node")
    builder.add_edge("researcher_node", "human_review_node")
    builder.add_conditional_edges("human_review_node", route_after_research_review, {
        "researcher_node": "researcher_node",
        "writer_node": "writer_node"
    })
    builder.add_edge("writer_node", "human_review_draft_node")
    builder.add_conditional_edges("human_review_draft_node", route_after_draft_review, {
        "writer_node": "writer_node",
        "editor_node": "editor_node"
    })
    builder.add_edge("editor_node", END)

    GRAPH = builder.compile(checkpointer=InMemorySaver())
    return GRAPH

