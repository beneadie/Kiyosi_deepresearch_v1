
"""Research Agent Implementation.

This module implements a research agent that can perform iterative web searches
and synthesis to answer complex research questions.
"""

from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages

from deep_research.state_research import ResearcherState, ResearcherOutputState
from deep_research.utils import (
    tavily_search, get_today_str, think_tool, extract_text_from_response,
    google_search_grounding, get_subreddit_posts, get_reddit_post, search_term_in_subreddit,
    search_substack, read_substack_article
)
from deep_research.prompts import (
    research_agent_prompt,
    compress_research_system_prompt,
    compress_research_human_message,
    compress_discovery_system_prompt,
    compress_discovery_human_message
)
from deep_research.config import get_primary_model, get_writer_model, get_lite_model, get_resilient_model

# ===== CONFIGURATION =====

# Base tools for RESEARCH agents (ConductResearch)
# These are standard web search and Reddit tools
research_tools = [
    tavily_search, think_tool, get_reddit_post,
    google_search_grounding, get_subreddit_posts, search_term_in_subreddit
]
research_tools_by_name = {tool.name: tool for tool in research_tools}

# Extended tools for DISCOVERY agents (DiscoverOpportunities)
# Includes everything research agents have PLUS Substack tools
discovery_tools = [
    tavily_search, think_tool, get_reddit_post,
    google_search_grounding, get_subreddit_posts, search_term_in_subreddit,
    search_substack, read_substack_article  # Discovery-only tools
]
discovery_tools_by_name = {tool.name: tool for tool in discovery_tools}

# Initialize resilient models with tool bindings
model_with_research_tools = get_resilient_model(tools=research_tools)
model_with_discovery_tools = get_resilient_model(tools=discovery_tools)
summarization_model = get_resilient_model() # Base models also get fallbacks
compress_model = get_resilient_model(max_tokens=32000)

# ===== AGENT NODES =====

async def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Uses different tool sets based on agent_type:
    - "discovery": Gets Substack tools in addition to base tools
    - "researcher" (default): Gets only base research tools

    Returns updated state with the model's response.
    """
    # Select resilient model based on agent type
    if state.get("agent_type") == "discovery":
        model = model_with_discovery_tools
    else:
        model = model_with_research_tools

    messages = [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]

    # The model already has fallbacks configured
    response = await model.ainvoke(messages)

    return {
        "researcher_messages": [response]
    }

async def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses.
    Uses appropriate tool registry based on agent_type.
    Returns updated state with tool execution results.
    """
    # Select tool registry based on agent type
    if state.get("agent_type") == "discovery":
        tools_by_name = discovery_tools_by_name
    else:
        tools_by_name = research_tools_by_name

    tool_calls = state["researcher_messages"][-1].tool_calls

    # Execute all tool calls and collect search queries
    observations = []
    search_queries = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        # Track tavily_search queries
        if tool_call["name"] == "tavily_search":
            query = tool_call["args"].get("query", "")
            if query:
                search_queries.append(query)
        observations.append(await tool.ainvoke(tool_call["args"]))

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs, "search_queries": search_queries}

async def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """

    # Select prompts based on agent type
    if state.get("agent_type") == "discovery":
        system_prompt = compress_discovery_system_prompt
        human_prompt = compress_discovery_human_message
    else:
        system_prompt = compress_research_system_prompt
        human_prompt = compress_research_human_message

    system_message = system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=human_prompt.format(research_topic=state.get("research_topic", "")))]

    # Use resilient writer model for compression
    response = await compress_model.ainvoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        extract_text_from_response(m.content) for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": extract_text_from_response(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we have a final answer
    return "compress_research"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call") # Loop back for more research
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()
