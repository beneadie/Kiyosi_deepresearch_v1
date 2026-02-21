
"""Multi-agent supervisor for coordinating research across multiple specialized agents.

This module implements a supervisor pattern where:
1. A supervisor agent coordinates research activities and delegates tasks
2. Multiple researcher agents work on specific sub-topics independently
3. Results are aggregated and compressed for final reporting

The supervisor uses parallel research execution to improve efficiency while
maintaining isolated context windows for each research topic.
"""

import asyncio

from typing_extensions import Literal

from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research.prompts import lead_researcher_with_multiple_steps_diffusion_double_check_prompt, discovery_agent_prompt
from deep_research.research_agent import researcher_agent
from deep_research.state_multi_agent_supervisor import (
    SupervisorState,
    ConductResearch,
    ResearchComplete,
    DiscoverOpportunities
)
from deep_research.utils import get_today_str, think_tool, refine_draft_report
from deep_research.config import get_primary_model, RESEARCH_STRICT_TIMEOUT_MINUTES, MAX_RESEARCHER_ITERATIONS, get_resilient_model
import time
from deep_research.observability import log_conductor_turn, log_sub_agent, log_trace_delegation, log_trace_findings, log_trace_supervisor_reaction
from deep_research.console_logger import Colors
from deep_research import console_logger

def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history.

    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. When the supervisor delegates research to
    sub-agents via ConductResearch tool calls, each sub-agent returns its
    compressed findings as the content of a ToolMessage. This function
    extracts all such ToolMessage content to compile the final research notes.

    Args:
        messages: List of messages from supervisor's conversation history

    Returns:
        List of research note strings extracted from ToolMessage objects
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

# Ensure async compatibility for Jupyter environments
try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it


# ===== CONFIGURATION =====

supervisor_tools = [ConductResearch, ResearchComplete, DiscoverOpportunities, think_tool, refine_draft_report]
# Get a model that automatically falls back to alternates
supervisor_model_with_tools = get_resilient_model(tools=supervisor_tools)

# System constants
# Maximum number of tool call iterations for individual researcher agents
# This prevents infinite loops and controls research depth per topic
max_researcher_iterations = MAX_RESEARCHER_ITERATIONS # Calls to think_tool + ConductResearch + refine_draft_report

# Maximum number of concurrent research agents the supervisor can launch
# This is passed to the lead_researcher_prompt to limit parallel research tasks
max_concurrent_researchers = 4
max_concurrent_discovery = 2

# ===== SUPERVISOR NODES =====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """Coordinate research activities.

    Analyzes the research brief and current progress to decide:
    - What research topics need investigation
    - Whether to conduct parallel research
    - When research is complete

    Args:
        state: Current supervisor state with messages and research progress

    Returns:
        Command to proceed to supervisor_tools node with updated state
    """
    supervisor_messages = state.get("supervisor_messages", [])

    # Prepare system message with current date and constraints
    import time
    start_time = state.get("start_time", 0.0)
    elapsed_minutes = 0.0
    if start_time > 0:
        elapsed_minutes = (time.time() - start_time) / 60.0

    time_info = f"\n\nCURRENT PROGRESS: You have been researching for {elapsed_minutes:.1f} minutes."

    system_message = lead_researcher_with_multiple_steps_diffusion_double_check_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_concurrent_discovery_units=max_concurrent_discovery,
        max_researcher_iterations=max_researcher_iterations
    ) + time_info

    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # Make decision about next research steps
    response = await supervisor_model_with_tools.ainvoke(messages)

    # Console logging for real-time visibility
    iteration = state.get("research_iterations", 0) + 1
    console_logger.log_supervisor_start(iteration, elapsed_minutes)
    if response.tool_calls:
        console_logger.log_supervisor_tool_calls(response.tool_calls)

    # Log this conductor turn for observability
    log_conductor_turn(
        system_prompt=system_message,
        messages=supervisor_messages,
        response=response,
        elapsed_minutes=elapsed_minutes,
        iteration=iteration
    )

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """Execute supervisor decisions - either conduct research or end the process.

    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Determining when research is complete

    Args:
        state: Current supervisor state with messages and iteration count

    Returns:
        Command to continue supervision, end process, or handle errors
    """
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # Initialize variables for return pattern
    tool_messages = []
    all_raw_notes = []
    draft_report = state.get("draft_report", "")
    next_step = "supervisor"  # Default next step
    should_end = False

    # Check exit criteria first
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # Calculate elapsed minutes for hard stop check
    start_time = state.get("start_time", 0.0)
    elapsed_minutes = 0.0
    if start_time > 0:
        elapsed_minutes = (time.time() - start_time) / 60.0

    exceeded_time = elapsed_minutes >= RESEARCH_STRICT_TIMEOUT_MINUTES

    if exceeded_iterations or no_tool_calls or research_complete or exceeded_time:
        if exceeded_time:
             print(f"\n{Colors.RED}Hard stop triggered: Research exceeded {RESEARCH_STRICT_TIMEOUT_MINUTES} minutes limit.{Colors.RESET}")
        should_end = True
        next_step = END

    else:
        # Execute ALL tool calls before deciding next step
        try:
            # Separate think_tool calls from ConductResearch calls
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            refine_report_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "refine_draft_report"
            ]

            discover_opportunities_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "DiscoverOpportunities"
            ]

            # Handle think_tool calls (synchronous)
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                # Log supervisor's reaction for research trace
                log_trace_supervisor_reaction(tool_call["args"].get("reflection", ""))
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # Handle ConductResearch calls (asynchronous)
            if conduct_research_calls:
                # Log sub-agents starting
                agent_ids = []
                trace_indices = []  # Track trace indices for each delegation
                for tool_call in conduct_research_calls:
                    agent_id = console_logger.log_sub_agent_start(tool_call["args"]["research_topic"])
                    agent_ids.append(agent_id)
                    # Log delegation for research trace
                    trace_idx = log_trace_delegation(tool_call["args"]["research_topic"])
                    trace_indices.append(trace_idx)


                # Launch parallel research agents
                coros = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"]
                    })
                    for tool_call in conduct_research_calls
                ]

                # Wait for all research to complete
                tool_results = await asyncio.gather(*coros)

                # Format research results as tool messages
                # Each sub-agent returns compressed research findings in result["compressed_research"]
                # We write this compressed research as the content of a ToolMessage, which allows
                # the supervisor to later retrieve these findings via get_notes_from_tool_calls()
                research_tool_messages = []
                for i, (result, tool_call) in enumerate(zip(tool_results, conduct_research_calls)):
                    compressed = result.get("compressed_research", "Error synthesizing research report")
                    research_tool_messages.append(
                        ToolMessage(
                            content=compressed,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        )
                    )
                    # Log each sub-agent for observability
                    log_sub_agent(
                        research_topic=tool_call["args"]["research_topic"],
                        system_prompt="(see research_agent.py for sub-agent prompt)",
                        compressed_research=compressed,
                        agent_type="research_agent",
                        search_queries=result.get("search_queries", [])
                    )
                    # Log findings for research trace
                    if i < len(trace_indices):
                        log_trace_findings(trace_indices[i], compressed)
                    # Console log completion
                    if i < len(agent_ids):
                        console_logger.log_sub_agent_complete(agent_ids[i], len(result.get("search_queries", [])))

                tool_messages.extend(research_tool_messages)

                # Aggregate raw notes from all research
                all_raw_notes = [
                    "\n".join(result.get("raw_notes", []))
                    for result in tool_results
                ]

            for tool_call in refine_report_calls:
              console_logger.log_refine_start()
              notes = get_notes_from_tool_calls(supervisor_messages)
              findings = "\n".join(notes)

              draft_report = await refine_draft_report.ainvoke({
                    "research_brief": state.get("research_brief", ""),
                    "findings": findings,
                    "draft_report": state.get("draft_report", "")
              })

              tool_messages.append(
                ToolMessage(
                    content=draft_report,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
              )
              console_logger.log_refine_complete()

            # Handle DiscoverOpportunities calls (asynchronous)
            if discover_opportunities_calls:
                # Log discovery agents starting
                discovery_agent_ids = []
                discovery_trace_indices = []  # Track trace indices for each discovery delegation
                for tool_call in discover_opportunities_calls:
                    agent_id = console_logger.log_discovery_start(tool_call["args"]["discovery_brief"])
                    discovery_agent_ids.append(agent_id)
                    # Log delegation for research trace (prefix with "Discovery:" to distinguish)
                    trace_idx = log_trace_delegation("Discovery: " + tool_call["args"]["discovery_brief"])
                    discovery_trace_indices.append(trace_idx)

                # Reuse researcher_agent but with discovery-focused prompt from prompts module
                discovery_coros = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=discovery_agent_prompt + tool_call["args"]["discovery_brief"])
                        ],
                        "research_topic": "Discovery: " + tool_call["args"]["discovery_brief"],
                        "agent_type": "discovery"
                    })
                    for tool_call in discover_opportunities_calls
                ]

                discovery_results = await asyncio.gather(*discovery_coros)

                for i, (result, tool_call) in enumerate(zip(discovery_results, discover_opportunities_calls)):
                    compressed = result.get("compressed_research", "No discoveries found")
                    tool_messages.append(
                        ToolMessage(
                            content=compressed,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        )
                    )
                    log_sub_agent(
                        research_topic="Discovery: " + tool_call["args"]["discovery_brief"],
                        system_prompt=discovery_agent_prompt,
                        compressed_research=compressed,
                        agent_type="discovery_agent",
                        search_queries=result.get("search_queries", [])
                    )
                    # Log findings for research trace
                    if i < len(discovery_trace_indices):
                        log_trace_findings(discovery_trace_indices[i], compressed)
                    # Console log completion
                    if i < len(discovery_agent_ids):
                        console_logger.log_discovery_complete(discovery_agent_ids[i], len(result.get("search_queries", [])))


        except Exception as e:
            should_end = True
            next_step = END

    # Single return point with appropriate state updates
    if should_end:
        console_logger.log_research_complete()
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
                "draft_report": draft_report
            }
        )
    elif len(refine_report_calls) > 0:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes,
                "draft_report": draft_report
            }
        )
    else:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes
            }
        )


# ===== GRAPH CONSTRUCTION =====

# Build supervisor graph
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()

