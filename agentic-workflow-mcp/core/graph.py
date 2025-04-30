import json
import os

from functools import partial
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_ollama import ChatOllama
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import  StructuredTool
from mcp.server.fastmcp import Context

from tools.file_system import list_files, read_file, read_multiple_files, write_file
from tools.web_search import web_search
from tools.embedding_retriever import retrieve_embeddings
from core.log import log_message
from core.model import DefaultWorkflowState, DEFAULT_WORKFLOW_CONFIG, WorkflowConfig, AgentConfig
from core.util import typed_dict_from_json_schema, get_full_schema, ensure_utf8

def generate_graph_from_workflow(workflow_config: WorkflowConfig, logs: list, ctx: Context) -> CompiledStateGraph:
    """
    Generates a graph from the workflow configuration supporting flexible LangGraph structures
    including conditional branching, dynamic routing, parallel execution, orchestrator-worker pattern, 
    evaluator-optimizer pattern, and complex workflows.

    Args:
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
        user_prompt (str): The user prompt to process.
        logs (list): List to store log messages during graph generation and execution.
        ctx (Context): The MCP context containing application resources like the default LLM.

    Returns:
        Compiled StateGraph: The compiled sequential state graph ready for execution.
    """
    log_message(logs, "Generating sequential graph from workflow configuration.")

    schema = get_full_schema(workflow_config)
    graph = None
    if schema:
        WorkflowState = typed_dict_from_json_schema("WorkflowState", schema)
        graph = StateGraph(WorkflowState)
    else:
        graph = StateGraph(DefaultWorkflowState)

    agents = workflow_config.get("agents", [])
    if not agents:
        log_message(logs, "No agents found in the workflow configuration. Returning empty compiled graph.")
        return graph.compile() # Return an empty but compiled graph

    # Add all nodes to the graph based on the agents defined in the workflow config
    add_nodes(graph, agents, workflow_config, ctx, logs)

    if "orchestrator" in workflow_config:
        # Handle orchestrator-worker pattern if specified
        handle_orchestrator_flow(workflow_config, graph, ctx, logs)
    if "evaluator_optimizer" in workflow_config:
        # Handle evaluator-optimizer pattern if specified
        handle_evaluator_optimizer_flow(workflow_config, graph, logs)
    if "edges" in workflow_config:
        # Add edges based on configuration (for non-orchestrator patterns)
        handle_non_orchestrator_edges(workflow_config, graph)
    elif not "orchestrator" in workflow_config and not "evaluator_optimizer" in workflow_config:
        handle_sequential_flow(workflow_config, graph)

    # Add any branches or conditional logic defined in workflow config
    add_branches(workflow_config, graph)
    # Add any dynamic routing defined in workflow config
    add_routing(workflow_config, graph)
    # Handle parallel flow paths
    handle_parallel_flow(workflow_config, graph)

    # Compile the graph
    compiled_graph = graph.compile()
    log_message(logs, "Graph compilation complete.")

    return compiled_graph


def prepare_prompt_and_flags(agent_name: str, prompt_template: str, workflow_config: WorkflowConfig, state: dict) -> tuple:
    """
    Prepares the prompt template and flags for parallel and orchestrator nodes.

    Args:
        agent_name (str): The name of the agent node.
        prompt_template (str): The prompt template for the agent.
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
        state (dict): The current state of the workflow.
    
    Returns:
        tuple: A tuple containing the updated prompt template, and flags indicating if the node is a parallel or orchestrator worker node.
    """
    is_parallel_node = False
    is_orchestrator_worker_node = False

    if "parallel" in workflow_config:
        for parallel in workflow_config.get("parallel", []):
            if agent_name in parallel.get("nodes", []):
                is_parallel_node = True
            if agent_name == parallel.get("join"):
                for node in parallel.get("nodes", []):
                    node_output = state.get(node + "_output", "")
                    prompt_template += f"\nResponse from {node}: {node_output}"

    if "orchestrator" in workflow_config:
        if agent_name in workflow_config["orchestrator"].get("workers", []):
            is_orchestrator_worker_node = True
        if agent_name == workflow_config["orchestrator"].get("aggregator"):
            workers = [agent for agent in workflow_config.get("agents", []) if agent["name"] in state.get("workers", [])]
            for worker in workers:
                worker_output = state.get(worker + "_output", "")
                prompt_template += f"\nResponse from {worker}: {worker_output}"

    return prompt_template, is_parallel_node, is_orchestrator_worker_node


def invoke_llm(model: ChatOllama, prompt_template: str, current_input: str, tools: list, output_format: dict, agent_name: str, logs: list) -> str:
    """
    Invokes the LLM with the given prompt template and input, ensuring all inputs are UTF-8 encoded.

    Args:
        model (ChatOllama): The LLM model to use for invocation.
        prompt_template (str): The prompt template for the agent.
        current_input (str): The current input to the LLM.
        tools (list): List of tools available to the agent.
        output_format (dict): The expected output format for the LLM response.
        agent_name (str): The name of the agent node.
        logs (list): List to store log messages during invocation.

    Returns:
        str: The LLM response.
    """

    llm_response = None
    try:
        log_message(logs, ensure_utf8(f"Node {agent_name} using tools: {[getattr(t, '__name__', str(t)) for t in tools]}"))
        final_model = model
        if tools:
            final_model = final_model.bind_tools(tools)
        if output_format:
            final_model = final_model.with_structured_output(schema=output_format)
        # Ensure all inputs are utf-8
        safe_prompt = ensure_utf8(prompt_template)
        safe_input = ensure_utf8(current_input)
        llm_response = final_model.invoke([
            SystemMessage(content=safe_prompt),
            HumanMessage(content=safe_input)
        ])
        log_message(logs, ensure_utf8(f"Node {agent_name} LLM response (truncated): {llm_response}..."))
    except Exception as e:
        error_msg = ensure_utf8(f"Error invoking LLM for node {agent_name}: {e}")
        log_message(logs, error_msg)
        llm_response = ensure_utf8(f"Error in {agent_name}: {e}")
    return llm_response


def handle_tool_calls(llm_response: AIMessage, tools: list, logs: list, workspace_path: str, ctx: Context) -> dict:
    """
    Handles tool calls from the LLM response and executes them.

    Args:
        llm_response (AIMessage): The LLM response containing tool calls.
        tools (list): List of tools available to the agent.
        logs (list): List to store log messages during tool invocation.
        workspace_path (str): The workspace path for file operations.
    
    Returns:
        dict: A dictionary containing the results of the tool calls.
    """
    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        tool_results = []
        for tool_call in llm_response.tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("args", {})
            arguments["workspace_path"] = workspace_path
            arguments["logs"] = logs
            arguments["ctx"] = ctx
            tool_func = next(
                (t for t in tools if getattr(t, "name", None) == tool_name or getattr(t, "__name__", None) == tool_name),
                None
            )
            if tool_func:
                try:
                    if hasattr(tool_func, "invoke"):
                        result = tool_func.invoke(arguments)
                    else:
                        result = tool_func(**arguments)
                    log_message(logs, f"Tool '{tool_name}' invoked successfully with result: {result}")
                except Exception as tool_exc:
                    result = f"Tool '{tool_name}' error: {tool_exc}"
                    log_message(logs, f"Error invoking tool '{tool_name}': {tool_exc}")
            else:
                result = f"Tool '{tool_name}' not found"
                log_message(logs, f"Tool '{tool_name}' not found in tools list.")
            tool_results.append({"tool": tool_name, "result": result})
        return json.dumps({"tool_results": tool_results})
    return None


def parse_llm_response(llm_response: AIMessage, output_format: str) -> dict:
    """
    Parses the LLM response based on the specified output format.

    Args:
        llm_response (AIMessage): The LLM response to parse.
        output_format (str): The expected output format for the LLM response.

    Returns:
        dict: The parsed LLM response.
    """
    if output_format:
        llm_response_json = llm_response
        if isinstance(llm_response, str):
            llm_response_json = json.loads(llm_response)
        if isinstance(llm_response, AIMessage):
            if isinstance(llm_response.content, str):
                llm_response_json = json.loads(llm_response.content)
            else:
                llm_response_json = llm_response.content
        if isinstance(llm_response_json, str):
            llm_response_json = ensure_utf8(llm_response_json)
        elif isinstance(llm_response_json, dict):
            llm_response_json = {k: ensure_utf8(v) for k, v in llm_response_json.items()}
        return llm_response_json
    else:
        if hasattr(llm_response, "content"):
            content = llm_response.content
            if isinstance(content, str):
                return ensure_utf8(content)
            return content
        return ensure_utf8(llm_response)


def process_llm_response(
    llm_response: AIMessage, agent_name: str, agent_config: AgentConfig, output_format: dict, state: dict, workflow_config: WorkflowConfig,
    is_parallel_node: bool, logs: list, workspace_path: str, tools: list, ctx: Context
) -> dict:
    """
    Processes the LLM response and updates the state based on the agent configuration.

    Args:
        llm_response (AIMessage): The LLM response to process.
        agent_name (str): The name of the agent node.
        agent_config (AgentConfig): The configuration for this agent node.
        output_format (dict): The expected output format for the LLM response.
        state (dict): The current state of the workflow.
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
        is_parallel_node (bool): Flag indicating if the node is part of a parallel execution.
        logs (list): List to store log messages during processing.
        workspace_path (str): The workspace path for file operations.
        tools (list): List of tools available to the agent.
        ctx (Context): The MCP context containing application resources.

    Returns:
        dict: A dictionary containing the updated state after processing the LLM response.
    """
    try:
        llm_response_json = handle_tool_calls(llm_response, tools, logs, workspace_path, ctx)
        if llm_response_json is None:
            llm_response_json = parse_llm_response(llm_response, output_format)

        newInput = llm_response_json
        output_keys = agent_config.get("output_decision_keys", ["class"])
        result = None

        if output_format or (
            "parallel" in workflow_config and any(
                parallel.get("source") == agent_name for parallel in workflow_config.get("parallel", [])
            )
        ):
            newInput = state.get("input", "")
            if isinstance(llm_response_json, str):
                newInput += llm_response_json

        if output_format:
            result = {}
            for output_key in output_keys:
                result[output_key] = llm_response_json.get(output_key, "")

        update_dict = {
            f"{agent_name}_output": llm_response_json
        }

        if not is_parallel_node:
            update_dict["input"] = newInput
            update_dict["final_output"] = llm_response_json

        if result:
            for output_key in output_keys:
                update_dict[output_key] = result.get(output_key, "")
    except Exception as e:
        log_message(logs, f"Error processing LLM response for node {agent_name}: {e} {repr(e)}")
        return {}

    return update_dict


def agent_node_action(
        state: dict, model: ChatOllama, prompt_template: str, tools: list,
        agent_config: AgentConfig, workflow_config: WorkflowConfig,
        logs: list, ctx: Context
    ) -> dict:
    """
    Executes the action for a specific agent node in the graph.

    Args:
        state (dict): The current state of the workflow.
        model (ChatOllama): The LLM model to use for invocation.
        prompt_template (str): The prompt template for the agent.
        tools (list): List of tools available to the agent.
        agent_config (AgentConfig): The configuration for this agent node.
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
        logs (list): List to store log messages during execution.
        ctx (Context): The MCP context containing application resources.

    Returns:
        dict: A dictionary containing the updated state after executing the agent node.
    """
    agent_name = agent_config["name"]
    output_format = agent_config.get("output_format", None)
    workspace_path = os.getenv("WORKSPACE_PATH", ".")
    log_message(logs, f"Executing node: {agent_name}")
    current_input = state.get("input", "")
    log_message(logs, f"Node {agent_name} input (truncated): {current_input[:100]}...")

    prompt_template, is_parallel_node, is_orchestrator_worker_node = prepare_prompt_and_flags(
        agent_name, prompt_template, workflow_config, state
    )

    if is_orchestrator_worker_node and agent_name not in state.get("workers", []):
        log_message(logs, f"Skipping node {agent_name} as it is not being orchestrated.")
        return {}

    llm_response = invoke_llm(
        model, prompt_template, current_input, tools, output_format, agent_name, logs
    )

    update_dict = process_llm_response(
        llm_response, agent_name, agent_config, output_format, state, workflow_config,
        is_parallel_node, logs, workspace_path, tools, ctx
    )

    return update_dict


def add_tools(agent_config: AgentConfig, agent_name: str, logs: list) -> list:
    """
    Adds tools to the agent based on the configuration.

    Args:
        agent_config (AgentConfig): The configuration for this agent node.
        agent_name (str): The name of the agent.

    Returns:
        list: A list of tool functions available to this agent.
    """
    agent_tools = []
    tool_names = agent_config.get("tools", [])
    tool_functions = agent_config.get("tool_functions", [])
    tool_function = None
    for tool_name in tool_names:
        # Check if the tool is defined in the config
        if tool_name in tool_functions:
            function_string = tool_functions[tool_name].get("function_string", None)
            tool_description = tool_functions[tool_name].get("description", "No description provided")
            if function_string:
                # Create a structured tool from the function string
                tool_function = StructuredTool.from_function(
                    func=eval(function_string),
                    name=tool_name,
                    description=tool_description
                )
        # Map tool names from config to actual tool functions
        # Assumes tool functions are accessible in the current scope (e.g., imported)
        if tool_name == "read_file":
            agent_tools.append(tool_function if tool_function else read_file)
        elif tool_name == "read_multiple_files":
            agent_tools.append(tool_function if tool_function else read_multiple_files)
        elif tool_name == "list_files":
            agent_tools.append(tool_function if tool_function else list_files)
        elif tool_name == "write_file":
            agent_tools.append(tool_function if tool_function else write_file)
        elif tool_name == "web_search":
            agent_tools.append(tool_function if tool_function else web_search)
        elif tool_name == "retrieve_embeddings":
            agent_tools.append(tool_function if tool_function else retrieve_embeddings)
        # Add mappings for other tools here...
        # elif tool_name == "some_other_tool":
        #     agent_tools.append(some_other_tool_function)
        else:
            log_message(logs, f"Warning: Tool '{tool_name}' for agent '{agent_name}' is defined in config but not mapped to a function.")
    log_message(logs, f"Tools configured for {agent_name}: {[getattr(t, '__name__', str(t)) for t in agent_tools]}")

    return agent_tools


def add_nodes(graph: StateGraph, agents: list, workflow_config: WorkflowConfig, ctx: Context, logs: list) -> None:
    """
    Adds nodes to the graph based on the agents defined in the workflow configuration.

    Args:
        graph (StateGraph): The state graph to which nodes will be added.
        agents (list): List of agent configurations from the workflow config.
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
        ctx (Context): The MCP context containing application resources.
        logs (list): List to store log messages during node addition.

    Returns:
        None
    """
    workspace_path = os.getenv('WORKSPACE_PATH', '.')
    for agent_config in agents:
        agent_name = agent_config["name"]
        model_name = agent_config.get("model_name", None)
        output_format = agent_config.get("output_format", None)
        log_message(logs, f"Configuring agent node: {agent_name}")

        # Determine the LLM for this agent node
        if model_name or output_format:
            # Initialize a specific Ollama model for this agent
            model_name = agent_config.get("model_name", workflow_config.get("default_model", DEFAULT_WORKFLOW_CONFIG["default_model"]))
            model = ChatOllama(
                model=model_name,
                temperature=agent_config.get("temperature", workflow_config.get("default_temperature", DEFAULT_WORKFLOW_CONFIG["default_temperature"])),
            )
            log_message(logs, f"Using specific model for {agent_name}: {model_name}")
        else:
            # Fallback to the default LLM from the application context
            model = ctx.request_context.lifespan_context.llm
            log_message(logs, f"Using default LLM for {agent_name}")

        # Get the prompt template for the agent
        # Default to just passing the input if no prompt is specified
        prompt_template = agent_config.get("prompt", "{input}")

        if agent_config.get("prompt_file"):
            # Read the prompt from a file if specified
            prompt_file_path = agent_config["prompt_file"]
            try:
                # Prepend workspace_path if not absolute
                if not os.path.isabs(prompt_file_path):
                    prompt_file_path = os.path.join(workspace_path, prompt_file_path)
                with open(prompt_file_path, "r") as file:
                    prompt_template = file.read()
                log_message(logs, f"Loaded prompt from file for {agent_name}: {prompt_file_path}")
            except FileNotFoundError:
                log_message(logs, f"Prompt file not found for {agent_name}: {prompt_file_path}. Using default prompt.")
            except Exception as e:
                log_message(logs, f"Error reading prompt file for {agent_name}: {e}. Using default prompt.")

        # Prepare the list of tools for this agent
        agent_tools = add_tools(agent_config, agent_name, logs)

        # Use functools.partial to create a node-specific action function
        # This binds the model, prompt, tools, agent_name, and logs list for the current node
        specific_node_action = partial(
            agent_node_action,
            model=model,
            prompt_template=prompt_template,
            tools=agent_tools,
            agent_config=agent_config,
            workflow_config=workflow_config,
            logs=logs,
            ctx=ctx
        )

        # Add the configured node to the graph
        graph.add_node(agent_name, specific_node_action)
        log_message(logs, f"Added node '{agent_name}' to the graph.")


def handle_orchestrator_flow(workflow_config: WorkflowConfig, graph: StateGraph, ctx: Context, logs: list) -> None:
    """
    Handles the orchestrator-worker pattern in the workflow configuration.

    Args:
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
        graph (StateGraph): The state graph to which nodes will be added.
        ctx (Context): The MCP context containing application resources.

    Returns:
        None
    """
    orchestrator_config = workflow_config["orchestrator"]
    orchestrator_name = orchestrator_config["name"]
    # Get agent config for all worker names from workers
    worker_names = orchestrator_config.get("workers", [])
    workers = [agent for agent in workflow_config.get("agents", []) if agent["name"] in worker_names]
    completion_condition = orchestrator_config.get("completion_condition", DEFAULT_WORKFLOW_CONFIG["default_orchestrator_completion_condition"])
            
    # Add orchestrator node if not already present in agents
    agent_names = [agent["name"] for agent in workflow_config.get("agents", [])]
    if orchestrator_name not in agent_names:
        # Use add_nodes to add orchestrator as a node
        add_nodes(graph, [orchestrator_config], workflow_config, ctx, logs)

    # Connect orchestrator to workers and back
    for worker in workers:
        worker_name = worker["name"]
        # Add edge from orchestrator to worker
        graph.add_edge(orchestrator_name, worker_name)
        # Add conditional edge from worker back to orchestrator for supervision
        if orchestrator_config.get("supervise_workers", True):
            graph.add_conditional_edges(
                worker_name,
                eval(completion_condition),
                {True: END, False: orchestrator_name}
            )

    # Connect START to orchestrator
    graph.add_edge(START, orchestrator_name)

    # Connect orchestrator to END if specified
    if orchestrator_config.get("can_end_workflow", True):
        graph.add_conditional_edges(
            orchestrator_name,
            eval(completion_condition),
            {True: END, False: orchestrator_name}
        )


def handle_evaluator_optimizer_flow(workflow_config: WorkflowConfig, graph: StateGraph, logs: list) -> None:
    """
    Handles the evaluator-optimizer pattern in the workflow configuration.

    Args:
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
        graph (StateGraph): The state graph to which nodes will be added.
        logs (list): List to store log messages during node addition.

    Returns:
        None
    """
    eo_config = workflow_config["evaluator_optimizer"]
    executor_name = eo_config.get("executor")
    evaluator_name = eo_config.get("evaluator")
    optimizer_name = eo_config.get("optimizer")
    
    # Verify that all required nodes exist
    if not all([executor_name, evaluator_name, optimizer_name]):
        log_message(logs, "Error: Evaluator-optimizer flow requires executor, evaluator, and optimizer nodes.")
    
    if not "edges" in workflow_config:
        # Connect START to executor
        graph.add_edge(START, executor_name)
    
    # Connect executor to evaluator
    graph.add_edge(executor_name, evaluator_name)
    
    # Add conditional branching from evaluator based on quality threshold
    quality_condition = eval(eo_config.get("quality_condition", 
                                            "lambda state: state.get('quality_score', 0) >= state.get('quality_threshold', 0.7)"))
    
    graph.add_conditional_edges(
        evaluator_name,
        quality_condition,
        {
            True: END,  # If quality is good enough, end the workflow
            False: optimizer_name  # Otherwise, send to optimizer for improvement
        }
    )
    
    # Connect optimizer back to executor for another attempt
    graph.add_edge(optimizer_name, executor_name)
    
    # Add iteration limit check if specified
    if "max_iterations" in eo_config:
        iteration_condition = eval(f"lambda state: state.get('iteration_count', 0) < {eo_config['max_iterations']}")
        
        # Apply iteration condition to optimizer output
        graph.add_conditional_edges(
            optimizer_name,
            iteration_condition,
            {
                True: executor_name,  # Continue to executor if under max iterations
                False: END  # End if max iterations reached
            }
        )


def handle_non_orchestrator_edges(workflow_config: WorkflowConfig, graph: StateGraph) -> None:
    """
    Handles non-orchestrator edges in the workflow configuration.
    
    Args:
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
        graph (StateGraph): The state graph to which edges will be added.
    
    Returns:
        None
    """
    # Use explicitly defined edges
    for edge in workflow_config["edges"]:
        source = edge["source"]
        target = edge["target"]
        
        # Handle conditional routing if specified
        if "condition" in edge:
            condition_func = eval(edge["condition"])  # This requires careful validation
            graph.add_conditional_edges(source, condition_func, target)
        else:
            graph.add_edge(source, target)


def handle_sequential_flow(workflow_config: WorkflowConfig, graph: StateGraph) -> None:
    """
    Handles the default sequential flow in the workflow configuration.

    Args:
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
        graph (StateGraph): The state graph to which nodes will be added.

    Returns:
        None
    """
    # Default sequential flow if no edges are specified
    agents = workflow_config["agents"]
    for i, agent in enumerate(agents):
        if i == 0:
            graph.add_edge(START, agent["name"])
        elif i == len(agents) - 1:
            graph.add_edge(agents[i-1]["name"], agent["name"])
            graph.add_edge(agent["name"], END)
        else:
            graph.add_edge(agents[i-1]["name"], agent["name"])


def add_branches(workflow_config: WorkflowConfig, graph: StateGraph) -> None:
    """
    Adds branches to the graph based on the workflow configuration.

    Args:
        graph (StateGraph): The state graph to which branches will be added.
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
    
    Returns:
        None
    """
    for branch in workflow_config.get("branches", []):
        source = branch["source"]
        condition = eval(branch["condition"])  # This requires careful validation
        targets = branch["targets"]
        
        graph.add_conditional_edges(
            source,
            condition,
            {result: target for result, target in targets.items()}
        )


def add_routing(workflow_config: WorkflowConfig, graph: StateGraph) -> None:
    """
    Adds dynamic routing to the graph based on the workflow configuration.

    Args:
        graph (StateGraph): The state graph to which routing will be added.
        workflow_config (WorkflowConfig): The workflow configuration dictionary.

    Returns:
        None
    """
    for router in workflow_config.get("routers", []):
        source = router["source"]
        router_func = eval(router["router_function"])  # This requires careful validation
        graph.add_conditional_edges(source, router_func, {
            **{agent["name"]: agent["name"] for agent in workflow_config.get("agents", [])}
        })


def handle_parallel_flow(workflow_config: WorkflowConfig, graph: StateGraph) -> None:
    """
    Handles parallel flow paths in the workflow configuration.

    Args:
        workflow_config (WorkflowConfig): The workflow configuration dictionary.
        graph (StateGraph): The state graph to which parallel flow paths will be added.

    Returns:
        None
    """
    for parallel in workflow_config.get("parallel", []):
        # Each parallel section defines a branch point and multiple parallel nodes
        source = parallel.get("source", START)
        nodes = parallel.get("nodes", [])
        join_node = parallel.get("join", END)

        # Add edges from source to all parallel nodes
        for node in nodes:
            graph.add_edge(source, node)
            # Add edges from all parallel nodes to join node
            graph.add_edge(node, join_node)
