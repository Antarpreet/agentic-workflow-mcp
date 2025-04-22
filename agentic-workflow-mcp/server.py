import asyncio
from mcp.server.fastmcp import FastMCP
import httpx
import json
from datetime import datetime

mcp = FastMCP("Local Agentic Workflow MCP Server")


def log_message(logs, message, log_file="agentic-workflow-mcp/logs.txt"):
    """
    Logs a message with a timestamp to the logs list and writes it to a log file.
    """
    timestamped_message = f"{datetime.now().isoformat()} - {message}"
    logs.append(timestamped_message)
    with open(log_file, "a") as file:
        file.write(timestamped_message + "\n")


async def get_agents(config):
    """
    Extracts and returns a list of available agents' names and descriptions 
    from the configuration, excluding 'aggregator' and 'orchestrator' agents.
    """
    excluded_agents = {"aggregator", "orchestrator"}
    return [
        f"{agent['name']}: {agent['description']}"
        for agent in config.get("agents", [])
        if agent["name"] not in excluded_agents
    ]


async def get_config(file_path):
    """
    Reads and returns the configuration from the specified JSON file.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Error reading config file {file_path}: {e}")


async def invoke_agent_generic(agent, config, prompt_data, logs):
    """
    Generic function to invoke an agent (orchestrator, regular agent, or aggregator).
    Sends a request to the agent's endpoint and returns the response.
    """
    try:
        # Prepare agent prompt
        if "prompt" in agent:
            agent_prompt = agent["prompt"].format(**prompt_data)
        else:
            with open(agent["prompt_file"], "r") as file:
                prompt_template = file.read()
                agent_prompt = prompt_template.format(**prompt_data)

        payload = {
            "model": agent.get("model", config["default_model"]),
            "prompt": agent_prompt,
            "stream": config.get("stream", False),
            "format": agent.get("output_format", config["default_output_format"])
        }

        # Set a longer read timeout (e.g., 30 seconds)
        timeout = httpx.Timeout(timeout=10.0, read=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{config['url']}/api/generate", json=payload)
            response.raise_for_status()
            agent_response = response.json().get("response", "")

        log_message(logs, f"{agent['name']} response: {agent_response}")
        return agent_response

    except httpx.RequestError as e:
        log_message(logs, f"HTTP request error while invoking {agent['name']}: {e} - {repr(e)}")
        return f"Error: {e}"
    except Exception as e:
        log_message(logs, f"Unexpected error while invoking {agent['name']}: {e}")
        return f"Error: {e}"


async def invoke_orchestrator(orchestrator, config, user_prompt, agents_list, logs):
    """
    Invokes the orchestrator agent with the user prompt and list of agents.
    """
    prompt_data = {
        "agents_list": ", ".join(agents_list),
        "user_prompt": user_prompt
    }
    return json.loads(await invoke_agent_generic(orchestrator, config, prompt_data, logs))


async def invoke_agent(agent_name, config, user_prompt, logs):
    """
    Invokes a single agent with the given user prompt.
    """
    agent = next((agent for agent in config["agents"] if agent["name"] == agent_name), None)
    if not agent:
        log_message(logs, f"Agent {agent_name} not found in configuration.")
        return {agent_name: f"Error: Agent not found"}

    prompt_data = {"user_prompt": user_prompt}
    response = await invoke_agent_generic(agent, config, prompt_data, logs)
    return {agent_name: response}


async def invoke_aggregator(aggregator, config, combined_responses, logs):
    """
    Invokes the aggregator agent with the combined responses from all agents.
    """
    prompt_data = {"combined_responses": combined_responses}
    return await invoke_agent_generic(aggregator, config, prompt_data, logs)


@mcp.tool()
async def start_workflow(user_prompt) -> dict:
    """
    Processes the workflow based on user_prompt:
    - Retrieves the list of agents from the config file.
    - Identifies the orchestrator and aggregator agents.
    - Invokes the orchestrator with the user prompt and agents list.
    - Invoke all agents from the orchestrator's response in parallel.
    - Collects the results from each agent.
    - Aggregates the results using the aggregator agent.
    - Returns the final result.
    """
    logs = []
    result = {}
    log_message(logs, f"Received user prompt: {user_prompt}")

    try:
        # Load configuration and extract agents
        config = await get_config("agentic-workflow-mcp/config.json")
        agents_list = await get_agents(config)

        # Identify orchestrator and aggregator agents
        orchestrator = next((agent for agent in config["agents"] if agent["name"] == "orchestrator"), None)
        aggregator = next((agent for agent in config["agents"] if agent["name"] == "aggregator"), None)

        if not orchestrator:
            raise ValueError("Orchestrator agent not found in configuration.")
        if not aggregator:
            raise ValueError("Aggregator agent not found in configuration.")

        orchestrator_response = await invoke_orchestrator(orchestrator, config, user_prompt, agents_list, logs)

        # Get the list of agents to invoke from the orchestrator's response
        agents_to_invoke = orchestrator_response.get("agents", [])

        # Invoke each agent in parallel

        # Collect responses from all agents
        agent_responses = await asyncio.gather(
            *(invoke_agent(agent, config, user_prompt, logs) for agent in agents_to_invoke)
        )

        # Combine all agent responses into a single dictionary
        combined_responses = {k: v for response in agent_responses for k, v in response.items()}
        log_message(logs, f"Combined agent responses: {combined_responses}")

        aggregator_response = await invoke_aggregator(aggregator, config, combined_responses, logs)

        result = {
            "response": aggregator_response
        }

    except (ValueError, KeyError) as e:
        log_message(logs, f"Configuration error: {e}")
    except httpx.RequestError as e:
        log_message(logs, f"HTTP request error: {e}")
    except Exception as e:
        log_message(logs, f"Unexpected error: {e}")
    finally:
        if config.get("verbose", False):
            result["logs"] = logs
        return result
