import asyncio
from mcp.server.fastmcp import FastMCP
import httpx
import json
from datetime import datetime

mcp = FastMCP("Local Agentic Workflow MCP Server")
excluded_agents = []


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
        excluded_agents.extend(config.get("excluded_agents", []))
        log_message(logs, f"Excluded agents: {excluded_agents}")
        agents_list = await get_agents(config)

        # Identify orchestrator and aggregator agents
        orchestrator = next((agent for agent in config["agents"] if agent["name"] == "orchestrator"), None)
        aggregator = next((agent for agent in config["agents"] if agent["name"] == "aggregator"), None)

        if not orchestrator:
            raise ValueError("Orchestrator agent not found in configuration.")

        # Check config for parallel execution (default to True if not specified)
        is_parallel = config.get("parallel", True)

        if is_parallel:
            # --- Parallel Flow ---
            log_message(logs, "Starting parallel workflow execution.")
            if not aggregator:
                raise ValueError("Aggregator agent is required for parallel execution.")

            # Invoke orchestrator to decide which agents to run
            orchestrator_response_raw = await invoke_orchestrator(orchestrator, config, user_prompt, agents_list, logs)

            # Parse orchestrator response (expecting JSON like {"agents": ["agent1", "agent2"]})
            try:
                # invoke_orchestrator already tries json.loads, but let's be safe
                if isinstance(orchestrator_response_raw, str):
                    orchestrator_response_data = json.loads(orchestrator_response_raw)
                elif isinstance(orchestrator_response_raw, dict):
                    orchestrator_response_data = orchestrator_response_raw
                else:
                    raise ValueError(f"Unexpected orchestrator response type: {type(orchestrator_response_raw)}")
            except json.JSONDecodeError as e:
                log_message(logs, f"Orchestrator response is not valid JSON: {orchestrator_response_raw} - Error: {e}")
                raise ValueError(f"Orchestrator response could not be parsed as JSON: {e}")
            except ValueError as e:
                log_message(logs, f"Error processing orchestrator response: {e}")
                raise e

            agents_to_invoke = orchestrator_response_data.get("agents", [])
            log_message(logs, f"Orchestrator selected agents: {agents_to_invoke}")

            if not agents_to_invoke:
                log_message(logs, "Orchestrator did not select any agents to invoke.")
                # Return the raw orchestrator response or a specific message
                aggregator_response = f"Orchestrator did not select any agents. Raw response: {orchestrator_response_raw}"
            else:
                # Invoke selected agents in parallel
                agent_tasks = [invoke_agent(agent_name, config, user_prompt, logs) for agent_name in agents_to_invoke]
                agent_responses = await asyncio.gather(*agent_tasks)

            # Combine responses for the aggregator
            combined_responses_dict = {k: v for response in agent_responses for k, v in response.items()}
            # Format combined_responses for the aggregator prompt (e.g., as a JSON string)
            combined_responses_str = json.dumps(combined_responses_dict)
            log_message(logs, f"Combined agent responses: {combined_responses_str}")

            # Invoke aggregator
            aggregator_response = await invoke_aggregator(aggregator, config, combined_responses_str, logs)

        else:
            # --- Sequential Flow ---
            log_message(logs, "Starting sequential workflow execution.")
            # Define the sequence: orchestrator first, then others (excluding aggregator)
            # Maintain order from config for 'other_agents'
            other_agents = [
                agent for agent in config.get("agents", []) if agent["name"] not in excluded_agents
            ]
            agent_sequence = [orchestrator] + other_agents
            log_message(logs, f"Sequential agent execution order: {[agent['name'] for agent in agent_sequence]}")

            current_input_data = user_prompt # Initial input for the first agent
            final_response = "No agents executed in sequence." # Default if sequence is empty

            for i, agent in enumerate(agent_sequence):
                log_message(logs, f"Invoking agent {i+1}/{len(agent_sequence)}: {agent['name']}")

                prompt_data = {}
                file_extractor_response = {}
                input_files = []
                input_data = {}
                output_file = None
                agents_to_use = agent.get("agents_to_use", [])
                if agents_to_use:
                    # Filter agents based on the orchestrator's response
                    agents_to_use = [agent for agent in agents_to_use if agent not in excluded_agents]
                    log_message(logs, f"Agents to use for {agent['name']}: {agents_to_use}")   
                    # Invoke the agents_to_use if specified
                    for agent_name in agents_to_use:
                        if agent_name == "file_extractor":
                            # Special case for file_extractor, which needs a different prompt
                            prompt_data = {"user_prompt": user_prompt}
                            file_extractor_response = await invoke_agent(agent_name, config, prompt_data, logs)
                            log_message(logs, f"File extractor response: {file_extractor_response}")
                            # Check if there are any input_files to process from the file_extractor response
                            if isinstance(file_extractor_response, dict) and "input_files" in file_extractor_response:
                                input_files = file_extractor_response.get("input_files", [])
                                output_file = file_extractor_response.get("output_file", None)
                                log_message(logs, f"Input files extracted: {input_files}")
                                # Check if base_path is provided in the config
                                if not "base_path" in config:
                                    raise ValueError("Base path not found in configuration for file extraction.")
                                # Ensure input_files is a list
                                if isinstance(input_files, str):
                                    input_files = [input_files]
                                # Prepare absolute paths for input files combining config["base_path"] and input_files
                                input_files = [f"{config['base_path']}/{file}" for file in input_files]
                                log_message(logs, f"Input files with base path: {input_files}")
                                # Read all the input files and prepare the prompt data
                                for input_file in input_files:
                                    try:
                                        with open(input_file, "r") as file:
                                            file_content = file.read()
                                            input_data[input_file] = file_content
                                            log_message(logs, f"Read input file: {input_file}")
                                    except FileNotFoundError:
                                        log_message(logs, f"Input file not found: {input_file}")
                                        continue
                                    except Exception as e:
                                        log_message(logs, f"Error reading input file {input_file}: {e}")
                                        continue
                                # Prepare the prompt data for the current agent
                                prompt_data = {
                                    **prompt_data,
                                    "user_prompt": user_prompt,
                                    "input_data": input_data
                                }
                            else:
                                log_message(logs, "No input files found in file_extractor response.")
                                input_files = []

                # Prepare prompt data based on position in sequence
                # First agent gets user_prompt, subsequent agents get user_prompt and previous_response
                if i == 0 or not agent["pass_to_next"]: # First agent (orchestrator)
                    # In sequential mode, orchestrator just processes the prompt, doesn't need agents_list
                    prompt_data = {
                        **prompt_data,
                        "user_prompt": user_prompt
                        }
                elif agent["pass_to_next"]:
                    # Subsequent agents receive the original user_prompt and the previous agent's response if pass_to_next is True
                    prompt_data = {
                        **prompt_data,
                        "user_prompt": user_prompt,
                        "previous_response": current_input_data
                        }

                # Invoke the current agent
                # Ensure prompt templates are designed to handle the keys in prompt_data
                current_response = await invoke_agent_generic(agent, config, prompt_data, logs)
                
                # Check for errors from invoke_agent_generic
                if isinstance(current_response, str) and current_response.startswith("Error:"):
                    log_message(logs, f"Error invoking agent {agent['name']}. Stopping sequence.")
                    final_response = current_response # Propagate the error
                    break # Stop the sequence on error

                if output_file:
                    # If output_file is specified, save the response to the file
                    try:
                        with open(output_file, "w") as file:
                            file.write(current_response)
                            log_message(logs, f"Response saved to {output_file}")
                    except Exception as e:
                        log_message(logs, f"Error saving response to {output_file}: {e}")

            current_input_data = current_response # Output of current becomes input for next step's previous_response
            final_response = current_response # Keep track of the last successful response

            # The final result is the response from the last agent in the sequence
            aggregator_response = final_response # Use the same variable name for consistency


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
