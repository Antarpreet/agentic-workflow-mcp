# Local Agentic Workflow MCP Server

This MCP server allows you to run the Agentic Workflows using a local LLM server using Ollama CLI. The workflow is designed as follows:

1. `Orchestrator Agent`: This agent is the first to be called. It is responsible for orchestrating the workflow and calling the other agents as needed, depending on the user prompt.
2. `Other Agents`: The orchestrator agent calls these agents to perform specific tasks. The Orchestrator Agent calls these agents in parallel and waits for their responses before proceeding to the next step.
3. `Aggregator Agent`: This agent is called at last to aggregate the results from the other agents and return the final result to Github Copilot.

This has been tested for `VS Code`.

## Prerequisites

```plaintext
- Python 3.8 or higher
- pip (for installing Python packages)
- Ollama CLI (for local LLMs)
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Antarpreet/agentic-workflow-mcp.git
   ```

2. Start an LLM server using the ollama CLI. For example, to start the `deepseek-r1:14b` model, run:

    ```bash
    ollama run deepseek-r1:14b
    ```

    This will start the LLM server on `http://localhost:11434` by default.

3. Install the required Python packages:

   ```bash
    pip install -r requirements.txt
    ```

4. Add MCP Server to VS Code:

    - Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS).
    - Type MCP and select `MCP: Add MCP Server`.
    - Select `Command (stdio)` as the server type.
    - Enter the command to start the MCP server.

    ```bash
    python -m uv run mcp run agentic-workflow-mcp/server.py
    ```

    - Name the server: `Agentic Workflow`.
    - Select `User settings` to add the server to all workspaces or `Workspace settings` to add it to the current workspace only.
    - If you already have other workspaces open, restart VS Code for changes to take effect.
    - This will open the `settings.json` file with the new MCP server configuration, which should look like this:

    ```json
    "mcp": {
        "servers": {
            "Agentic Workflow": {
                "type": "stdio",
                "command": "python",
                "args": [
                    "-m",
                    "uv",
                    "run",
                    "mcp",
                    "run",
                    "agentic-workflow-mcp/server.py"
                ]
            }
        }
    }
    ```

5. Add Config for the MCP Server as follows:

    - Adjust the default config in `config.json` as needed. The config settings are detailed further below.
    - Copy the server folder to the user folder. (`C:\Users\<username>\agentic-workflow-mcp` on Windows or `~/agentic-workflow-mcp` on Linux/MacOS).

        Windows:

        ```cmd
        xcopy /E /I agentic-workflow-mcp %homedrive%%homepath%\agentic-workflow-mcp
        ```

        Mac/Linux:

        ```bash
        rm -rf ~/agentic-workflow-mcp
        cp -r agentic-workflow-mcp ~/agentic-workflow-mcp
        ```

    - Anytime you make any changes to these files, copy them to the user folder again and restart the MCP server in the `settings.json` file for the changes to take effect.

6. Start the MCP server:

    - Click the start button above the MCP server configuration in the `settings.json` file.
    - This will start the MCP server; you can see the logs in the Output panel under `MCP: Agentic Workflow` by clicking either the `Running` or `Error` button above the MCP server configuration.

7. Start using the MCP server:

    - Open GitHub Copilot in VS Code and switch to `Agent` mode.
    - You should see the `Agentic Workflow` MCP server and `start_workflow` tool in the Copilot tools panel.
    - You can now use the `start_workflow` tool to start the Agentic Workflow. Prompt example:

    ```plaintext
    Use MCP Tools to start a workflow to YOUR_PROMPT_HERE.
    ```

## Config Settings

### The config settings are defined as follows

| Key | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `default_model` | string | The default model to use for the LLM server. | `true` | `deepseek-r1:14b` |
| `url` | string | The URL of the Ollama LLM server. | `true` | `http://localhost:11434` |
| `verbose` | boolean | Whether to enable verbose logging. | `false` | `false` |
| `default_output_format` | object | The default output format for the LLM server. | `false` | `{"type": "object", "properties": {"response": {"type": "string"}}, "required": ["response"]}` |
| `agents` | object[] | The agents used in the workflow. | `false` | Agents |

### Agents are defined as follows

| Key | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `name` | string | The name of the agent. | `true` | `Orchestrator Agent` |
| `description` | string | The description of the agent. | `true` | `This agent is responsible for orchestrating the workflow and calling the other agents as needed, depending on the user prompt.` |
| `model` | string | The model to use for the agent. | `false` | `deepseek-r1:14b` |
| `prompt` | string | The prompt to use for the agent. This takes precedence over `prompt_file`. | `true` | `You are an agent that is responsible for orchestrating the workflow and calling the other agents as needed, depending on the user prompt. You will call the other agents in parallel and wait for their responses before proceeding to the next step. You will also call the Aggregator Agent in the end to aggregate the results from the other agents and return the final result to Github Copilot.` |
| `prompt_file` | string | Either the absolute path to the prompt file or path to the prompt file in the format `agentic-workflow-mcp/YOUR_PROMPT_FILE_NAME` if the prompt file is added to the `agentic-workflow-mcp` in this repo. | `false` | `""` |
| `output_format` | object | The output format for the agent. | `false` | `{"type": "object", "properties": {"response": {"type": "string"}}, "required": ["response"]}` |

## Tools

### `start_workflow`

This tool is used to start the Agentic Workflow. It takes a prompt as input and returns the result of the workflow.

## Example usage

The example config contains agents for getting information regarding a country. There are three agents other than the `Orchestrator Agent` and `Aggregator Agent`: `Country`, `Flag`, and `Language`. Depending on your prompt, each of them will return their responses, and the combined responses are sent to the `Aggregator Agent`, which will return the final result.

There is also a `prompt.txt` file, which contains the prompt for the `Orchestrator Agent` as an example to show how to use a file instead of a string for the prompt. The `prompt.txt` file is used in the `prompt_file` key of the `Orchestrator Agent` in `config.json`.

Test Prompt: `Use MCP tools to start workflow to confirm whether France is real or not`

## Troubleshooting

- If the MCP server is not making any requests to the LLM server, do the following:

    1. Restart the MCP server in the `settings.json` file.
    2. Create a new chat in GitHub Copilot.
    3. Sometimes the Ollama server doesn't respond if not used for a while. In that case, open the ollama url (`http://localhost:11434`) in a browser. You should see the message `Ollama is running`. If you don't see this message, restart the Ollama server using the command `ollama run deepseek-r1:14b`.

- For seeing the internal logs, you can set the `verbose` key in the `config.json` file to `true`. This will include all the logs in the response. You can see these by expanding the `start_workflow` tool in the Copilot chat window.

## TODO

- Add support for multi-modality using local LLM servers.
