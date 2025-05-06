# Local Agentic Workflow MCP Server

This MCP server allows you to run the Agentic Workflows using a local LLM server using Ollama CLI. This has been tested for `VS Code`. The workflows are designed as follows:

For supported workflow example configurations, see the [Config Examples](config_examples/Readme.md) file.

## Table of Contents

- [Articles](#articles)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Config Settings](#config-settings)
- [Config Examples](config_examples/Readme.md)
- [Environment Variables](#environment-variables)
- [GitHub Copilot Tools](#github-copilot-tools)
- [Custom Embeddings for RAG](#custom-embeddings-for-rag)
- [Local LLM Tools](#local-llm-tools)
- [Troubleshooting](#troubleshooting)

## Articles

- [🧠 How to Set Up a Local Agentic Workflow with MCP and Ollama (Without Losing Your Mind)](https://medium.com/@antarpreetsingh/how-to-set-up-a-local-agentic-workflow-with-mcp-and-ollama-95864d30f462)
- [🧠Create Vector Embeddings for Your Local Agentic Workflow Using an MCP Server (The easy way)](https://medium.com/@antarpreetsingh/create-vector-embeddings-for-your-local-agentic-workflow-using-an-mcp-server-0e424e2cc6b7)

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

2. Start an LLM server using the ollama CLI. For example, to start the `llama3.2:3b` model, run:

    ```bash
    ollama run llama3.2:3b
    ```

    This will start the LLM server on `http://localhost:11434` by default.

    > If you are using tools in your workflow, please ensure the model you are using supports them: [Models supporting tools](https://ollama.com/search?c=tools)

    If you will be using local vector embeddings in your workflow, you can also start the `nomic-embed-text` model using the following command:

    ```bash
    ollama run nomic-embed-text
    ```

3. Install the required Python packages:

   ```bash
    pip install -r requirements.txt
    ```

4. Add MCP Server to VS Code:

    - Open `.vscode/mcp.json` in your workspace folder. If it doesn't exist, create it.
    - Update `PATH_TO_YOUR_CONFIG` in the `WORKFLOW_CONFIG_PATH` environment variable to point to the config file in your workspace folder.

     > - The default config uses `workspaceFolder` environment variable from `VS Code` to get the path of the workspace.
     > - If you would like to use `User Settings`, make sure to replace the environment variable with the absolute path of your workspace folder.
     > - You can open the User `settings.json` file directly by using the command `Preferences: Open User Settings (JSON)` in the Command Palette for updating `User Settings` and add the following config in a `mcp` object: `mcp: { "servers": ... }`.
     > - The path to the `config.json` file in the `WORKFLOW_CONFIG_PATH` environment variable `PATH_TO_YOUR_CONFIG` should point to the `config.json` file in your workspace folder. This allows you to use different configurations for different projects.

    ```json
    // .vscode/mcp.json in your workspace folder
    {
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
                ],
                "env": {
                    "WORKSPACE_PATH": "${workspaceFolder}",
                    "WORKFLOW_CONFIG_PATH": "${workspaceFolder}/PATH_TO_YOUR_CONFIG/config.json",
                }
            }
        }
    }
    ```

5. Add Config for the MCP Server as follows:

    - Use one of the default configurations from `config_examples` in your `config.json` file as needed. The config settings are detailed further below. There are example configurations in the `config_examples` folder. You can use them as a reference to create your own configuration.
    - Copy the server folder to the user folder. (`C:\Users\<username>\agentic-workflow-mcp` on Windows or `~/agentic-workflow-mcp` on Linux/MacOS). This will make it easier to access the server files across different projects. You can do this by running the following commands in your terminal:

        Windows:

        ```cmd
        xcopy /E /I agentic-workflow-mcp %homedrive%%homepath%\agentic-workflow-mcp
        ```

        Mac/Linux:

        ```bash
        rm -rf ~/agentic-workflow-mcp
        cp -r agentic-workflow-mcp ~/agentic-workflow-mcp
        ```

    > Anytime you make any changes to these files, copy them to the user folder again and restart the MCP server in the `.vscode/mcp.json` file for the changes to take effect.

6. Start the MCP server:

    - Click the `Start` button above the MCP server configuration in the `.vscode/mcp.json` file in your workspace folder.
    - This will start the MCP server; you can see the logs in the Output panel under `MCP: Agentic Workflow` by clicking either the `Running` or `Error` button above the MCP server configuration.

7. Start using the MCP server:

    - Open GitHub Copilot in VS Code and switch to `Agent` mode.
    - You should see the `Agentic Workflow` MCP server and `start_workflow` tool in the Copilot tools panel.
    - You can now start using the MCP tools. Prompt example:

    ```typescript
    // This will create a `graph.png` file in your workspace folder.
    // It's recommended to use this before running the workflow
    // to see the graph of the agents and their connections.
    Use MCP Tools to display the graph.
    // This will start the workflow.
    Use MCP Tools to start a workflow to YOUR_PROMPT_HERE.
    // This will create embeddings for the files passed in the prompt.
    Use MCP tool to embed files #file:Readme.md
    // This will create a 2D or 3D visualization of the embeddings.
    Use MCP tool to visualize embeddings.
    ```

## Config Settings

### Workflow

| Key | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `default_model` | string | The default model to use for the LLM server. | `true` | `llama3.2:3b` |
| `default_temperature` | number | The default temperature to use for the LLM server. | `false` | `0.0` |
| `embedding_model` | string | The embedding model to use for the LLM server. | `false` | `nomic-embed-text` |
| `collection_name` | string | The name of the ChromaDB vector database collection to use for the LLM server. | `false` | `langchain_chroma_collection` |
| `delete_missing_embeddings` | boolean | Whether to delete the embeddings for files that are no longer present in the workspace. | `false` | `true` |
| `vector_directory` | string | The directory to store the vector database. | `false` | `chroma_vector_db` |
| `rag_prompt_template` | string | The prompt template for the RAG agent. | `false` | `Answer the following question based only on the provided context: <context> {context} </context> Question: {input}` |
| `state_schema` | object | The schema for the workflow state. | `false` | `{"type": "object", "properties": {"input": {"type": "string"},"final_output": {"type": "string"}}, "required": ["input","final_output"]}` |
| `agents` | object[] | The agents used in the workflow. | `true` | [Agent](#agent) |
| `orchestrator` | object | The orchestrator agent configuration. | `false` | [Orchestrator](#orchestrator) |
| `evaluator_optimizer` | object | The evaluator configuration. | `false` | [Evaluator](#evaluator) |
| `edges` | object[] | The edges between the agents in the workflow. | `false` | [Edge](#edge) |
| `parallel` | object[] | The parallel agents configuration. | `false` | [Parallel](#parallel) |
| `branches` | object[] | The branches in the workflow. | `false` | [Branch](#branch) |
| `routers` | object[] | The routers in the workflow. | `false` | [Router](#router) |

### Agent

| Key | Type | Description | Required | Example |
| --- | --- | --- | --- | --- |
| `name` | string | The name of the agent. | `true` | `Orchestrator Agent` |
| `model_name` | string | The model to use for the agent. If different from the default model. | `false` | `llama3.2:3b` |
| `temperature` | number | The temperature to use for the agent. If different from the default temperature. | `false` | `0.0` |
| `prompt` | string | The prompt to use for the agent. This takes precedence over `prompt_file`. | `true` | `You are an agent that orchestrates the workflow.` |
| `prompt_file` | string | Either the absolute path to the prompt file or path to the prompt file in the format `agentic-workflow-mcp/YOUR_PROMPT_FILE_NAME` if the prompt file is added to the `agentic-workflow-mcp` in this repo. | `false` | `prompt.txt` |
| `output_decision_keys` | string[] | The keys in the output that will be used in the workflow state. | `false` | `["decision_key"]` |
| `output_format` | object | The output format for the agent. | `false` | `{"type": "object", "properties": {"response": {"type": "string"}}, "required": ["response"]}` |
| `tools` | string[] | The tools to use for the agent. | `false` | `["read_file"]` |
| `tool_functions` | object[] | The functions to use for the tools. | `false` | `{"read_file":` [Tool](#tool)`}` |
| `embeddings_collection_name` | string | The name of the ChromaDB vector database collection to use for the agent. | `false` | `langchain_chroma_collection` |

### Tool

| Key | Type | Description | Required | Example |
| --- | --- | --- | --- | --- |
| `description` | string | The description of the tool. | `true` | `Reads the contents of a file and returns it as a string.` |
| `function_string` | string | The function string to use for the tool. | `true` | `lambda filename, workspace_path=None: open(filename if workspace_path is None else f'{workspace_path}/{filename}', 'r', encoding='utf-8').read()` |

### Orchestrator

| Key | Type | Description | Required | Example |
| --- | --- | --- | --- | --- |
| `name` | string | The name of the orchestrator agent. | `true` | `OrchestratorAgent` |
| `model_name` | string | The model to use for the orchestrator agent. | `false` | `llama3.2:3b` |
| `temperature` | number | The temperature to use for the orchestrator agent. | `false` | `0.0` |
| `aggregator` | string | The name of the aggregator agent. | `true` | `AggregatorAgent` |
| `prompt` | string | The prompt to use for the orchestrator agent. | `true` | `You are an agent that orchestrates the workflow.` |
| `prompt_file` | string | The prompt file to use for the orchestrator agent. | `false` | `prompt.txt` |
| `output_decision_keys` | string[] | The keys in the output that will be used in the workflow state. | `false` | `["decision_key"]` |
| `output_format` | object | The output format for the orchestrator agent. | `false` | `{"type": "object", "properties": {"response": {"type": "string"}}, "required": ["response"]}` |
| `tools` | string[] | The tools to use for the orchestrator agent. | `false` | `["read_file"]` |
| `tool_functions` | object[] | The functions to use for the tools. | `false` | `{"read_file": TOOL}` |
| `workers` | string[] | The workers to use for the orchestrator agent. | `true` | `["Agent1", "Agent2"]` |
| `supervise_workers` | boolean | Whether to supervise the workers. | `false` | `false` |
| `can_end_workflow` | boolean | Whether the orchestrator can end the workflow. | `false` | `false` |
| `completion_condition` | string | The completion condition for the orchestrator agent. | `true` | `lambda state: state.get('final_output') is not None` |

### Evaluator

| Key | Type | Description | Required | Example |
| --- | --- | --- | --- | --- |
| `executor` | string | The name of the executor agent. | `true` | `ExecutorAgent` |
| `evaluator` | string | The name of the evaluator agent. | `true` | `EvaluatorAgent` |
| `optimizer` | string | The name of the optimizer agent. | `true` | `OptimizerAgent` |
| `quality_condition` | string | The quality condition for the evaluator agent. | `true` | `lambda state: state.get('quality_score', 0) >= state.get('quality_threshold', 0.8)` |
| `max_iterations` | integer | The maximum number of iterations for the evaluator agent. | `false` | `5` |

### Edge

| Key | Type | Description | Required | Example |
| --- | --- | --- | --- | --- |
| `source` | string | The source agent. The value can also be `__start__` representing start of the workflow. | `true` | `OrchestratorAgent` |
| `target` | string | The target agent. The value can also be `__end__` representing end of the workflow. | `true` | `AggregatorAgent` |

### Parallel

| Key | Type | Description | Required | Example |
| --- | --- | --- | --- | --- |
| `source` | string | The source agent that will call the parallel agents. | `true` | `SplitAgent` |
| `nodes` | string[] | The parallel agents. | `true` | `["Agent1", "Agent2"]` |
| `join` | string | The agent that will join the responses from parallel agents. | `true` | `JoinAgent` |

### Branch

| Key | Type | Description | Required | Example |
| --- | --- | --- | --- | --- |
| `source` | string | The source agent that will call the branch agents. | `true` | `InputClassifierAgent` |
| `condition` | string | The condition for the branch. | `true` | `lambda state: state.get('class')'` |
| `targets` | object | The target agents for the branch. | `true` | `{"class1": "Agent1", "class2": "Agent2"}` |

### Router

| Key | Type | Description | Required | Example |
| --- | --- | --- | --- | --- |
| `source` | string | The source agent that will call the router agents. | `true` | `RouterAgent` |
| `router_function` | string | The function to use for the router. | `true` | `lambda state: state.get('next_step')` |

## Environment Variables

These are the environment variables that are used in the MCP server. You can set them in the `.vscode/mcp.json` file as shown above.

| Key | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `WORKSPACE_PATH` | string | The workspace path to the files to read. | `true` | `${workspaceFolder}` |
| `WORKFLOW_CONFIG_PATH` | string | The path to the config file. | `true` | `${workspaceFolder}/PATH_TO_YOUR_CONFIG/config.json` |

## GitHub Copilot Tools

### `display_graph`

Generates a graph image from the workflow configuration and saves it to a `graph.png` file in the workspace folder. This is useful for visualizing the workflow and understanding the connections between agents.

### `start_workflow`

This tool is used to start the Agentic Workflow. It takes a prompt as input and returns the result of the workflow.

### `embed_files`

This tool creates embeddings for one or more files and stores them in the local ChromaDB vector database. These embeddings can be used using the `retrieve_embeddings` tool in the agent configuration.

### `visualize_embeddings`

Generates a 2D or 3D visualization of the embeddings in the local ChromaDB vector database. This is useful for understanding the distribution of the embeddings and identifying clusters or patterns in the data.

## Custom Embeddings for RAG

Custom Embeddings for your local files can be created using the `embed_files` tool.

This tool creates embeddings for one or more files and stores them in the local ChromaDB vector database.

The local `chroma_vector_db` vector database is created in the workspace folder. You can add it to your `.gitignore` file to avoid committing it to your repository.

If the absolute path to the file is not provided, the tool will look for the file in the workspace folder. The `workspace_path` variable is set to `${workspaceFolder}` by default, which is the path to your workspace folder.

The embeddings are automatically created, updated and deleted when invoking the `embed_files` tool. `delete_missing_embeddings` is set to `true` by default. This means that if a file is deleted from the workspace, its embedding will be deleted from the vector database next time the `embed_files` tool is invoked.

The local embeddings can be made available to any agent in the chain by using the `retrieve_embeddings` tool in the agent configuration. This tool will retrieve the embeddings from the local vector database and use them to answer questions.

You can visualize the embeddings using the `visualize_embeddings` tool. This will create a 2D or 3D visualization of the embeddings in the local ChromaDB vector database. This is useful for understanding the distribution of the embeddings and identifying clusters or patterns in the data.

## Local LLM Tools

These are local tools available to the local Ollama LLM server. You can use these tools in your workflow to perform various tasks. These tools will be invoked as part of the workflow so you don't have to worry about calling them separately. The tools are defined in the `config.json` file and can be used in the workflow by specifying the tool names in the agent config.

> You can add your own tools directly in the `config.json` file as described above.

### `read_file`

Reads the content of a file and returns it as a string.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `file_path` | string | The path to the file to read. | `true` | `""` |
| `workspace_path` | string | The workspace path to the file to read. | `false` | `${workspaceFolder}` |

### `read_multiple_files`

Reads the content of multiple files and returns them as a single string.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `file_paths` | string[] | The paths to the files to read. | `true` | `[]` |
| `workspace_path` | string | The workspace path to the files to read. | `false` | `${workspaceFolder}` |

### `list_files`

Lists all files in a given directory.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `directory` | string | The path to the directory to list files from. | `true` | `""` |
| `workspace_path` | string | The workspace path to the directory to list files from. | `false` | `${workspaceFolder}` |

### `write_file`

Writes the given content to a file. Creates directories if they don't exist.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `file_path` | string | The path to the file to write to. | `true` | `""` |
| `content` | string | The content to write to the file. | `true` | `""` |
| `workspace_path` | string | The workspace path to the file to write to. | `false` | `${workspaceFolder}` |

### `write_file_lines`

Write lines content at the specified line numbers to a file. Creates directories if they don't exist.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `file_path` | string | The path to the file to write to. | `true` | `""` |
| `lines` | object | The object containing line numbers as keys and content as values. | `true` | `{}` |
| `workspace_path` | string | The workspace path to the file to write to. | `false` | `${workspaceFolder}` |

### `append_file`

Appends the given content to a file. Creates directories if they don't exist.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `file_path` | string | The path to the file to append to. | `true` | `""` |
| `content` | string | The content to append to the file. | `true` | `""` |
| `workspace_path` | string | The workspace path to the file to append to. | `false` | `${workspaceFolder}` |

### `append_file_lines`

Appends lines content at the specified line numbers to a file. Creates directories if they don't exist.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `file_path` | string | The path to the file to append to. | `true` | `""` |
| `lines` | object | The object containing line numbers as keys and content as values. | `true` | `{}` |
| `workspace_path` | string | The workspace path to the file to append to. | `false` | `${workspaceFolder}` |

### `web_search`

Performs a web search using DuckDuckGo and returns the results.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `query` | string | The search query. | `true` | `""` |
| `max_results` | integer | The maximum number of results to return. | `false` | `5` |

### `api_fetch`

Fetch data from an API endpoint.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `url` | string | The API endpoint URL. | `true` | `""` |
| `method` | string | The HTTP method to use (GET, POST, etc.). | `false` | `GET` |
| `headers` | object | The headers to include in the request. | `false` | `{}` |
| `params` | object | The query parameters to include in the request. | `false` | `{}` |
| `data` | object | The data to include in the request body. | `false` | `{}` |
| `json` | object | The JSON data to include in the request body. | `false` | `{}` |
| `timeout` | integer | The timeout for the request in seconds. | `false` | `10` |

### `retrieve_embeddings`

Fetches the embeddings from the local vector database and uses them to answer questions.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `input` | string | The input to the agent. | `true` | `""` |

### `modify_embeddings`

Updates the embeddings for the specified files.

| Item | Type | Description | Required | Defaults |
| --- | --- | --- | --- | --- |
| `file_paths` | string[] | The paths to the files to update. | `true` | `[]` |

## Troubleshooting

- If the MCP server is not making any requests to the LLM server, do the following:

    1. Restart VS Code as a sanity check.
    2. Ensure the Ollama LLM server is running and accessible.
    3. Copy the server files again using the commands above, another sanity check.
    4. Restart the MCP server in the `.vscode/mcp.json` file.
    5. Create a new chat in GitHub Copilot and switch to `Agent` mode.

- You can check the logs using:

    Windows:

    ```cmd
    type %homedrive%%homepath%\agentic-workflow-mcp\logs.txt
    ```

    Mac/Linux:

    ```bash
    tail -f ~/agentic-workflow-mcp/logs.txt
     ```
