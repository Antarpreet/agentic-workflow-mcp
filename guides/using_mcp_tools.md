# Guide: Using MCP Tools

This guide will help you get started with the MCP server's built-in tools, enabling you to leverage local LLMs and vector databases for advanced agentic workflows.

## Table of Contents

- [Overview of MCP Tools](#overview-of-mcp-tools)
- [How to Use Each Tool](#how-to-use-each-tool)

---

## Overview of MCP Tools

The MCP server provides several tools that can be used directly from VS Code (with GitHub Copilot Agent mode) or as part of your agentic workflow configuration. These tools include:

| Tool | Example Use Cases |
| --- | --- |
| `display_graph` | Visualize workflow, debug structure, document agent flows |
| `start_workflow` | Run sequential, parallel, branching, router, orchestrator, RAG, or embedding update workflows |
| `embed_files` | Index docs/code, update embeddings after edits, add new files/folders, remove missing embeddings |
| `visualize_embeddings` | Explore/validate embedding space, find clusters, share visualizations |

## How to Use Each Tool

### 1. display_graph

Generates a `graph.png` file in your workspace, showing the structure of your workflow (agents and their connections).

**Prompt Example:**

```plaintext
Use MCP Tools to display the graph.
```

**Example Use Cases:**

- See the structure and flow of agents, routers, branches, and parallel nodes in your workflow before running it.
- Debug or document the workflow by generating a `graph.png` file that shows how data and control move between agents.
- Share the workflow structure with team members or include it in documentation.

### 2. start_workflow

Starts the agentic workflow using your current configuration and a user prompt.

**Prompt Example:**

```plaintext
Use MCP Tools to start a workflow to "Summarize the main points of Readme.md".
```

**Example Use Cases:**

- `Sequential Workflow`: Build up a country profile step by step (e.g., "Provide information about Antarctica").
- `Parallel Workflow`: Split a meeting agenda into project updates and action items, process them in parallel, and join the results.
- `Branching Workflow`: Route a customer query to either support or sales based on input classification.
- `Routers Workflow`: Dynamically select the next agent based on user intent (e.g., "I would like to change my email").
- `Orchestrator-Worker Workflow`: Distribute tasks among multiple agents and aggregate their results.
- `Complex Combined Workflow`: Summarize a news article using multiple agents and logic branches.
- `Embedding Retrieval Workflow`: Use RAG (Retrieval-Augmented Generation) to answer questions based on local document embeddings (e.g., "create an email about Agentic Workflow Server if it has custom embedding support").
- `Embedding Updater Workflow`: Update or refresh embeddings for a folder or set of files.

### 3. embed_files

Creates vector embeddings for one or more files and stores them in the local ChromaDB vector database.

**Prompt Example:**

```plaintext
Use MCP tool to embed files #file:Readme.md
```

**Example Use Cases:**

- Index documentation, code, or knowledge base files for semantic search and retrieval.
- Update embeddings after editing files to keep the vector database in sync.
- Add new files or folders to the embedding database for use in RAG workflows.
- Remove embeddings for files that no longer exist (if configured).

### 4. visualize_embeddings

Creates a visualization (2D/3D) of the embeddings in your local ChromaDB vector database. It displays data for the collection name from config file unless specified in the prompt.

**Prompt Example:**

```plaintext
Use MCP tool to visualize embeddings for collection name "langchain_chroma_collection"
```

**Example Use Cases:**

- Explore the semantic structure of your document or codebase embeddings.
- Identify clusters, outliers, or gaps in your knowledge base.
- Validate that similar documents are close together in embedding space.
- Share visualizations with stakeholders to demonstrate coverage or organization.
