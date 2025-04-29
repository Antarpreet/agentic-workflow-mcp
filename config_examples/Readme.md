# Workflow Config Examples

## Table of Contents

- [Sequential](#sequential)
- [Branching](#branching)
- [Evaluator-Optimizer](#evaluator-optimizer)
- [Explicit Edges](#explicit-edges)
- [Parallel](#parallel)
- [Routers](#routers)
- [Orchestrator-Worker](#orchestrator-worker)
- [Complex Combined Config](#complex-combined-config)
- [Local Embedding Retriever](#local-embedding-retriever)

## Sequential

![Sequential](./images/sequential_config.png)

Test prompt: `Use MCP tool to start workflow Provide information about Antarctica`

## Branching

![Branching](./images/branches_config.png)

Test prompt: `Use MCP tool to start workflow Provide information about Nvidia RTX 3090 sales`

## Evaluator-Optimizer

![Evaluator-Optimizer](./images/evaluator_optimizer_config.png)

Test prompt: `Use MCP tool to start workflow Translate Is this really a evaluator optimized workflow? to French`

## Explicit Edges

![Explicit Edges](./images/explicit_edges_config.png)

Test prompt: `Use MCP tool to start workflow to draft an email`

## Parallel

![Parallel](./images/parallel_config.png)

Test prompt: `Use MCP tool to start workflow Agenda is to build a CICD pipeline`

## Routers

![Routers](./images/routers_config.png)

Test prompt: `Use MCP tool to start workflow I would like to change my email`

## Orchestrator-Worker

![Orchestrator-Worker](./images/orchestrator_worker_config.png)

Test prompt: `Use MCP tool to start workflow to read the ".gitignore" file`

## Complex Combined Config

![Complex Combined Config](./images/complex_combined_config.png)

Test prompt:

```plaintext
Use MCP tool to start workflow Summarize the following news article: The "most intense global coral bleaching event ever" has so far struck 84 per cent of the world's reefs and is ongoing, the International Coral Reef Initiative (ICRI) — a global partnership between nations and non-governmental and international organizations focused on sustainable management of coral reefs — reported on Wednesday.

The new figure is far worse than previous events that hit 21 to 68 per cent of reefs.

But scientists say the reefs and the corals are not all dead yet and could still  bounce back if people take the right steps, including conservation and cutting greenhouse gas emissions.

Corals are small marine animals that live in colonies with colourful symbiotic algae that give them their rainbow hues and supply them with most of their food. But when the water gets too warm for too long, the algae release toxic compounds, and the corals expel them, leaving behind a white skeleton — causing "bleaching."
```

## Local Embedding Retriever

![Local Embedding Retriever](./images/embedding_retrieval_config.png)

Test prompt:

```plaintext
1. Use MCP tool to embed files #file:Readme.md

2. Use MCP tool to start workflow to create an email about Agentic Workflow Server if it has custom embedding support
```
