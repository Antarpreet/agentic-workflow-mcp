# In-Depth Guide: Example Workflow Configurations

This guide provides a detailed explanation of each example configuration in the `config_examples` folder. It covers the workflow logic, agent roles, configuration structure, and practical use cases. Use this as a reference to understand, adapt, or extend these workflows for your own projects.

---

## Table of Contents

- [Sequential Workflow (`sequential_config.json`)](#sequential-workflow-sequential_configjson)
- [Parallel Workflow (`parallel_config.json`)](#parallel-workflow-parallel_configjson)
- [Branching Workflow (`branches_config.json`)](#branching-workflow-branches_configjson)
- [Routers Workflow (`routers_config.json`)](#routers-workflow-routers_configjson)
- [Orchestrator-Worker Workflow (`orchestrator_worker_config.json`)](#orchestrator-worker-workflow-orchestrator_worker_configjson)
- [Embedding Retrieval Workflow (`embedding_retrieval_config.json`)](#embedding-retrieval-workflow-embedding_retrieval_configjson)
- [Embedding Updater Workflow (`embedding_updater_config.json`)](#embedding-updater-workflow-embedding_updater_configjson)
- [Evaluator-Optimizer Workflow (`evaluator_optimizer_config.json`)](#evaluator-optimizer-workflow-evaluator_optimizer_configjson)
- [Explicit Edges Workflow (`explicit_edges_config.json`)](#explicit-edges-workflow-explicit_edges_configjson)
- [Complex Combined Workflow (`complex_combined_config.json`)](#complex-combined-workflow-complex_combined_configjson)
- [Complex Evaluator-Optimizer Workflow (`complex_evaluator_optimizer_config.json`)](#complex-evaluator-optimizer-workflow-complex_evaluator_optimizer_configjson)

---

## Sequential Workflow (`sequential_config.json`)

**Purpose:**

A simple linear workflow where agents process information in a strict sequence. Each agent adds or transforms information before passing it to the next.

**How it works:**

- The workflow starts with `CountryAgent`, which identifies the country from user input.
- The output is passed to `LanguageAgent`, which adds language information.
- Next, `FlagAgent` adds a description of the country's flag.
- Finally, `FinalAgent` aggregates all information and produces a concise summary.

**Agent Roles:**

- `CountryAgent`: Extracts country name from input.
- `LanguageAgent`: Adds language details.
- `FlagAgent`: Adds flag description (UTF-8 only).
- `FinalAgent`: Summarizes all collected information.

**Use Case:**

- Step-by-step enrichment of data, e.g., building a country profile.

**How to Adapt:**

- Add more agents for additional enrichment steps.
- Change prompts to suit different domains (e.g., city, company, etc.).

---

## Parallel Workflow (`parallel_config.json`)

**Purpose:**

Run multiple agents in parallel from a single source, then join their results. Useful for tasks that can be split and processed independently.

**How it works:**

- `SplitAgent` divides the agenda into two parts: 'Project Updates' and 'Action Items'.
- `ParallelTask1` and `ParallelTask2` run in parallel, each handling one part.
- `JoinAgent` combines the outputs into a single meeting agenda.

**Agent Roles:**

- `SplitAgent`: Orchestrates the split.
- `ParallelTask1`: Summarizes project updates.
- `ParallelTask2`: Lists action items.
- `JoinAgent`: Merges both outputs.

**Special Feature:**

- The `parallel` section defines which agents run in parallel and how their results are joined.

**Use Case:**

- Multi-perspective analysis, ensemble tasks, or any scenario where independent subtasks can be processed simultaneously.

**How to Adapt:**

- Add more parallel tasks or change the join logic.

---

## Branching Workflow (`branches_config.json`)

**Purpose:**

Conditional branching based on workflow state. The workflow can follow different paths depending on agent output.

**How it works:**

- `InputClassifier` decides if the input is for 'support' or 'sales'.
- The `branches` section uses a lambda to route to either `SupportAgent` or `SalesAgent` based on the `class` key in the state.
- Each branch ends the workflow after its agent completes.

**Agent Roles:**

- `InputClassifier`: Classifies the request.
- `SupportAgent`: Handles support queries.
- `SalesAgent`: Handles sales queries.

**Special Feature:**

- The `branches` section with a condition lambda and target mapping.

**Use Case:**

- Classification, decision trees, or any workflow requiring conditional logic.

**How to Adapt:**

- Add more branches or change the classification logic.

---

## Routers Workflow (`routers_config.json`)

**Purpose:**

Dynamic routing to agents based on a function, allowing for flexible, context-aware agent selection.

**How it works:**

- `RouterAgent` inspects the request and sets `next_step` to either `StepOneAgent` or `StepTwoAgent`.
- The `routers` section uses a lambda to select the next agent based on `next_step`.
- Each step agent completes its task and ends the workflow.

**Agent Roles:**

- `RouterAgent`: Decides the next step.
- `StepOneAgent`: Handles password resets.
- `StepTwoAgent`: Handles email updates.

**Special Feature:**

- The `routers` section with a router function for dynamic agent selection.

**Use Case:**

- Smart dispatch, context-aware workflows, or multi-step forms.

**How to Adapt:**

- Add more step agents or expand the router logic.

---

## Orchestrator-Worker Workflow (`orchestrator_worker_config.json`)

**Purpose:**

An orchestrator agent manages multiple worker agents, delegating tasks and aggregating results.

**How it works:**

- `OrchestratorAgent` decides which worker(s) to call based on the task.
- `WorkerAgent1` reads files; `WorkerAgent2` writes files.
- `AggregatorAgent` combines worker outputs.
- The orchestrator supervises workers and can end the workflow when a completion condition is met.

**Agent Roles:**

- `OrchestratorAgent`: Manages workflow and delegates tasks.
- `WorkerAgent1`: Reads files.
- `WorkerAgent2`: Writes files.
- `AggregatorAgent`: Aggregates results.

**Special Feature:**

- The `orchestrator` section with workers, aggregator, and completion logic.

**Use Case:**

- Task delegation, coordination, or any scenario requiring supervision of multiple agents.

**How to Adapt:**

- Add more workers or change the aggregation logic.

---

## Embedding Retrieval Workflow (`embedding_retrieval_config.json`)

**Purpose:**

Retrieve vector embeddings for Retrieval-Augmented Generation (RAG) using a local knowledge base.

**How it works:**

- `RetrieverAgent` uses the `retrieve_embeddings` tool to fetch relevant information.
- `Summarizer` condenses the retrieved information.
- `Emailer` drafts an email with the summary.

**Agent Roles:**

- `RetrieverAgent`: Fetches embeddings.
- `Summarizer`: Summarizes information.
- `Emailer`: Prepares communication.

**Special Feature:**

- Integration with local vector database for semantic search.

**Use Case:**

- Knowledge retrieval, semantic search, or automated reporting.

**How to Adapt:**

- Change tools or add more processing steps.

---

## Embedding Updater Workflow (`embedding_updater_config.json`)

**Purpose:**

Update vector embeddings for files, keeping the knowledge base in sync with file changes.

**How it works:**

- `ModifyAgent` uses the `modify_embeddings` tool to update embeddings in the specified collection.
- The workflow is linear: start → modify → end.

**Agent Roles:**

- `ModifyAgent`: Updates embeddings.

**Special Feature:**

- Direct manipulation of the vector database.

**Use Case:**

- Keeping embeddings current after file edits or additions.

**How to Adapt:**

- Change the collection or add validation steps.

---

## Evaluator-Optimizer Workflow (`evaluator_optimizer_config.json`)

**Purpose:**

Add evaluation and optimization steps to ensure output quality, with iterative improvement if needed.

**How it works:**

- `ExecutorAgent` performs the main task (e.g., translation).
- `EvaluatorAgent` rates the output quality and sets a threshold.
- If quality is below threshold, `OptimizerAgent` suggests improvements.
- The `evaluator_optimizer` section defines the evaluation loop and stopping condition.

**Agent Roles:**

- `ExecutorAgent`: Main task performer.
- `EvaluatorAgent`: Evaluates quality.
- `OptimizerAgent`: Improves output if needed.

**Special Feature:**

- Iterative loop for quality control, with a maximum number of iterations.

**Use Case:**

- Translation, summarization, or any task requiring quality assurance.

**How to Adapt:**

- Change the evaluation criteria or add more optimization steps.

---

## Explicit Edges Workflow (`explicit_edges_config.json`)

**Purpose:**

Manually define all agent connections for full control over the workflow graph.

**How it works:**

- Each agent is connected via explicit `edges`.
- The workflow is strictly sequential: Researcher → Summarizer → Emailer.

**Agent Roles:**

- `Researcher`: Gathers information.
- `Summarizer`: Condenses information.
- `Emailer`: Communicates results.

**Special Feature:**

- No advanced logic; all transitions are explicit.

**Use Case:**

- Simple, predictable workflows where control is paramount.

**How to Adapt:**

- Add or rearrange edges to change the workflow.

---

## Complex Combined Workflow (`complex_combined_config.json`)

**Purpose:**

Combine multiple workflow patterns (sequential, parallel, branching, evaluation) for advanced, real-world scenarios.

**How it works:**

- `ProcessClassifier` decides if the task should be processed in parallel or sequentially.
- For parallel: `SplitAgent` divides the input, `ParallelTask1` and `ParallelTask2` process parts, `JoinAgent` combines results.
- For sequential: `ExecutorAgent` processes the whole input.
- `EvaluatorAgent` rates the output; if below threshold, `OptimizerAgent` improves it.
- The `evaluator_optimizer`, `parallel`, and `branches` sections orchestrate the complex logic.

**Agent Roles:**

- `ProcessClassifier`: Chooses workflow path.
- `SplitAgent`, `ParallelTask1`, `ParallelTask2`, `JoinAgent`: Handle parallel processing.
- `ExecutorAgent`: Handles sequential processing.
- `EvaluatorAgent`: Evaluates quality.
- `OptimizerAgent`: Improves output.

**Special Feature:**

- Integration of branching, parallelism, and evaluation/optimization in one config.

**Use Case:**

- Advanced document processing, multi-step reasoning, or any scenario requiring flexible, robust workflows.

**How to Adapt:**

- Add more branches, parallel tasks, or evaluation criteria as needed.

---

## Complex Evaluator-Optimizer Workflow (`complex_evaluator_optimizer_config.json`)

**Purpose:**

Add evaluation and optimization steps to ensure output quality, with iterative improvement if needed in a complex workflow with defined edges.

**How it works:**

- `IdeaGeneratorAgent` creates a detailed, complex idea, process, or product that requires multiple rounds of improvement.
- `InitialImproverAgent` takes the original idea and provides the first round of improvements.
- `SecondaryImproverAgent` further enhances the improved idea.
- `TertiaryImproverAgent` makes additional refinements to the idea or product.
- `QualityEvaluatorAgent` evaluates the current version and assigns a quality score from 0 (poor) to 1 (excellent).
- If the quality score is below 1, `ImprovementSuggesterAgent` proposes concrete ways to enhance the idea.
- `FinalizationAgent` produces the final improved version after all necessary iterations.
- `ConfirmationAgent` confirms and presents the finalized, improved idea, process, or product.

**Agent Roles:**

- `IdeaGeneratorAgent`: Generates a detailed, complex idea, process, or product that requires multiple rounds of improvement.
- `InitialImproverAgent`: Provides the first round of improvements to the generated idea.
- `SecondaryImproverAgent`: Further enhances the idea with additional improvements.
- `TertiaryImproverAgent`: Makes further refinements to the improved idea or product.
- `QualityEvaluatorAgent`: Evaluates the current version and assigns a quality score from 0 (poor) to 1 (excellent).
- `ImprovementSuggesterAgent`: Suggests concrete ways to enhance the idea if the quality score is below 1.
- `FinalizationAgent`: Produces the final improved version after all necessary iterations.
- `ConfirmationAgent`: Confirms and presents the finalized, improved idea, process, or product.
**Special Feature:**

- Multi-stage, iterative improvement pipeline with explicit agents for each refinement phase and a quality evaluation loop.

**Use Case:**

- Complex idea generation and enhancement, product/process design, or any scenario requiring multiple rounds of structured improvement and quality assessment.

**How to Adapt:**

- Adjust the number or type of improver agents for more or fewer refinement stages.
- Modify the evaluation criteria or scoring thresholds in `QualityEvaluatorAgent`.
- Customize prompts for domain-specific improvement or evaluation needs.
- Add or remove agents (e.g., more improvement or validation steps) to fit your workflow.

---

**Tips for All Configs:**

- Visualize your workflow using the `display_graph` tool for clarity.
- Adapt agent prompts and logic to fit your domain.
- Use the guides in the `guides` folder for setup and troubleshooting.
