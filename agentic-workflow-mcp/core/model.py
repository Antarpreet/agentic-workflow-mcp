from dataclasses import dataclass
from typing_extensions import TypedDict, Optional, List, Dict

from chromadb import Client
from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.schema.vectorstore import VectorStore
from mcp.server.fastmcp import FastMCP
import os
import sys

# Determine the log file path based on the platform
if sys.platform.startswith("win"):
    log_file_path = os.path.expandvars(r"%USERPROFILE%\agentic-workflow-mcp\logs.txt")
else:
    log_file_path = os.path.expanduser("~/agentic-workflow-mcp/logs.txt")

DEFAULT_WORKFLOW_CONFIG = {
    "embedding_model": "nomic-embed-text",
    "default_model": "llama3.2:3b",
    "default_temperature": 0.0,
    "recursion_limit": 25,
    "collection_name": "langchain_chroma_collection",
    "default_orchestrator_completion_condition": "lambda state: state.get('final_output') is not None",
    "delete_missing_embeddings": True,
    "log_file_path": log_file_path,
    "state_schema": {
        "type": "object",
        "properties": {
            "user_input": {
                "type": "string"
            },
            "input": {
                "type": "string"
            },
            "final_output": {
                "type": "string"
            }
        },
        "required": [
            "user_input",
            "input",
            "final_output"
        ]
    },
    "rag_prompt_template": """Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}""",
    "vector_directory": "chroma_vector_db"
}


class DefaultWorkflowState(TypedDict):
    """Default state for the workflow."""
    input: str               # Holds the current input/data being passed between nodes
    final_output: str        # Accumulates or holds the final result for the user


class ToolFunctionConfig(TypedDict):
    """Interface for tool function configuration."""
    description: str
    function_string: str


class AgentConfig(TypedDict):
    """Interface for agent configuration."""
    name: str
    model_name: Optional[str]
    temperature: Optional[float]
    prompt: Optional[str]
    prompt_file: Optional[str]
    output_decision_keys: Optional[List[str]]
    output_format: Optional[dict]
    tools: Optional[List[str]]
    tool_functions: Optional[List[ToolFunctionConfig]]


class EdgeConfig(TypedDict):
    """Interface for edge configuration."""
    source: str
    target: str


class RouterConfig(TypedDict):
    """Interface for router configuration."""
    source: str
    router_function: str


class ParallelConfig(TypedDict):
    """Interface for parallel workflow configuration."""
    source: str
    nodes: List[str]
    join: str


class OrchestratorConfig(TypedDict):
    """Interface for orchestrator configuration."""
    name: str
    model_name: Optional[str]
    temperature: Optional[float]
    aggregator: str
    prompt: Optional[str]
    prompt_file: Optional[str]
    output_decision_keys: Optional[List[str]]
    output_format: Optional[dict]
    tools: Optional[List[str]]
    tool_functions: Optional[List[ToolFunctionConfig]]
    workers: List[str]
    supervise_workers: Optional[bool]
    can_end_workflow: Optional[bool]
    completion_condition: str


class EvaluatorOptimizerConfig(TypedDict):
    """Interface for evaluator-optimizer configuration."""
    executor: str
    evaluator: str
    optimizer: str
    quality_condition: str
    max_iterations: Optional[int]


class BranchConfig(TypedDict):
    """Interface for branch configuration."""
    source: str
    condition: str
    targets: Dict[str, str]


class WorkflowConfig(TypedDict):
    """Interface for workflow configuration."""
    default_model: str
    default_temperature: Optional[float]
    embedding_model: Optional[str]
    collection_name: Optional[str]
    delete_missing_embeddings: Optional[bool]
    state_schema: dict
    agents: List[AgentConfig]
    edges: List[EdgeConfig]
    routers: List[RouterConfig]
    parallel: List[ParallelConfig]
    orchestrator: OrchestratorConfig
    evaluator_optimizer: EvaluatorOptimizerConfig
    branches: List[BranchConfig]


@dataclass
class AppContext:
    """Holds application-wide resources."""
    embedding_model: Embeddings
    chroma_client: Client
    vectorstore: VectorStore
    llm: BaseLLM
    retriever: BaseRetriever
    retrieval_chain: Chain
    workflow_config: WorkflowConfig
    server: Optional[FastMCP] = None
