import hashlib
import os
from datetime import datetime
from typing import List, Optional, Union

from chromadb import ClientAPI
from langchain.schema.vectorstore import VectorStore
from mcp.server.fastmcp import Context

from tools.file_system import read_file
from core.log import log_message

def update_embeddings(
    file_paths: Union[str, List[str]], 
    ctx: Context, 
    collection_name: Optional[str] = None,
    delete_missing: bool = True
) -> dict:
    """
    Update embeddings for files by creating new ones and deleting outdated ones.
    
    Args:
        file_paths: Path to a file or list of file paths to embed
        ctx: The MCP context containing application resources
        collection_name: Optional name for the ChromaDB collection
        delete_missing: Whether to delete embeddings for files that no longer exist
    
    Returns:
        dict: Information about the update operation, including created and deleted embeddings
    """
    logs = []
    workspace_path = os.getenv('WORKSPACE_PATH')
    log_message(logs, f"Updating embeddings for files: {file_paths}")
    
    # Convert single file path to list for uniform processing
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Prepend workspace_path to any relative file paths
    if workspace_path:
        file_paths = [
            file_path if os.path.isabs(file_path) else os.path.join(workspace_path, file_path)
            for file_path in file_paths
        ]
    
    # Get resources from context
    vectorstore: VectorStore = ctx.request_context.lifespan_context.vectorstore
    chroma_client: ClientAPI = ctx.request_context.lifespan_context.chroma_client
    
    # Get the collection
    collection = chroma_client.get_collection(collection_name)
    
    # Process files to create or update embeddings
    documents = []
    document_ids = []
    metadata_list = []
    
    for file_path in file_paths:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                log_message(logs, f"File not found: {file_path}")
                continue
            
            # Read file content using the LangChain BaseTool's invoke method
            content = read_file.invoke({"file_path": file_path, "workspace_path": workspace_path})
            
            # Generate a unique ID for the document based on content and path
            doc_id = hashlib.md5((file_path + content).encode()).hexdigest()
            
            # Check if document already exists in the collection
            existing_docs = collection.get(
                where={"source": file_path},
                include=["metadatas", "documents", "embeddings"]
            )
            
            # Create metadata
            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "file_type": os.path.splitext(file_path)[1],
                "created_at": datetime.now().isoformat()
            }
            
            # If the document exists but content has changed, delete the old version
            if existing_docs["ids"] and any(existing_doc != content for existing_doc in existing_docs["documents"]):
                collection.delete(ids=existing_docs["ids"])
                log_message(logs, f"Deleted outdated embedding for file: {file_path}")
                
                # Add to list for creating new embedding
                documents.append(content)
                document_ids.append(doc_id)
                metadata_list.append(metadata)
                log_message(logs, f"Queued updated embedding for file: {file_path}")
            
            # If document doesn't exist, add it
            elif not existing_docs["ids"]:
                documents.append(content)
                document_ids.append(doc_id)
                metadata_list.append(metadata)
                log_message(logs, f"Queued new embedding for file: {file_path}")
            
            else:
                log_message(logs, f"File {file_path} already has up-to-date embedding, skipping")
            
        except Exception as e:
            log_message(logs, f"Error processing file {file_path}: {str(e)}")
    
    # Create new embeddings if needed
    created_count = 0
    if documents:
        vectorstore.add_texts(
            texts=documents,
            metadatas=metadata_list,
            ids=document_ids
        )
        created_count = len(documents)
        log_message(logs, f"Added/updated {created_count} documents in ChromaDB collection: {collection_name}")
    
    # Delete embeddings for files that don't exist anymore
    deleted_count = 0
    if delete_missing:
        try:
            all_docs = collection.get(include=["metadatas"])
            
            for i, metadata in enumerate(all_docs["metadatas"]):
                if "source" in metadata:
                    file_path = metadata["source"]
                    if not os.path.exists(file_path):
                        collection.delete(ids=[all_docs["ids"][i]])
                        deleted_count += 1
                        log_message(logs, f"Deleted embedding for missing file: {file_path}")
        except Exception as e:
            log_message(logs, f"Error while cleaning up deleted files: {str(e)}")
    
    return {
        "status": "success",
        "created_count": created_count,
        "deleted_count": deleted_count,
        "document_ids": document_ids,
        "logs": logs
    }
