import hashlib
import os
from datetime import datetime
from typing import List, Optional, Union

from chromadb import ClientAPI
from langchain.schema.vectorstore import VectorStore
from mcp.server.fastmcp import Context
from sklearn.decomposition import PCA
import plotly.express as px

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


def visualize(ctx: Context, collection_name="langchain_chroma_collection"):
    """
    Visualize embeddings from a ChromaDB collection using PCA and Plotly.

    Args:
        ctx: The MCP context containing application resources
        collection_name: Name of the ChromaDB collection to visualize.

    This function connects to the ChromaDB database using the provided context,
    retrieves embeddings, reduces their dimensionality using PCA, and displays
    an interactive 2D or 3D plot using Plotly.
    """
    # Connect to the ChromaDB database and load the collection
    chroma = ctx.request_context.lifespan_context.chroma_client
    collection = chroma.get_collection(collection_name)

    print("Collection name:", collection.name)
    print("Number of embeddings:", collection.count())

    # Retrieve the embeddings, document IDs, and metadatas from the collection
    result = collection.get(include=['embeddings', 'metadatas'])
    embeddings = result['embeddings']
    docs = collection.get(include=['documents'])['ids']
    metadatas = result['metadatas']

    # Use absolute path with filename as dots if available, otherwise fallback to id
    point_labels = []
    for i, metadata in enumerate(metadatas):
        if isinstance(metadata, dict) and "source" in metadata and "filename" in metadata:
            # Replace path separators with dots for the label
            abs_path = metadata["source"]
            label = abs_path.replace("\\", ".").replace("/", ".")
            point_labels.append(label)
        elif isinstance(metadata, dict) and "filename" in metadata:
            point_labels.append(metadata["filename"])
        else:
            point_labels.append(docs[i])

    import numpy as np
    embeddings = np.array(embeddings)

    print("Embeddings shape:", embeddings.shape)
    print("Collection columns:", result.keys())
    print("Document label example:", point_labels[0])

    n_samples, n_features = embeddings.shape

    print("Number of samples:", n_samples)
    print("Number of features:", n_features)

    if n_samples == 0 or n_features == 0:
        raise ValueError("No embeddings found in the collection.")

    if n_samples < 2:
        raise ValueError("At least 2 samples are required for PCA visualization.")

    n_components = min(3, n_samples, n_features)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    vis_dims = pca.fit_transform(embeddings)

    print("PCA components shape:", vis_dims.shape)

    if (vis_dims == 0).all():
        raise ValueError("PCA resulted in all-zero components. Not enough variance in the data to visualize.")

    explained_var = pca.explained_variance_ratio_ * 100  # as percentage

    labels = {}
    for i in range(n_components):
        labels[f"{'xyz'[i]}"] = f"PCA Component {i+1} ({explained_var[i]:.1f}% var)"

    import plotly.express as px
    if n_components == 3:
        fig = px.scatter_3d(
            x=vis_dims[:, 0],
            y=vis_dims[:, 1],
            z=vis_dims[:, 2],
            text=point_labels,
            labels=labels,
            title='3D PCA of Embeddings'
        )
    elif n_components == 2:
        fig = px.scatter(
            x=vis_dims[:, 0],
            y=vis_dims[:, 1],
            text=point_labels,
            labels=labels,
            title='2D PCA of Embeddings'
        )
    else:
        raise ValueError("Not enough samples or features for 2D or 3D visualization.")

    fig.show()

    return "Visualization complete. Check your browser for the interactive plot."
