"""
MCP Memory Server - Recursive Memory System with Qdrant and Local Models
Brain-inspired memory clustering using Qdrant vector database and local embedding/LLM models
"""

import asyncio
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    UpdateStatus,
    ScoredPoint,
)
from aiohttp import web

from mcp.server.fastmcp import FastMCP

# Configuration
MAX_DEPTH = 100
SIMILARITY_THRESHOLD = 0.7
DECAY_FACTOR = 0.99
REINFORCEMENT_FACTOR = 1.1
MERGE_SIMILARITY_THRESHOLD = 0.85

# Local model configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "microsoft/phi-2")
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 dimension

# Initialize FastMCP
mcp = FastMCP(name="memory-server", host="0.0.0.0", port=9321)

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "memories"

# Global model instances
embedding_model = None
llm_model = None
llm_tokenizer = None
device = None


# ============================================================================
# Local Model Management
# ============================================================================


def initialize_models() -> None:
    """Initialize local embedding and LLM models"""
    global embedding_model, llm_model, llm_tokenizer, device

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Load embedding model
    print(f"📥 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_model.to(device)
    print(
        f"✅ Embedding model loaded (dimension: {embedding_model.get_sentence_embedding_dimension()})"
    )

    # Update embedding dimension
    global EMBEDDING_DIMENSION
    EMBEDDING_DIMENSION = embedding_model.get_sentence_embedding_dimension()

    # Load LLM for summaries
    print(f"📥 Loading LLM model: {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME, trust_remote_code=True
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    llm_model.to(device)
    llm_model.eval()
    print(f"✅ LLM model loaded")


def get_embedding(text: str) -> list[float]:
    """Generate embedding using local model"""
    if embedding_model is None:
        initialize_models()

    with torch.no_grad():
        embedding = embedding_model.encode(text, convert_to_tensor=False)

    return embedding.tolist()


def generate_summary(text: str) -> str:
    """Generate summary using local LLM"""
    if llm_model is None or llm_tokenizer is None:
        initialize_models()

    # Truncate long text
    max_input_length = 500
    if len(text) > max_input_length:
        text = text[:max_input_length] + "..."

    prompt = (
        f"Summarize the following text in one concise sentence:\n\n{text}\n\nSummary:"
    )

    try:
        inputs = llm_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id,
            )

        summary = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the summary part
        if "Summary:" in summary:
            summary = summary.split("Summary:")[-1].strip()

        # Take first sentence if multiple
        if "." in summary:
            summary = summary.split(".")[0] + "."

        return summary[:200]  # Limit length

    except Exception as e:
        print(f"⚠️  Summary generation failed: {e}")
        # Fallback to simple truncation
        return text[:100] + "..." if len(text) > 100 else text


def merge_texts(text1: str, text2: str) -> str:
    """Merge two texts using local LLM"""
    if llm_model is None or llm_tokenizer is None:
        initialize_models()

    # Truncate if too long
    max_length = 300
    if len(text1) > max_length:
        text1 = text1[:max_length] + "..."
    if len(text2) > max_length:
        text2 = text2[:max_length] + "..."

    prompt = f"""Combine these two related memories into one coherent memory. Keep all important information.

Memory 1: {text1}

Memory 2: {text2}

Combined memory:"""

    try:
        inputs = llm_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id,
            )

        result = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the combined memory part
        if "Combined memory:" in result:
            result = result.split("Combined memory:")[-1].strip()

        return result[:500]  # Limit length

    except Exception as e:
        print(f"⚠️  Text merging failed: {e}")
        # Fallback to simple concatenation
        return f"{text1} {text2}"


# ============================================================================
# Helper Functions
# ============================================================================


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a_array = np.array(a, dtype=np.float64)
    b_array = np.array(b, dtype=np.float64)
    return float(
        np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array))
    )


@dataclass
class Deps:
    """Dependencies container"""

    qdrant: AsyncQdrantClient


# ============================================================================
# Qdrant Client Management
# ============================================================================


async def init_qdrant_collection(client: AsyncQdrantClient) -> None:
    """Initialize Qdrant collection if it doesn't exist"""
    try:
        await client.get_collection(collection_name=COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' already exists")
    except Exception:
        print(f"📦 Creating collection '{COLLECTION_NAME}'...")
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION, distance=Distance.COSINE
            ),
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created")


async def get_qdrant_client() -> AsyncQdrantClient:
    """Get async Qdrant client and ensure collection exists"""
    client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    await init_qdrant_collection(client)
    return client


# ============================================================================
# Memory Node Model
# ============================================================================


class MemoryNode(BaseModel):
    """Represents a single memory node with clustering capabilities"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    summary: str = ""
    importance: float = 1.0
    access_count: int = 0
    timestamp: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )
    embedding: list[float]
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @classmethod
    async def from_content(
        cls, content: str, deps: Deps, tags: list[str] = None, metadata: dict = None
    ):
        """Create a new memory node from content"""
        # Generate embedding using local model
        embedding = get_embedding(content)

        # Generate summary using local LLM
        summary = generate_summary(content)

        return cls(
            content=content,
            embedding=embedding,
            summary=summary,
            tags=tags or [],
            metadata=metadata or {},
        )

    async def save(self, deps: Deps):
        """Save memory node to Qdrant"""
        point = PointStruct(
            id=self.id,
            vector=self.embedding,
            payload={
                "content": self.content,
                "summary": self.summary,
                "importance": self.importance,
                "access_count": self.access_count,
                "timestamp": self.timestamp,
                "tags": self.tags,
                "metadata": self.metadata,
            },
        )
        await deps.qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])

    async def merge_with(self, other: "MemoryNode", deps: Deps):
        """Merge this memory with another similar memory"""
        # Combine content using local LLM
        self.content = merge_texts(self.content, other.content)

        # Update importance and access count
        self.importance += other.importance
        self.access_count += other.access_count

        # Average the embeddings
        self.embedding = [(a + b) / 2 for a, b in zip(self.embedding, other.embedding)]

        # Merge tags and metadata
        self.tags = list(set(self.tags + other.tags))
        self.metadata.update(other.metadata)

        # Generate new summary
        self.summary = generate_summary(self.content)

        # Save updated memory and delete the merged one
        await self.save(deps)
        await deps.qdrant.delete(
            collection_name=COLLECTION_NAME, points_selector=[other.id]
        )

    def get_effective_importance(self) -> float:
        """Calculate effective importance with access count bonus"""
        return self.importance * (1 + math.log(self.access_count + 1))

    @classmethod
    def from_scored_point(cls, point: ScoredPoint):
        """Create MemoryNode from Qdrant ScoredPoint"""
        payload = point.payload
        return cls(
            id=point.id,
            content=payload["content"],
            summary=payload.get("summary", ""),
            importance=payload["importance"],
            access_count=payload["access_count"],
            timestamp=payload["timestamp"],
            embedding=[],
            tags=payload.get("tags", []),
            metadata=payload.get("metadata", {}),
        )


# ============================================================================
# Core Memory Operations
# ============================================================================


async def add_memory(
    content: str, deps: Deps, tags: list[str] = None, metadata: dict = None
) -> dict:
    """Add a new memory with automatic clustering and merging"""
    new_memory = await MemoryNode.from_content(content, deps, tags, metadata)
    await new_memory.save(deps)

    similar_memories = await find_similar_memories(new_memory.embedding, deps, limit=5)

    for memory in similar_memories:
        if memory.id != new_memory.id:
            memory_full = await get_memory_by_id(memory.id, deps)
            if memory_full and memory_full.embedding:
                similarity = cosine_similarity(
                    new_memory.embedding, memory_full.embedding
                )
                if similarity > MERGE_SIMILARITY_THRESHOLD:
                    print(f"🔗 Merging memories (similarity: {similarity:.3f})")
                    await new_memory.merge_with(memory_full, deps)
                    break

    await update_importance(new_memory.embedding, deps)
    await prune_memories(deps)

    return {
        "id": new_memory.id,
        "content": new_memory.content,
        "summary": new_memory.summary,
        "importance": new_memory.importance,
        "message": "Memory stored and clustered successfully",
    }


async def get_memory_by_id(memory_id: str, deps: Deps) -> Optional[MemoryNode]:
    """Get a specific memory by ID with full embedding"""
    result = await deps.qdrant.retrieve(
        collection_name=COLLECTION_NAME, ids=[memory_id], with_vectors=True
    )

    if not result:
        return None

    point = result[0]
    payload = point.payload

    return MemoryNode(
        id=point.id,
        content=payload["content"],
        summary=payload.get("summary", ""),
        importance=payload["importance"],
        access_count=payload["access_count"],
        timestamp=payload["timestamp"],
        embedding=point.vector,
        tags=payload.get("tags", []),
        metadata=payload.get("metadata", {}),
    )


async def find_similar_memories(
    embedding: list[float], deps: Deps, limit: int = 10
) -> list[MemoryNode]:
    """Find memories similar to the given embedding"""
    results = await deps.qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=limit,
        with_payload=True,
        with_vectors=True,
    )

    for result in results:
        await deps.qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"access_count": result.payload["access_count"] + 1},
            points=[result.id],
        )

    return [MemoryNode.from_scored_point(result) for result in results]


async def search_memories(
    query: str, deps: Deps, tags: list[str] = None, limit: int = 10
) -> list[dict]:
    """Search memories by query or tags"""
    query_embedding = get_embedding(query)

    query_filter = None
    if tags:
        query_filter = Filter(
            must=[FieldCondition(key="tags", match=MatchValue(any=tags))]
        )

    results = await deps.qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=query_filter,
        limit=limit * 2,
        with_payload=True,
        score_threshold=0.3,
    )

    memories = []
    for result in results:
        memory = MemoryNode.from_scored_point(result)
        memories.append(
            {
                "id": memory.id,
                "content": memory.content,
                "summary": memory.summary,
                "importance": memory.importance,
                "effective_importance": memory.get_effective_importance(),
                "access_count": memory.access_count,
                "tags": memory.tags,
                "timestamp": memory.timestamp,
                "relevance_score": result.score,
            }
        )

    memories.sort(key=lambda m: m["effective_importance"], reverse=True)

    return memories[:limit]


async def update_importance(user_embedding: list[float], deps: Deps):
    """Update importance of all memories based on similarity to new memory"""
    offset = None
    batch_size = 100

    while True:
        results, next_offset = await deps.qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )

        if not results:
            break

        for point in results:
            similarity = cosine_similarity(user_embedding, point.vector)
            current_importance = point.payload["importance"]

            if similarity > SIMILARITY_THRESHOLD:
                new_importance = current_importance * REINFORCEMENT_FACTOR
            else:
                new_importance = current_importance * DECAY_FACTOR

            await deps.qdrant.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"importance": new_importance},
                points=[point.id],
            )

        if next_offset is None:
            break
        offset = next_offset


async def prune_memories(deps: Deps):
    """Remove least important memories beyond MAX_DEPTH"""
    collection_info = await deps.qdrant.get_collection(collection_name=COLLECTION_NAME)
    total_count = collection_info.points_count

    if total_count <= MAX_DEPTH:
        return

    print(f"🌿 Pruning memories: {total_count} -> {MAX_DEPTH}")

    memories_with_importance = []
    offset = None

    while True:
        results, next_offset = await deps.qdrant.scroll(
            collection_name=COLLECTION_NAME, limit=100, offset=offset, with_payload=True
        )

        if not results:
            break

        for point in results:
            memory = MemoryNode.from_scored_point(point)
            effective_importance = memory.get_effective_importance()
            memories_with_importance.append((effective_importance, point.id))

        if next_offset is None:
            break
        offset = next_offset

    memories_with_importance.sort()
    to_remove = memories_with_importance[: len(memories_with_importance) - MAX_DEPTH]
    ids_to_remove = [mem_id for _, mem_id in to_remove]

    if ids_to_remove:
        await deps.qdrant.delete(
            collection_name=COLLECTION_NAME, points_selector=ids_to_remove
        )
        print(f"🗑️  Removed {len(ids_to_remove)} least important memories")


async def display_memory_tree(deps: Deps, limit: int = 20) -> str:
    """Display memory tree ordered by importance"""
    memories = []
    offset = None

    while True:
        results, next_offset = await deps.qdrant.scroll(
            collection_name=COLLECTION_NAME, limit=100, offset=offset, with_payload=True
        )

        if not results:
            break

        for point in results:
            memory = MemoryNode.from_scored_point(point)
            memories.append(memory)

        if next_offset is None:
            break
        offset = next_offset

    memories.sort(key=lambda m: m.get_effective_importance(), reverse=True)

    result = "Memory Tree (ordered by importance):\n"
    result += "=" * 50 + "\n\n"

    for i, memory in enumerate(memories[:limit], 1):
        effective_importance = memory.get_effective_importance()
        result += f"{i}. {memory.summary or memory.content[:100]}\n"
        result += f"   Importance: {effective_importance:.2f} "
        result += f"(base: {memory.importance:.2f}, accessed: {memory.access_count}x)\n"
        result += f"   ID: {memory.id}\n"
        if memory.tags:
            result += f"   Tags: {', '.join(memory.tags)}\n"
        result += "\n"

    return result


async def get_stats(deps: Deps) -> dict:
    """Get memory bank statistics"""
    collection_info = await deps.qdrant.get_collection(collection_name=COLLECTION_NAME)
    total_memories = collection_info.points_count

    if total_memories == 0:
        return {
            "total_memories": 0,
            "avg_importance": 0,
            "avg_effective_importance": 0,
            "total_accesses": 0,
            "unique_tags": 0,
            "tags": [],
            "max_depth": MAX_DEPTH,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "merge_threshold": MERGE_SIMILARITY_THRESHOLD,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "llm_model": LLM_MODEL_NAME,
            "device": str(device),
        }

    all_tags = set()
    total_importance = 0
    total_effective_importance = 0
    total_accesses = 0
    offset = None

    while True:
        results, next_offset = await deps.qdrant.scroll(
            collection_name=COLLECTION_NAME, limit=100, offset=offset, with_payload=True
        )

        if not results:
            break

        for point in results:
            memory = MemoryNode.from_scored_point(point)
            total_importance += memory.importance
            total_effective_importance += memory.get_effective_importance()
            total_accesses += memory.access_count
            all_tags.update(memory.tags)

        if next_offset is None:
            break
        offset = next_offset

    return {
        "total_memories": total_memories,
        "avg_importance": (
            total_importance / total_memories if total_memories > 0 else 0
        ),
        "avg_effective_importance": (
            total_effective_importance / total_memories if total_memories > 0 else 0
        ),
        "total_accesses": total_accesses,
        "unique_tags": len(all_tags),
        "tags": sorted(list(all_tags)),
        "max_depth": MAX_DEPTH,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "merge_threshold": MERGE_SIMILARITY_THRESHOLD,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "llm_model": LLM_MODEL_NAME,
        "device": str(device),
        "qdrant_host": QDRANT_HOST,
        "qdrant_port": QDRANT_PORT,
    }


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
async def remember(
    contents: list[str] = Field(
        description="List of observations or memories to store"
    ),
    tags: list[str] = Field(
        default=None, description="Optional tags for categorization"
    ),
    metadata: dict = Field(default=None, description="Optional metadata dictionary"),
) -> str:
    """
    Store new memories with automatic clustering and importance tracking.
    Uses local embedding model for vector generation.
    """
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        results = []
        for content in contents:
            result = await add_memory(content, deps, tags, metadata)
            results.append(result)

        return json.dumps({"stored": len(results), "memories": results}, indent=2)
    finally:
        await deps.qdrant.close()


@mcp.tool()
async def recall(
    query: str = Field(description="Search query to find relevant memories"),
    tags: list[str] = Field(default=None, description="Optional tags to filter by"),
    limit: int = Field(default=10, description="Maximum number of memories to return"),
) -> str:
    """
    Search and retrieve memories based on semantic similarity using local embedding model.
    Returns memories ordered by relevance and importance.
    """
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        results = await search_memories(query, deps, tags, limit)

        return json.dumps(
            {"query": query, "found": len(results), "memories": results}, indent=2
        )
    finally:
        await deps.qdrant.close()


@mcp.tool()
async def read_profile(
    limit: int = Field(default=20, description="Number of memories to display")
) -> str:
    """
    Display the memory tree showing the most important memories.
    Memories are ordered by effective importance.
    """
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        return await display_memory_tree(deps, limit)
    finally:
        await deps.qdrant.close()


@mcp.tool()
async def get_memory_stats() -> str:
    """
    Get statistics about the memory bank including model information.
    """
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        stats = await get_stats(deps)
        return json.dumps(stats, indent=2)
    finally:
        await deps.qdrant.close()


@mcp.tool()
async def forget_memory(
    memory_id: str = Field(description="ID of memory to delete"),
) -> str:
    """
    Delete a specific memory by ID from Qdrant.
    """
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        result = await deps.qdrant.delete(
            collection_name=COLLECTION_NAME, points_selector=[memory_id]
        )

        if result.status == UpdateStatus.COMPLETED:
            return json.dumps(
                {"success": True, "message": f"Memory {memory_id} deleted"}
            )
        return json.dumps(
            {"success": False, "error": f"Failed to delete memory {memory_id}"}
        )
    finally:
        await deps.qdrant.close()


@mcp.tool()
async def reinforce_memory(
    memory_id: str = Field(description="ID of memory to reinforce"),
) -> str:
    """
    Manually reinforce a memory's importance.
    """
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        memory = await get_memory_by_id(memory_id, deps)
        if not memory:
            return json.dumps(
                {"success": False, "error": f"Memory {memory_id} not found"}
            )

        memory.importance *= REINFORCEMENT_FACTOR
        memory.access_count += 1
        await memory.save(deps)

        return json.dumps(
            {
                "success": True,
                "memory_id": memory_id,
                "new_importance": memory.importance,
                "effective_importance": memory.get_effective_importance(),
            }
        )
    finally:
        await deps.qdrant.close()


# ============================================================================
# HTTP API Handlers
# ============================================================================


async def health_check(request):
    """Health check endpoint"""
    try:
        qdrant = await get_qdrant_client()
        collection_info = await qdrant.get_collection(collection_name=COLLECTION_NAME)
        await qdrant.close()

        return web.json_response(
            {
                "status": "healthy",
                "service": "mcp-memory-server-qdrant-local",
                "version": "4.0",
                "memory_count": collection_info.points_count,
                "qdrant": f"{QDRANT_HOST}:{QDRANT_PORT}",
                "embedding_model": EMBEDDING_MODEL_NAME,
                "llm_model": LLM_MODEL_NAME,
                "device": str(device),
            }
        )
    except Exception as e:
        return web.json_response({"status": "unhealthy", "error": str(e)}, status=503)


async def api_remember(request):
    """Store new memory via HTTP"""
    data = await request.json()
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        result = await add_memory(
            content=data.get("content"),
            deps=deps,
            tags=data.get("tags"),
            metadata=data.get("metadata"),
        )
        return web.json_response(result)
    finally:
        await deps.qdrant.close()


async def api_recall(request):
    """Search memories via HTTP"""
    data = await request.json()
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        results = await search_memories(
            query=data.get("query"),
            deps=deps,
            tags=data.get("tags"),
            limit=data.get("limit", 10),
        )
        return web.json_response({"found": len(results), "memories": results})
    finally:
        await deps.qdrant.close()


async def api_profile(request):
    """Get memory profile via HTTP"""
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        limit = int(request.query.get("limit", 20))
        profile = await display_memory_tree(deps, limit)
        return web.Response(text=profile, content_type="text/plain")
    finally:
        await deps.qdrant.close()


async def api_stats(request):
    """Get stats via HTTP"""
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        stats = await get_stats(deps)
        return web.json_response(stats)
    finally:
        await deps.qdrant.close()


async def api_forget(request):
    """Delete memory via HTTP"""
    memory_id = request.match_info["id"]
    deps = Deps(qdrant=await get_qdrant_client())

    try:
        result = await deps.qdrant.delete(
            collection_name=COLLECTION_NAME, points_selector=[memory_id]
        )

        if result.status == UpdateStatus.COMPLETED:
            return web.json_response({"success": True, "id": memory_id})
        return web.json_response({"error": "Memory not found"}, status=404)
    finally:
        await deps.qdrant.close()


def create_http_app():
    """Create HTTP API application"""
    http_app = web.Application()

    http_app.router.add_get("/health", health_check)
    http_app.router.add_get("/stats", api_stats)
    http_app.router.add_get("/profile", api_profile)
    http_app.router.add_post("/remember", api_remember)
    http_app.router.add_post("/recall", api_recall)
    http_app.router.add_delete("/forget/{id}", api_forget)

    return http_app


# ============================================================================
# Main Application
# ============================================================================


async def main():
    """Main entry point"""
    # Initialize local models
    print("🚀 Initializing MCP Memory Server with Local Models...")
    initialize_models()

    # Initialize Qdrant connection
    try:
        qdrant = await get_qdrant_client()
        await qdrant.close()
        print("✅ Qdrant connection successful")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print(f"Make sure Qdrant is running at {QDRANT_HOST}:{QDRANT_PORT}")
        return

    # Start HTTP server
    http_app = create_http_app()
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 9393)
    await site.start()

    print("\n🧠 MCP Recursive Memory Server with Local Models started!")
    print("=" * 60)
    print(f"📡 HTTP API: http://localhost:9393")
    print(f"🗄️  Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"📦 Collection: {COLLECTION_NAME}")
    print(f"🤖 Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"🤖 LLM Model: {LLM_MODEL_NAME}")
    print(f"💻 Device: {device}")
    print(f"📐 Embedding Dimension: {EMBEDDING_DIMENSION}")
    print(f"🔧 MCP Tools: 6 recursive memory tools")
    print(f"⚙️  Max Depth: {MAX_DEPTH} memories")
    print(f"⚙️  Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"⚙️  Merge Threshold: {MERGE_SIMILARITY_THRESHOLD}")
    print("\nHTTP Endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /stats - Memory statistics")
    print("  GET  /profile?limit=20 - Memory tree")
    print("  POST /remember - Store memory")
    print("  POST /recall - Search memories")
    print("  DELETE /forget/{id} - Delete memory")
    print("\nMCP Tools:")
    print("  - remember: Store new memories with auto-clustering")
    print("  - recall: Search memories by semantic similarity")
    print("  - read_profile: Display memory tree")
    print("  - get_memory_stats: View statistics")
    print("  - forget_memory: Delete a memory")
    print("  - reinforce_memory: Boost importance")
    print("=" * 60)
    print("\n✨ All models loaded and ready!")
    print("💡 Tip: First embedding generation may be slow, then it's fast\n")

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
