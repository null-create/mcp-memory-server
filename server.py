"""
MCP Server for Long-term Memory Storage with HTTP API
Uses TinyDB (NoSQL) and FastMCP for tool management
"""

import json
import logging
import asyncio
import argparse
from datetime import datetime
from typing import Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
from aiohttp import web

# Server configs
HOST_ADDR = "0.0.0.0"
HOST_PORT = 9321

# Set up logging
logging.basicConfig(level="INFO")
logger = logging.getLogger(__file__)

# Initialize FastMCP server
mcp = FastMCP(name="memory-server", host=HOST_ADDR, port=HOST_PORT)

# Database setup
DB_PATH = Path("/data/memory_bank.json")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
db = TinyDB(DB_PATH, storage=CachingMiddleware(JSONStorage))
memories_table = db.table("memories")
Memory = Query()


# Parse args
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Memory MCP Server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Whether to run in http server mode (defaults to running MCP server directly)",
        default=False,
    )
    return parser.parse_args()


# ============================================================================
# Database Helper Functions
# ============================================================================


def create_memory(content: str, tags: list[str] = None, metadata: dict = None) -> dict:
    """Create a new memory entry"""
    memory = {
        "content": content,
        "tags": tags or [],
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
        "access_count": 0,
        "last_accessed": None,
    }
    memory_id = memories_table.insert(memory)
    memory["id"] = memory_id
    return memory


def search_memories(
    query: str = None, tags: list[str] = None, limit: int = 10
) -> list[dict]:
    """Search memories by content or tags"""
    results = []

    if tags:
        results = memories_table.search(Memory.tags.any(tags))
    elif query:
        results = memories_table.search(Memory.content.search(query, flags=0))
    else:
        results = memories_table.all()

    # Add IDs and update access count
    for memory in results[:limit]:
        memory["id"] = memory.doc_id
        memories_table.update(
            {
                "access_count": memory.get("access_count", 0) + 1,
                "last_accessed": datetime.now().isoformat(),
            },
            doc_ids=[memory.doc_id],
        )

    return results[:limit]


def get_memory_by_id(memory_id: int) -> Optional[dict]:
    """Retrieve a specific memory by ID"""
    memory = memories_table.get(doc_id=memory_id)
    if memory:
        memory["id"] = memory_id
        memories_table.update(
            {
                "access_count": memory.get("access_count", 0) + 1,
                "last_accessed": datetime.now().isoformat(),
            },
            doc_ids=[memory_id],
        )
    return memory


def update_memory(
    memory_id: int, content: str = None, tags: list[str] = None, metadata: dict = None
) -> bool:
    """Update an existing memory"""
    updates = {}
    if content is not None:
        updates["content"] = content
    if tags is not None:
        updates["tags"] = tags
    if metadata is not None:
        updates["metadata"] = metadata

    if updates:
        updates["updated_at"] = datetime.now().isoformat()
        return len(memories_table.update(updates, doc_ids=[memory_id])) > 0
    return False


def delete_memory(memory_id: int) -> bool:
    """Delete a memory by ID"""
    return len(memories_table.remove(doc_ids=[memory_id])) > 0


def get_recent_memories(limit: int = 10) -> list[dict]:
    """Get most recently created memories"""
    all_memories = memories_table.all()
    for memory in all_memories:
        memory["id"] = memory.doc_id
    sorted_memories = sorted(
        all_memories, key=lambda x: x.get("timestamp", ""), reverse=True
    )
    return sorted_memories[:limit]


def get_frequent_memories(limit: int = 10) -> list[dict]:
    """Get most frequently accessed memories"""
    all_memories = memories_table.all()
    for memory in all_memories:
        memory["id"] = memory.doc_id
    sorted_memories = sorted(
        all_memories, key=lambda x: x.get("access_count", 0), reverse=True
    )
    return sorted_memories[:limit]


def get_stats() -> dict:
    """Get memory bank statistics"""
    total_memories = len(memories_table)
    all_memories = memories_table.all()

    total_accesses = sum(m.get("access_count", 0) for m in all_memories)
    all_tags = set()
    for memory in all_memories:
        all_tags.update(memory.get("tags", []))

    return {
        "total_memories": total_memories,
        "total_accesses": total_accesses,
        "unique_tags": len(all_tags),
        "tags": sorted(list(all_tags)),
    }


def get_all_tags() -> list[str]:
    """Get all unique tags"""
    all_memories = memories_table.all()
    all_tags = set()
    for memory in all_memories:
        all_tags.update(memory.get("tags", []))
    return sorted(list(all_tags))


# ============================================================================
# MCP Tools using @mcp.tool decorator
# ============================================================================


@mcp.tool()
def store_memory(content: str, tags: list[str] = None, metadata: dict = None) -> str:
    """
    Store a new memory in the long-term memory bank.

    Args:
        content: The memory content to store
        tags: Optional list of tags to categorize the memory (e.g., 'user_preference', 'fact', 'conversation')
        metadata: Optional metadata dictionary for additional context

    Returns:
        JSON string with the stored memory including its ID
    """
    memory = create_memory(content, tags, metadata)
    return json.dumps(memory, indent=2)


@mcp.tool()
def search_memory(query: str = None, tags: list[str] = None, limit: int = 10) -> str:
    """
    Search for memories by keyword or tags.

    Args:
        query: Search query to find in memory content
        tags: List of tags to filter memories by
        limit: Maximum number of memories to return (default: 10)

    Returns:
        JSON string with found memories
    """
    results = search_memories(query, tags, limit)
    if results:
        return json.dumps({"count": len(results), "memories": results}, indent=2)
    return json.dumps(
        {"count": 0, "message": "No memories found matching the criteria"}
    )


@mcp.tool()
def get_memory(memory_id: int) -> str:
    """
    Retrieve a specific memory by its ID.

    Args:
        memory_id: The ID of the memory to retrieve

    Returns:
        JSON string with the memory or error message
    """
    memory = get_memory_by_id(memory_id)
    if memory:
        return json.dumps(memory, indent=2)
    return json.dumps({"error": f"Memory with ID {memory_id} not found"})


@mcp.tool()
def update_memory_content(
    memory_id: int, content: str = None, tags: list[str] = None, metadata: dict = None
) -> str:
    """
    Update an existing memory's content, tags, or metadata.

    Args:
        memory_id: The ID of the memory to update
        content: New content for the memory (optional)
        tags: New tags for the memory (optional)
        metadata: New metadata for the memory (optional)

    Returns:
        JSON string with success status
    """
    success = update_memory(memory_id, content, tags, metadata)
    if success:
        return json.dumps(
            {"success": True, "message": f"Memory {memory_id} updated successfully"}
        )
    return json.dumps(
        {"success": False, "error": f"Failed to update memory {memory_id}"}
    )


@mcp.tool()
def delete_memory_by_id(memory_id: int) -> str:
    """
    Delete a memory from the memory bank.

    Args:
        memory_id: The ID of the memory to delete

    Returns:
        JSON string with success status
    """
    success = delete_memory(memory_id)
    if success:
        return json.dumps(
            {"success": True, "message": f"Memory {memory_id} deleted successfully"}
        )
    return json.dumps(
        {"success": False, "error": f"Failed to delete memory {memory_id}"}
    )


@mcp.tool()
def get_recent_memory_list(limit: int = 10) -> str:
    """
    Get the most recently stored memories.

    Args:
        limit: Maximum number of memories to return (default: 10)

    Returns:
        JSON string with recent memories
    """
    memories = get_recent_memories(limit)
    return json.dumps({"count": len(memories), "memories": memories}, indent=2)


@mcp.tool()
def get_frequent_memory_list(limit: int = 10) -> str:
    """
    Get the most frequently accessed memories.

    Args:
        limit: Maximum number of memories to return (default: 10)

    Returns:
        JSON string with frequently accessed memories
    """
    memories = get_frequent_memories(limit)
    return json.dumps({"count": len(memories), "memories": memories}, indent=2)


@mcp.tool()
def list_all_tags() -> str:
    """
    Get a list of all unique tags used in memories.

    Returns:
        JSON string with all tags
    """
    tags = get_all_tags()
    return json.dumps({"count": len(tags), "tags": tags}, indent=2)


@mcp.tool()
def get_memory_stats() -> str:
    """
    Get statistics about the memory bank.

    Returns:
        JSON string with memory bank statistics
    """
    stats = get_stats()
    return json.dumps(stats, indent=2)


# ============================================================================
# HTTP API Handlers
# ============================================================================


async def health_check(request):
    """Health check endpoint"""
    return web.json_response(
        {"status": "healthy", "service": "mcp-memory-server", "version": "2.0-fastmcp"}
    )


async def api_store_memory(request):
    """Store a new memory"""
    data = await request.json()
    memory = create_memory(
        content=data.get("content"),
        tags=data.get("tags"),
        metadata=data.get("metadata"),
    )
    return web.json_response(memory)


async def api_search_memories(request):
    """Search memories"""
    data = await request.json()
    results = search_memories(
        query=data.get("query"), tags=data.get("tags"), limit=data.get("limit", 10)
    )
    return web.json_response({"count": len(results), "memories": results})


async def api_get_memory(request):
    """Get a specific memory"""
    memory_id = int(request.match_info["id"])
    memory = get_memory_by_id(memory_id)
    if memory:
        return web.json_response(memory)
    return web.json_response({"error": "Memory not found"}, status=404)


async def api_update_memory(request):
    """Update a memory"""
    memory_id = int(request.match_info["id"])
    data = await request.json()
    success = update_memory(
        memory_id=memory_id,
        content=data.get("content"),
        tags=data.get("tags"),
        metadata=data.get("metadata"),
    )
    if success:
        return web.json_response({"success": True, "id": memory_id})
    return web.json_response({"error": "Memory not found"}, status=404)


async def api_delete_memory(request):
    """Delete a memory"""
    memory_id = int(request.match_info["id"])
    success = delete_memory(memory_id)
    if success:
        return web.json_response({"success": True, "id": memory_id})
    return web.json_response({"error": "Memory not found"}, status=404)


async def api_get_recent(request):
    """Get recent memories"""
    limit = int(request.query.get("limit", 10))
    memories = get_recent_memories(limit)
    return web.json_response({"count": len(memories), "memories": memories})


async def api_get_frequent(request):
    """Get frequently accessed memories"""
    limit = int(request.query.get("limit", 10))
    memories = get_frequent_memories(limit)
    return web.json_response({"count": len(memories), "memories": memories})


async def api_get_stats(request):
    """Get memory bank statistics"""
    stats = get_stats()
    return web.json_response(stats)


async def api_get_tags(request):
    """Get all tags"""
    tags = get_all_tags()
    return web.json_response({"tags": tags})


def create_http_app():
    """Create the HTTP API application"""
    http_app = web.Application()

    # Add routes
    http_app.router.add_get("/health", health_check)
    http_app.router.add_get("/stats", api_get_stats)
    http_app.router.add_get("/tags", api_get_tags)
    http_app.router.add_post("/memories", api_store_memory)
    http_app.router.add_post("/memories/search", api_search_memories)
    http_app.router.add_get("/memories/recent", api_get_recent)
    http_app.router.add_get("/memories/frequent", api_get_frequent)
    http_app.router.add_get("/memories/{id}", api_get_memory)
    http_app.router.add_put("/memories/{id}", api_update_memory)
    http_app.router.add_delete("/memories/{id}", api_delete_memory)

    return http_app


# ============================================================================
# Main Web Application
# ============================================================================


async def run_http_server():
    """Run HTTP API server"""
    http_app = create_http_app()
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 9393)
    await site.start()

    print("🚀 MCP Memory Server started!")
    print("📡 HTTP API: http://localhost:9393")
    print("💾 Database: /data/memory_bank.json")
    print("🔧 MCP Tools: 9 tools registered via FastMCP")
    print("\nAvailable HTTP endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /stats - Memory statistics")
    print("  POST /memories - Create memory")
    print("  POST /memories/search - Search memories")
    print("  GET  /memories/{id} - Get memory")
    print("  PUT  /memories/{id} - Update memory")
    print("  DELETE /memories/{id} - Delete memory")
    print("\nMCP Tools available:")
    print("  - store_memory")
    print("  - search_memory")
    print("  - get_memory")
    print("  - update_memory_content")
    print("  - delete_memory_by_id")
    print("  - get_recent_memory_list")
    print("  - get_frequent_memory_list")
    print("  - list_all_tags")
    print("  - get_memory_stats")

    return runner


async def run_http() -> None:
    """Main entry point for http server"""
    # Start HTTP server
    runner = await run_http_server()

    # Keep server running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await runner.cleanup()


if __name__ == "__main__":
    args = parse_args()
    if args.http:
        run_http()
    else:
        logger.info(f"Starting MCP server at {HOST_ADDR}:{HOST_PORT}...")
        mcp.run(transport="streamable-http")
