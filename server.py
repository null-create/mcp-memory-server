"""
MCP Server for Long-term Memory Storage with HTTP API
Uses TinyDB (NoSQL) to store and retrieve memories for LLM conversations
"""

import json
import asyncio
from datetime import datetime
from typing import Any, Optional
from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, TextContent
from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
from aiohttp import web

# Initialize MCP server
app = Server("memory-server")

# Database setup
DB_PATH = Path("/data/memory_bank.json")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
db = TinyDB(DB_PATH, storage=CachingMiddleware(JSONStorage))
memories_table = db.table("memories")
Memory = Query()


# Helper functions
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
    return {"id": memory_id, **memory}


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

    for memory in results[:limit]:
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
    sorted_memories = sorted(
        all_memories, key=lambda x: x.get("timestamp", ""), reverse=True
    )
    return sorted_memories[:limit]


def get_frequent_memories(limit: int = 10) -> list[dict]:
    """Get most frequently accessed memories"""
    all_memories = memories_table.all()
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


# HTTP API handlers
async def health_check(request):
    """Health check endpoint"""
    return web.json_response({"status": "healthy", "service": "mcp-memory-server"})


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
    all_memories = memories_table.all()
    all_tags = set()
    for memory in all_memories:
        all_tags.update(memory.get("tags", []))
    return web.json_response({"tags": sorted(list(all_tags))})


def create_http_app() -> web.Application:
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


# MCP Tool Handlers
@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available memory tools"""
    return [
        Tool(
            name="store_memory",
            description="Store a new memory in the long-term memory bank",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="search_memories",
            description="Search for memories by keyword or tags",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        ),
        Tool(
            name="get_memory",
            description="Retrieve a specific memory by ID",
            inputSchema={
                "type": "object",
                "properties": {"memory_id": {"type": "integer"}},
                "required": ["memory_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    if name == "store_memory":
        memory = create_memory(
            content=arguments["content"],
            tags=arguments.get("tags"),
            metadata=arguments.get("metadata"),
        )
        return [TextContent(type="text", text=json.dumps(memory, indent=2))]

    elif name == "search_memories":
        results = search_memories(
            query=arguments.get("query"),
            tags=arguments.get("tags"),
            limit=arguments.get("limit", 10),
        )
        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    elif name == "get_memory":
        memory = get_memory_by_id(arguments["memory_id"])
        return [TextContent(type="text", text=json.dumps(memory, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main() -> None:
    """Run both HTTP API and MCP server"""
    # Start HTTP server
    http_app = create_http_app()
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 9393)
    await site.start()

    print("🚀 MCP Memory Server started!")
    print("📡 HTTP API: http://localhost:9393")
    print("💾 Database: /data/memory_bank.json")
    print("\nAvailable endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /stats - Memory statistics")
    print("  POST /memories - Create memory")
    print("  POST /memories/search - Search memories")
    print("  GET  /memories/{id} - Get memory")
    print("  PUT  /memories/{id} - Update memory")
    print("  DELETE /memories/{id} - Delete memory")

    # Keep server running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
