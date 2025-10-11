# MCP Memory Server

A Model Context Protocol (MCP) server providing long-term memory storage for LLMs using a NoSQL database (TinyDB).

## Features

- 🧠 Long-term memory storage across conversations
- 🔍 Search by keywords and tags
- 📊 Access tracking and statistics
- 🐳 Docker & Docker Compose ready
- 🌐 HTTP REST API for easy testing
- 💾 Persistent storage with TinyDB

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone or create the repository
mkdir mcp-memory-server
cd mcp-memory-server

# Create all required files (see repository structure below)

# Start the server
docker-compose up -d

# Check logs
docker-compose logs -f

# Test the server
curl http://localhost:9393/health
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py
```

## API Endpoints

### Health Check

```bash
curl http://localhost:9393/health
```

### Store a Memory

```bash
curl -X POST http://localhost:9393/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User prefers dark mode",
    "tags": ["preference", "ui"],
    "metadata": {"importance": "high"}
  }'
```

### Search Memories

```bash
# Search by keyword
curl -X POST http://localhost:9393/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "dark mode", "limit": 5}'

# Search by tags
curl -X POST http://localhost:9393/memories/search \
  -H "Content-Type: application/json" \
  -d '{"tags": ["preference"], "limit": 10}'
```

### Get a Specific Memory

```bash
curl http://localhost:9393/memories/1
```

### Update a Memory

```bash
curl -X PUT http://localhost:9393/memories/1 \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode with blue accent"}'
```

### Delete a Memory

```bash
curl -X DELETE http://localhost:9393/memories/1
```

### Get Recent Memories

```bash
curl http://localhost:9393/memories/recent?limit=10
```

### Get Frequently Accessed Memories

```bash
curl http://localhost:9393/memories/frequent?limit=10
```

### Get Statistics

```bash
curl http://localhost:9393/stats
```

### Get All Tags

```bash
curl http://localhost:9393/tags
```

## Repository Structure

```
mcp-memory-server/
├── server.py              # Main server code
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker Compose configuration
├── README.md            # This file
├── .gitignore           # Git ignore rules
└── data/                # Persistent data directory
    └── memory_bank.json # TinyDB database (created automatically)
```

## Configuration

The server runs on port 9393 by default. To change:

1. Update `docker-compose.yml` ports mapping
2. Update the port in `server.py` (line with `TCPSite`)

## MCP Integration

To use with Claude Desktop or other MCP clients:

```json
{
  "mcpServers": {
    "memory": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-v", "./data:/data", "mcp-memory-server"]
    }
  }
}
```

## Development

```bash
# Build the Docker image
docker build -t mcp-memory-server .

# Run with custom data directory
docker run -p 9393:9393 -v $(pwd)/custom-data:/data mcp-memory-server

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

## Data Persistence

The database is stored in `/data/memory_bank.json` inside the container, which is mounted to `./data/` on your host machine. This ensures memories persist across container restarts.

## License

MIT
