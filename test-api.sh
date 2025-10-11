#!/bin/bash

echo "Testing MCP Memory Server API..."
echo "================================"

# Health check
echo -e "\n1. Health Check"
curl -s http://localhost:9393/health | jq

# Store memories
echo -e "\n2. Storing memories..."
curl -s -X POST http://localhost:9393/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "User loves Python programming", "tags": ["skill", "preference"]}' | jq

curl -s -X POST http://localhost:9393/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "User is working on an MCP server project", "tags": ["project", "current"]}' | jq

curl -s -X POST http://localhost:9393/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode interface", "tags": ["preference", "ui"]}' | jq

# Search
echo -e "\n3. Searching for 'Python'..."
curl -s -X POST http://localhost:9393/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Python"}' | jq

# Get by tag
echo -e "\n4. Searching by tag 'preference'..."
curl -s -X POST http://localhost:9393/memories/search \
  -H "Content-Type: application/json" \
  -d '{"tags": ["preference"]}' | jq

# Stats
echo -e "\n5. Getting statistics..."
curl -s http://localhost:9393/stats | jq

# Recent memories
echo -e "\n6. Getting recent memories..."
curl -s http://localhost:9393/memories/recent?limit=5 | jq

echo -e "\n✅ API tests completed!"