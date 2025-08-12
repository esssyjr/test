import json
import os
from mcp.server.fastmcp import FastMCP


# Keep using the same MCP instance as your other tools/resources
mcp = FastMCP("test", port = 8001)

# ✅ Simple tool (for MCP test)
@mcp.tool()
def sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


# ✅ Run the server
if __name__ == "__main__":
    mcp.run(transport='streamable-http')
