# Import tools so they get registered via decorators
import tools.metrics  # noqa: F401
import tools.plot  # noqa: F401
import tools.retrieve  # noqa: F401
from server import mcp

if __name__ == "__main__":
    mcp.run()
