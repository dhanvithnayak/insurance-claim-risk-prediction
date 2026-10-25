import os
import sys

# Add repository root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


# Normalize incoming paths from Vercel serverless rewrites
class VercelPathMiddleware:
    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            for prefix in ("/api/index.py", "/api/index"):
                if path == prefix:
                    scope["path"] = "/"
                    break
                elif path.startswith(prefix + "/"):
                    scope["path"] = path[len(prefix):]
                    break
        await self.asgi_app(scope, receive, send)


app.add_middleware(VercelPathMiddleware)
