"""
app/api/router.py
-----------------
Aggregates all route modules into a single router.

main.py imports only this — it never imports individual route files directly.
Adding a new resource means: create the route file, then add one line here.
"""

from fastapi import APIRouter

from app.api.routes import chat, conversations

api_router = APIRouter()

api_router.include_router(chat.router)
api_router.include_router(conversations.router)