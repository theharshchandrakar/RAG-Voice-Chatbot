"""
Database Module
SQL utilities for schema retrieval, query execution, safety checks, and SQL generation.
"""

import os
import re
import sqlite3
from pathlib import Path
import pandas as pd


# Constants
FORBIDDEN_KEYWORDS = {
    "delete", "drop", "update", "insert", "alter",
    "truncate", "create", "replace", "attach", "detach"
}
MAX_SQL_LIMIT = 100


def get_db_schema(db_path: Path) -> str:
    """
    Get SQLite database schema for LLM context.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        str: Schema description or error message
    """
    if not db_path.exists():
        return "No database found. Upload a CSV first."
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT name, sql FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """)
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        return "Database is empty. Upload CSV data first."
    
    return "\n\n".join(sql + ";" for _, sql in rows)


def execute_sql(sql: str, db_path: Path):
    """
    Execute SQL query and return results as list of dicts.
    
    Args:
        sql: SQL query to execute
        db_path: Path to SQLite database file
        
    Returns:
        list[dict]: Query results
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def enforce_sql_safety(sql: str) -> str:
    """
    Enforce read-only SQL and auto-add LIMIT.
    
    Args:
        sql: SQL query to validate
        
    Returns:
        str: Safe SQL query with LIMIT
        
    Raises:
        ValueError: If query is not SELECT or contains forbidden keywords
    """
    sql_clean = sql.strip().lower()
    
    # Must start with SELECT
    if not sql_clean.startswith("select"):
        raise ValueError("Only SELECT queries are allowed")
    
    # Check for forbidden keywords
    for word in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{word}\b", sql_clean):
            raise ValueError(f"Forbidden SQL operation: {word.upper()}")
    
    # Auto-add LIMIT if missing
    if " limit " not in sql_clean:
        sql = sql.rstrip(";") + f" LIMIT {MAX_SQL_LIMIT};"
    
    return sql


def repair_sql(sql: str) -> str:
    """
    Fix common SQL generation errors.
    
    Args:
        sql: Malformed SQL query
        
    Returns:
        str: Repaired SQL query
        
    Raises:
        ValueError: If SQL is unrepairable (missing FROM clause)
    """
    sql = sql.strip()
    sql = re.sub(r"\s+", " ", sql)  # Normalize whitespace
    
    # Fix: SELECT ... WHERE ... FROM table → SELECT ... FROM table WHERE ...
    sql = re.sub(
        r"SELECT (.+?) WHERE (.+?) FROM ([a-zA-Z0-9_]+)",
        r"SELECT \1 FROM \3 WHERE \2",
        sql, flags=re.I
    )
    
    # Fix: SELECT ... ORDER BY ... FROM table → SELECT ... FROM table ORDER BY ...
    sql = re.sub(
        r"SELECT (.+?) ORDER BY (.+?) FROM ([a-zA-Z0-9_]+)",
        r"SELECT \1 FROM \3 ORDER BY \2",
        sql, flags=re.I
    )
    
    # Remove duplicate FROM
    sql = re.sub(r"FROM\s+FROM", "FROM", sql, flags=re.I)
    
    # Ensure FROM exists
    if not re.search(r"\bfrom\b", sql, re.I):
        raise ValueError("Invalid SQL (missing FROM clause)")
    
    return sql.strip()


def format_sql_table(rows: list[dict]) -> str:
    """
    Format query results as markdown table.
    
    Args:
        rows: Query results (list of dicts)
        
    Returns:
        str: Markdown-formatted table
    """
    if not rows:
        return "*(no results)*"
    
    headers = list(rows[0].keys())
    
    # Create markdown table
    markdown = "\n"
    
    # Header row
    markdown += "| " + " | ".join(str(h) for h in headers) + " |\n"
    # Separator row
    markdown += "|" + "|".join(["---"] * len(headers)) + "|\n"
    
    # Data rows
    for r in rows:
        markdown += "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |\n"
    
    return markdown


def generate_sql_with_llm(
    prompt: str,
    schema: str,
    groq_client,
    groq_model: str,
    ollama_client=None,
    ollama_model: str = None,
    ollama_host: str = None,
    messages: list[dict] = None
) -> str:
    """
    Generate SQL using Ollama-first, Groq fallback with optional conversation context.
    
    Args:
        prompt: User's natural language query
        schema: Database schema description
        groq_client: Groq API client
        groq_model: Groq model name
        ollama_client: Optional Ollama client
        ollama_model: Optional Ollama model name
        ollama_host: Optional Ollama host URL
        messages: Optional conversation context
        
    Returns:
        str: Generated SQL query
        
    Raises:
        ValueError: If SQL generation fails on both Ollama and Groq
    """
    # Build conversation context summary if messages provided
    context_summary = ""
    if messages and len(messages) > 1:
        # Summarize recent messages (excluding the new user message at the end)
        recent_msgs = [m for m in messages[:-1] if m.get("role") in ["user", "assistant"]]
        if recent_msgs:
            context_summary = "\n\nRecent conversation context:\n"
            for msg in recent_msgs[-4:]:  # Last 2 exchanges
                role = msg["role"].upper()
                content = msg["content"][:200]  # Limit length
                context_summary += f"{role}: {content}\n"
    
    system_prompt = f"""You are an expert SQLite SQL generator who understands conversation context.

Database schema:
{schema}
{context_summary}

Rules:
- Use existing tables only
- SELECT queries only
- SQLite syntax
- No explanation or markdown
- Output ONLY the SQL query
- Use context from previous queries if the user is asking follow-up questions

User request:
{prompt}
"""
    
    # Try Ollama first
    try:
        if ollama_host and ollama_client:
            response = ollama_client.chat(
                model=ollama_model,
                messages=[{'role': 'user', 'content': system_prompt}],
                stream=False
            )
            sql = response['message']['content'].strip()
            # Remove markdown code blocks if present
            sql = re.sub(r'^```sql\s*', '', sql, flags=re.MULTILINE)
            sql = re.sub(r'^```\s*', '', sql, flags=re.MULTILINE)
            sql = re.sub(r'```$', '', sql)
            return sql.strip()
    except Exception as e:
        print(f"Ollama SQL generation failed: {e}")
    
    # Fallback to Groq
    try:
        response = groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.1
        )
        sql = response.choices[0].message.content.strip()
        # Remove markdown code blocks
        sql = re.sub(r'^```sql\s*', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'^```\s*', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'```$', '', sql)
        return sql.strip()
    except Exception as e:
        raise ValueError(f"SQL generation failed: {str(e)}")
