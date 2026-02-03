"""
CSV Handler Module
Handles CSV upload and SQL table creation.
"""

import io
import re
import sqlite3
import pandas as pd


async def process_csv(
    file_bytes: bytes,
    filename: str,
    db_path: str
):
    """
    Process uploaded CSV: parse, sanitize table name, and create SQL table.
    
    Args:
        file_bytes: Raw CSV file bytes
        filename: Original filename (used to derive table name)
        db_path: Path to SQLite database file
        
    Returns:
        dict: Status with table name, row count, columns, and message
        
    Raises:
        ValueError: If CSV parsing fails
    """
    print(f"📊 Processing CSV: {filename}")
    
    # Parse CSV
    df = pd.read_csv(io.BytesIO(file_bytes))
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Generate sanitized table name from filename
    table = filename.replace(".csv", "").lower().replace(" ", "_").replace("-", "_")
    table = re.sub(r'[^a-z0-9_]', '', table)  # Remove non-alphanumeric chars
    
    print(f"Creating table: {table} with {len(df)} rows")
    
    # Write to SQLite
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    
    return {
        "status": "success",
        "table": table,
        "rows_inserted": len(df),
        "columns": list(df.columns),
        "message": f"Uploaded {len(df)} rows to table '{table}'"
    }
