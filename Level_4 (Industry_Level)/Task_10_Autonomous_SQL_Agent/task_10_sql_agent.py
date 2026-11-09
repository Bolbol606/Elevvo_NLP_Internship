# Task 10 - Autonomous Text-to-SQL Agent (Chinook Database)
# The goal: a user types a plain English question like "how many tracks does each artist have?"
# and the agent translates that into a valid SQL query, runs it, and returns the answer.
# The interesting part is what happens when the generated SQL is wrong —
# the agent catches the error, reads the schema, and tries to fix it automatically.

import os
import sqlite3

# Try to use LangChain's SQL utilities — if they're not installed, we fall back
# to a simple sqlite3 wrapper that exposes the same interface.
try:
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_core.messages import SystemMessage, HumanMessage
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    SQLDatabase = SQLDatabaseToolkit = SystemMessage = HumanMessage = None

# Plug in your preferred LLM here — OpenAI, Gemini, Anthropic, local Ollama, whatever.
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "Chinook_Sqlite.sqlite")


# ---------------------------------------------------------------------------
# Lightweight fallback for when LangChain isn't installed
# ---------------------------------------------------------------------------
class SimpleSQLiteDB:
    """
    A thin wrapper around Python's built-in sqlite3 module that mimics
    the parts of LangChain's SQLDatabase we actually use.
    Useful for testing the agent logic without needing the full LangChain stack.
    """

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        print(f"  Connected directly via sqlite3: {db_path}")

    def get_usable_table_names(self):
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cur.fetchall()]

    def get_table_info(self):
        # Return enough schema info so the LLM can understand what it's querying
        tables = self.get_usable_table_names()
        lines = []
        cur = self.conn.cursor()
        for table in tables:
            cur.execute(f"PRAGMA table_info({table});")
            cols = [f"{c[1]} {c[2]}" for c in cur.fetchall()]
            lines.append(f"Table {table}({', '.join(cols)})")
        return "\n".join(lines)

    def run(self, query):
        cur = self.conn.cursor()
        cur.execute(query)
        if query.strip().upper().startswith("SELECT"):
            return cur.fetchall()
        self.conn.commit()
        return "OK"


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------
def connect_to_database(db_path):
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Run the download script to get Chinook_Sqlite.sqlite."
        )

    if HAS_LANGCHAIN:
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        print(f"  Connected via LangChain SQLDatabase")
    else:
        print("  LangChain not installed — using the built-in sqlite3 fallback")
        db = SimpleSQLiteDB(db_path)

    tables = db.get_usable_table_names()
    print(f"  Tables available: {tables}\n")
    return db


# ---------------------------------------------------------------------------
# LLM + tools setup
# ---------------------------------------------------------------------------
def setup_agent(db, openai_api_key=None):
    """
    Wire up the LLM and SQL tools.
    If you have an OpenAI key, pass it in and you get real AI-generated SQL.
    Without one, we fall back to a dummy LLM that always returns a canned query
    (useful for testing the retry/correction loop without an API bill).
    """
    if HAS_LANGCHAIN:
        if ChatOpenAI and openai_api_key:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
            print("  Using GPT-4o-mini as the SQL generation model")
        else:
            # Fake LLM — returns fixed responses in sequence, great for testing
            from langchain_core.language_models.fake import FakeListLLM
            llm = FakeListLLM(responses=["SELECT * FROM Album LIMIT 5;"])
            print("  Using FakeListLLM (no API key provided — set openai_api_key to use a real model)")

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        print(f"  Loaded {len(tools)} SQL tools from toolkit\n")
    else:
        # Without LangChain we simulate the tool list for display purposes
        llm = None
        tools = [{"name": "query_executor", "description": "Runs SQL against the Chinook database"}]
        print(f"  Running with {len(tools)} fallback tool(s)\n")

    return llm, tools


# ---------------------------------------------------------------------------
# Self-correction logic
# ---------------------------------------------------------------------------

# This is the prompt we send to the LLM when a query fails.
# The key insight: we give it the original question, the broken query,
# the exact error message, and the full schema — that's usually enough to fix it.
CORRECTION_PROMPT = """You are a SQLite expert and debugger.
A SQL query failed with an error. Using the database schema below, write a corrected query.
Return ONLY the raw SQL — no explanations, no markdown code fences, just the query.

Schema:
{schema}

Original question: {question}
Broken query: {bad_query}
Error: {error}
"""

def ask_llm_to_fix_query(llm, question, bad_query, error_msg, schema):
    """
    Send the failed query + error context to the LLM and ask for a fix.
    If no real LLM is available, we do a simple string substitution as a heuristic.
    """
    if llm and HAS_LANGCHAIN and hasattr(llm, "invoke"):
        prompt = CORRECTION_PROMPT.format(
            schema=schema,
            question=question,
            bad_query=bad_query,
            error=error_msg,
        )
        response = llm.invoke([
            SystemMessage(content="You fix broken SQLite queries."),
            HumanMessage(content=prompt),
        ])
        fixed = getattr(response, "content", str(response))

        # Strip markdown fences if the model wrapped it anyway
        if "```sql" in fixed:
            fixed = fixed.split("```sql")[1].split("```")[0]
        elif "```" in fixed:
            fixed = fixed.split("```")[1].split("```")[0]
        return fixed.strip()

    # Heuristic fallback: common pluralization mistakes in Chinook
    # (the tables are singular: Track, Album, Artist, Customer, etc.)
    print("  [Heuristic fix] Trying common singular/plural corrections...")
    fixes = {"Tracks": "Track", "Albums": "Album", "Artists": "Artist", "Customers": "Customer"}
    fixed = bad_query
    for wrong, right in fixes.items():
        fixed = fixed.replace(wrong, right)
    return fixed


def run_query(db, sql, question="", llm=None, max_attempts=3):
    """
    Run a SQL query. If it fails, ask the LLM to fix it and retry.
    Returns a dict with the final status, query, result, and attempt count.
    """
    current_sql = sql

    for attempt in range(1, max_attempts + 1):
        print(f"  Attempt {attempt}: {current_sql}")
        try:
            result = db.run(current_sql)
            print(f"  -> Success!\n")
            return {"status": "ok", "query": current_sql, "result": result, "attempts": attempt}

        except Exception as err:
            error_str = str(err)
            print(f"  -> Error: {error_str}")

            if attempt == max_attempts:
                print("  -> Exhausted retries. Giving up.")
                return {"status": "failed", "query": current_sql, "error": error_str, "attempts": attempt}

            # Pull the schema so the LLM knows exactly what tables/columns exist
            schema = db.get_table_info()
            print("  -> Sending error + schema to correction model...")
            current_sql = ask_llm_to_fix_query(llm, question, current_sql, error_str, schema)
            print(f"  -> Proposed fix: {current_sql}")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def main():
    print("=== Text-to-SQL Agent: Chinook Database Demo ===\n")

    db = connect_to_database(DB_PATH)
    llm, tools = setup_agent(db)

    # --- Demo 1: a correct query (should pass on the first try) ---
    q1 = "How many tracks are in the catalog?"
    sql1 = "SELECT COUNT(*) AS total_tracks FROM Track;"
    print(f"Question: '{q1}'")
    result1 = run_query(db, sql1, question=q1, llm=llm)
    print(f"Result: {result1['result']}\n")

    # --- Demo 2: an intentionally broken query (tests the correction loop) ---
    # "Tracks" doesn't exist — the real table is "Track" (singular).
    # Watch the agent catch the error and self-correct.
    q2 = "How many tracks are in the catalog?"
    sql2 = "SELECT COUNT(*) AS total_tracks FROM Tracks;"   # <-- deliberate typo
    print(f"Question: '{q2}' (with a broken SQL to demonstrate self-correction)")
    result2 = run_query(db, sql2, question=q2, llm=llm)
    print(f"Final outcome: {result2}\n")

    # --- Demo 3: a more interesting business query ---
    q3 = "Which genre has the most tracks?"
    sql3 = "SELECT g.Name, COUNT(t.TrackId) AS track_count FROM Genre g JOIN Track t ON g.GenreId = t.GenreId GROUP BY g.Name ORDER BY track_count DESC LIMIT 5;"
    print(f"Question: '{q3}'")
    result3 = run_query(db, sql3, question=q3, llm=llm)
    print(f"Top 5 genres by track count: {result3['result']}\n")


if __name__ == "__main__":
    main()
