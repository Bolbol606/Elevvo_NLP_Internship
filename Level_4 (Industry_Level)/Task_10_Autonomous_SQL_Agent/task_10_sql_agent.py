"""
Task 10: Autonomous "Text-to-SQL" Agent (Chinook Database)
Elevvo NLP Internship - Industry Level

Prompt Requirements:
- Connect to SQLite Chinook database using LangChain
- Initialize tool-calling setup with a basic LLM
- Do NOT write the self-correction error loop yet
- Leave a specific placeholder block for try/except database execution logic and retry mechanism
"""

import os
import sqlite3

try:
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.tools import Tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    SQLDatabase = None
    SQLDatabaseToolkit = None
    SystemMessage = HumanMessage = Tool = None

# Optional LLM initialization
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "Chinook_Sqlite.sqlite")


class NativeSQLiteAdapter:
    """Lightweight SQLite adapter fallback when LangChain is not installed."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def get_usable_table_names(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]

    def get_table_info(self):
        tables = self.get_usable_table_names()
        info = []
        cursor = self.conn.cursor()
        for t in tables[:5]:
            cursor.execute(f"PRAGMA table_info({t});")
            cols = [f"{c[1]} ({c[2]})" for c in cursor.fetchall()]
            info.append(f"Table {t}: {', '.join(cols)}")
        return "\n".join(info)

    def run(self, query: str):
        cursor = self.conn.cursor()
        cursor.execute(query)
        if query.strip().upper().startswith("SELECT"):
            return cursor.fetchall()
        self.conn.commit()
        return "Success"


def initialize_sql_environment(db_path: str):
    """Initializes connection to SQLite database and inspects available schema."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please check the path.")

    if LANGCHAIN_AVAILABLE:
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        print(f"Connected via LangChain SQLDatabase: {db_path}")
    else:
        print("[Note] LangChain not detected. Using built-in NativeSQLiteAdapter fallback.")
        db = NativeSQLiteAdapter(db_path)

    print(f"Discovered Tables: {db.get_usable_table_names()}")
    return db


def setup_tools_and_agent(db, api_key: str = None):
    """
    Sets up SQL database tools and tool-calling LLM.
    """
    if LANGCHAIN_AVAILABLE:
        if ChatOpenAI and api_key:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        else:
            from langchain_core.language_models.fake import FakeListLLM
            llm = FakeListLLM(responses=["SELECT * FROM Album LIMIT 5;"])

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
    else:
        llm = None
        tools = [{"name": "sql_db_query", "description": "Execute SQL queries against Chinook"}]

    print(f"\nInitialized {len(tools)} SQL Agent Tools:")
    for t in tools:
        t_name = getattr(t, "name", t.get("name"))
        t_desc = getattr(t, "description", t.get("description", ""))
        print(f"  • {t_name}: {t_desc[:80]}...")

    return db, llm, tools
    print(f"\nInitialized {len(tools)} SQL Agent Tools:")
    for t in tools:
        print(f"  • {t.name}: {t.description[:80]}...")

    return db, llm, tools


# =========================================================================
# SELF-CORRECTION PROMPT TEMPLATE & AGENT RETRY LOOP (Completed)
# =========================================================================
SELF_CORRECTION_SYSTEM_PROMPT = """You are an expert SQL debugger.
A previously generated SQLite query resulted in an execution error.
Analyze the database schema, table definitions, and exact SQLite error message to produce a corrected, valid SQL query.
Return ONLY the executable SQL query enclosed in ```sql ... ``` without preamble or extra commentary.
"""

def generate_corrected_query(llm, user_question: str, failed_query: str, error_message: str, schema_info: str) -> str:
    """Invokes LLM with error context to synthesize a corrected SQL query."""
    user_prompt = f"""Target Database Schema:
{schema_info}

User Question: {user_question}
Failed SQL Query: {failed_query}
SQLite Execution Error: {error_message}

Please repair the SQL query so that it executes successfully against SQLite."""

    if llm and hasattr(llm, "invoke"):
        messages = [
            SystemMessage(content=SELF_CORRECTION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response))
        # Strip markdown syntax if present
        if "```sql" in content:
            content = content.split("```sql")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content.strip()

    # Rule-based fallback fix for demonstration (e.g., typos in common column/table names)
    print("[Self-Correction Heuristic Fallback applied]")
    repaired = failed_query.replace("Tracks", "Track").replace("Albums", "Album").replace("Customers", "Customer")
    return repaired


def execute_query_with_self_correction_hook(
    db: SQLDatabase,
    sql_query: str,
    user_question: str = "Query Execution",
    llm=None,
    max_retries: int = 3
):
    """
    Executes SQL queries with autonomous self-correction loop. Catches execution errors,
    consults schema context, prompts LLM for correction, and retries automatically.
    """
    current_query = sql_query

    for attempt in range(1, max_retries + 1):
        print(f"\n[Attempt {attempt}/{max_retries}] Executing: {current_query}")
        try:
            result = db.run(current_query)
            print(" Execution successful!")
            return {
                "status": "success",
                "final_query": current_query,
                "result": result,
                "attempts": attempt
            }
        except Exception as e:
            error_msg = str(e)
            print(f" Execution Error: {error_msg}")

            if attempt == max_retries:
                print(" Reached maximum retry attempts. Raising final error.")
                return {
                    "status": "failed",
                    "final_query": current_query,
                    "error": error_msg,
                    "attempts": attempt
                }

            print(" Triggering Autonomous Self-Correction Loop...")
            schema_info = db.get_table_info()
            current_query = generate_corrected_query(
                llm=llm,
                user_question=user_question,
                failed_query=current_query,
                error_message=error_msg,
                schema_info=schema_info
            )
            print(f" Self-Correction proposed new query: {current_query}")


def main():
    db = initialize_sql_environment(DB_PATH)
    db, llm, tools = setup_tools_and_agent(db)

    # Demonstrate Autonomous Self-Correction:
    # Notice 'Tracks' has a plural typo (the SQLite Chinook table is 'Track')
    erroneous_sql = "SELECT COUNT(*) AS total_tracks FROM Tracks;"
    sample_question = "How many total tracks are in the Chinook catalog?"
    
    print("\n--- Demonstrating Autonomous Error Recovery & Self-Correction ---")
    outcome = execute_query_with_self_correction_hook(
        db=db,
        sql_query=erroneous_sql,
        user_question=sample_question,
        llm=llm,
        max_retries=3
    )
    print("\nFinal Result:\n", outcome)


if __name__ == "__main__":
    main()
