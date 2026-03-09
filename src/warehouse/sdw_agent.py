import json
from pathlib import Path
from typing import Any, Dict, List

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from .sdw_engine import SDWEngine


class WarehouseAgent:
    def __init__(self, model_name: str = "gpt-4o", api_key: str = None):
        self.engine = SDWEngine()
        self.features_metadata = self._load_features()

        # Setup the LLM
        self.llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0)

        # Setup tools
        self.tools = [
            Tool(
                name="duckdb_sql",
                func=self.engine.query,
                description=f"Execute DuckDB SQL queries against the microstructure warehouse. Available views: {', '.join(self.engine.list_views())}.",
            ),
            Tool(
                name="get_table_schema",
                func=self.engine.schema,
                description="Get the schema (columns and types) for a specific dataset view.",
            ),
        ]

        self.agent_executor = self._setup_agent()

    def _load_features(self) -> str:
        path = Path(__file__).parent / "features.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                # Summarize for the prompt to avoid token bloat
                summary = []
                for item in data[:50]:  # First 50 for now
                    summary.append(f"- {item['name']}: {item['category']}")
                return "\n".join(summary)
        return "No feature metadata available."

    def _setup_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a Quantitative Data Warehouse Agent for Makalu.
You help traders analyze market microstructure data using DuckDB SQL.

Available Tables:
- tradeaggression: Taker buy/sell pressure, VWAP, imbalance.
- bookstate: Orderbook imbalance, microprice, spread.
- bookchange: Orderbook dynamics (pulls, refills).
- leadlag: Lead-lag correlation between exchanges.

Feature Metadata (Snippets):
{self.features_metadata}

Guidelines:
1. Always use `duckdb_sql` to fetch data.
2. If the user asks for a metric you don't know, check the schema or the feature metadata.
3. Be concise and provide technical insights based on the data.
4. Parquet files are virtualized as hive-partitioned tables. You can filter by 'exchange', 'symbol', and 'date'.
""",
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def ask(self, query: str):
        return self.agent_executor.invoke({{"input": query}})


if __name__ == "__main__":
    # This will fail without an API key, but serves as the entry point
    # agent = WarehouseAgent()
    # print(agent.ask("What is the average imbalance_qty_1s for XRP_USDT on 2025-07-01?"))
    print("WarehouseAgent initialized. (API key required for full functionality)")
    # print(agent.ask("What is the average imbalance_qty_1s for XRP_USDT on 2025-07-01?"))
    print("WarehouseAgent initialized. (API key required for full functionality)")
