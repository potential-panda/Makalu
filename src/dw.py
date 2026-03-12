def engine_usage():
    from dw import Engine

    engine = Engine()
    engine.list()  # all 13 datasets

    # Pull real data
    df = engine.read(
        "VolSpread",
        exchange="BINANCE",
        symbol="BTC_USDT",
        date_from="20260101",
        date_to="20260110",
    )

    print(df.head())

    df = engine.query(
        """
    SELECT date_trunc('minute', to_timestamp(ts)) AS minute,
    avg(spread_pct_ema), avg(volatility_ema)
    FROM volspread
    WHERE exchange = 'BINANCE' AND symbol = 'BTC_USDT'
    AND date BETWEEN '20260101' AND '20260110'
    GROUP BY minute ORDER BY minute
    """
    )

    print(df.head())


def ai_usage():
    from qate.env import sys_env

    from dw import Agent, Engine, OpenAIBackend

    api_key = sys_env.get_ai()
    engine = Engine()

    # OpenAI
    agent = Agent(engine, OpenAIBackend(api_key=api_key))

    # OR local Ollama (no key needed)
    agent = Agent(
        engine,
        OpenAIBackend(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="llama3.1",
        ),
    )

    # Ask
    print(
        agent.ask(
            "What is the average spread_pct_ema for BINANCE BTC_USDT in January 2026?",
            verbose=True,  # prints each SQL query as it runs
        )
    )
    )
