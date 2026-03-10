#!/usr/bin/env python3
"""
Interactive research chat for the data warehouse.

Usage:
    python dw_chat.py                        # uses gpt-4o by default
    python dw_chat.py --model gpt-4o-mini    # cheaper, faster
    python dw_chat.py --verbose              # print SQL as it runs

Commands during chat:
    quit / exit    end the session
    reset          start a new conversation (clear history)
    datasets       list available datasets
"""

import argparse
import sys

from qate.env import sys_env

from dw import Agent, Engine, OpenAIBackend


def main():
    parser = argparse.ArgumentParser(description="Data warehouse research chat")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--verbose", action="store_true", help="Print SQL queries as they run")
    args = parser.parse_args()

    api_key = sys_env.get_ai()
    engine = Engine()
    datasets = engine.list()
    print(f"  {len(datasets)} datasets registered\n")

    backend = OpenAIBackend(api_key=api_key, model=args.model)
    agent = Agent(engine, backend)

    print(f"Model: {args.model}")
    print("Type your question. Commands: reset | datasets | quit\n")
    print("-" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("Bye.")
            break

        if question.lower() == "reset":
            agent.reset()
            print("Conversation reset.")
            continue

        if question.lower() == "datasets":
            print(datasets.to_string())
            continue

        print("\nAgent: ", end="", flush=True)
        response = agent.ask(question, verbose=args.verbose)
        print(response)


if __name__ == "__main__":
    main()
