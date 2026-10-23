from __future__ import annotations

from makalu import settings as sys_env

from makalu.backtest_runner import BacktestRunner, _extract_asset, _resolve_order_size


def register_cli(subparsers) -> None:
    setup_p = subparsers.add_parser("setup-backtest-env", help="Set up backtest env files for Himalaya")
    setup_p.add_argument("--strategy", required=True, help="Strategy name, e.g. orion")
    setup_p.add_argument("--trading-market", required=True, help="e.g. GMO:BTC_JPY")
    setup_p.add_argument("--from-dt", required=True, help="Start date YYYYMMDD")
    setup_p.add_argument("--to-dt", required=True, help="End date YYYYMMDD")
    setup_p.add_argument("--variant", default=None, help="Override default variant")
    setup_p.add_argument("--env", default="BENJAMIN_TATUM", help="Target env name")
    setup_p.set_defaults(func=run_setup)

    publish_p = subparsers.add_parser("publish-results", help="Publish backtest run-logs to Kilimanjaro viewer")
    publish_p.add_argument("--env", default="BENJAMIN_TATUM", help="Env name whose run_log/ to publish")
    publish_p.set_defaults(func=run_publish)


def run_setup(args) -> None:
    runner = BacktestRunner(str(sys_env.get_env_root_dir()))
    k = runner._knowledge.get(args.strategy)
    if k is None:
        print(f"Unknown strategy '{args.strategy}'. Known: {list(runner._knowledge)}")
        return

    asset = _extract_asset(args.trading_market)
    signal_market = k["asset_signal_map"].get(asset)
    if signal_market is None:
        print(f"No signal market mapping for asset '{asset}' in strategy '{args.strategy}'.")
        return

    variant = args.variant or k["default_variant"]
    order_size = _resolve_order_size(k, asset)
    runner._write_env(
        env_name=args.env,
        strategy=args.strategy,
        variant=variant,
        trading_market=args.trading_market,
        signal_market=signal_market,
        order_size=order_size,
        magnifier=k["magnifier"],
        param_grid=k["param_grid"],
    )
    print(f"Env '{args.env}' ready: strategy={args.strategy} variant={variant}")
    print(f"  trading={args.trading_market}  signal={signal_market}")
    print("Next steps:")
    print(
        f"  /panda/potential-panda/Himalaya/start.sh --env {args.env}"
        f" --from-dt {args.from_dt} --to-dt {args.to_dt} --optimize"
    )
    print(f"  python /home/lulurun/workspace/Makalu/cli.py publish-results --env {args.env}")


def run_publish(args) -> None:
    runner = BacktestRunner(str(sys_env.get_env_root_dir()))
    count = runner._publish_run_logs(args.env)
    print(f"Published {count} run-log(s) from env '{args.env}'")
