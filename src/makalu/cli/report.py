import argparse
import json
import sys
from datetime import date, timedelta
from logging import getLogger
from pathlib import Path

from qate.core.model import Market

LOG = getLogger(__name__)



def _parse_period(text: str) -> list[str]:
    """'Q4 2025:2025-10-01:2025-12-31' -> ['Q4 2025', '2025-10-01', '2025-12-31']"""
    name, start, end = text.rsplit(":", 2)
    date.fromisoformat(start), date.fromisoformat(end)
    return [name, start, end]


def _comparisons(values: list[str] | None) -> list[dict] | None:
    if not values:
        return None
    out = []
    for i, v in enumerate(values):
        try:
            a, b = v.split(",")
            out.append({"id": f"c{i}", "a": _parse_period(a), "b": _parse_period(b)})
        except ValueError:
            raise SystemExit(f"--compare must look like 'NAME:FROM:TO,NAME:FROM:TO', got {v!r}")
    return out


def execute_leadlag(args):
    from makalu.report import leadlag, site
    from pipeline.market_pair import EXECUTION_SIZE, LEAD_LAG_PAIRS

    if args.pair and (args.lead_market or args.lag_market):
        LOG.error("--pair is mutually exclusive with --lead-market/--lag-market.")
        sys.exit(1)
    if args.pair:
        if args.pair not in LEAD_LAG_PAIRS:
            LOG.error(f"Unknown pair '{args.pair}'. Known: {list(LEAD_LAG_PAIRS)}")
            sys.exit(1)
        lead, lag = LEAD_LAG_PAIRS[args.pair]
    elif args.lead_market and args.lag_market:
        lead, lag = Market.from_str(args.lead_market), Market.from_str(args.lag_market)
    else:
        LOG.error("Pass either --pair, or both --lead-market and --lag-market.")
        sys.exit(1)

    execution_size = args.execution_size or EXECUTION_SIZE.get(lag.symbol)
    if execution_size is None:
        LOG.error(f"No execution_size known for {lag.id}; pass --execution-size.")
        sys.exit(1)

    pair = leadlag.Pair(lead, lag, str(execution_size))
    end = args.end or (date.today() - timedelta(days=2)).isoformat()
    start = args.start
    if start is None:
        days = leadlag.available_days(pair)
        if not days:
            LOG.error(f"No day has MarketPrice for both {lead.id} and {lag.id} at execution size {execution_size}.")
            sys.exit(1)
        start = days[0]
    site_dir = Path(args.site_dir) if args.site_dir else site.default_site_dir()
    meta = site.build_leadlag(
        pair, start, end, site_dir,
        comparisons=_comparisons(args.compare), recompute=args.recompute,
    )
    print(json.dumps({
        "status": "ok",
        "report": str(site_dir / "leadlag" / meta["key"]),
        "path": f"leadlag/{meta['key']}/",
        "days": meta["days"],
        "hours": meta["hours"],
        "first_date": meta["first_date"],
        "last_date": meta["last_date"],
    }))


def execute_serve(args):
    from makalu import web
    from makalu.report import site

    site_dir = Path(args.site_dir) if args.site_dir else site.default_site_dir()
    web.serve(site_dir, args.host, args.port)


def register_cli(sub):
    p = sub.add_parser(
        "report",
        help="Build static report pages from the warehouse (WRITE: site folder only)",
        description=(
            "Builds static HTML reports into the site folder (default ~/env/LUIGI_MARIO/site),\n"
            "served by 'serve'. Reads the warehouse; writes nothing to it.\n"
            "Each report keeps its numbers in daily.csv and hourly.csv and only computes days they do not have."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rsub = p.add_subparsers(dest="report", required=True)

    ll = rsub.add_parser(
        "leadlag",
        help="Daily lead-lag between two markets, from silver MarketPrice",
        description=(
            "Per JST day: cross-correlation of lag-market returns with lead-market returns\n"
            "at 100ms, 1s and 5s, peak lag, pass-through, and the market context\n"
            "(volatility, tick size, updates, spreads), and the same per JST hour.\n"
            "Needs MarketPrice for both markets."
        ),
        epilog=(
            "Examples:\n"
            "  python cli.py report leadlag --pair xrp            # all available days\n"
            "  python cli.py report leadlag --lead-market BINANCE:XRP_USDT --lag-market GMO:XRP_JPY \\\n"
            "      --start 2025-07-01 --end 2026-06-29 \\\n"
            "      --compare 'Q4 2025:2025-10-01:2025-12-31,Q2 2026:2026-04-01:2026-06-30'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ll.add_argument("--pair", help="Known pair key from pipeline.market_pair.LEAD_LAG_PAIRS: btc, eth, xrp, sol")
    ll.add_argument("--lead-market", help="Lead market as EXCHANGE:SYMBOL, e.g. BINANCE:XRP_USDT")
    ll.add_argument("--lag-market", help="Lag market as EXCHANGE:SYMBOL, e.g. GMO:XRP_JPY")
    ll.add_argument("--execution-size", help="MarketPrice execution size (default: by lag symbol)")
    ll.add_argument("--start", help="First date YYYY-MM-DD (default: first day both markets have MarketPrice)")
    ll.add_argument("--end", help="Last date YYYY-MM-DD (default today - 2)")
    ll.add_argument("--compare", action="append", metavar="'NAME:FROM:TO,NAME:FROM:TO'",
                    help="A pair of periods to compare; repeat for more. Default: Q4 2025 vs Q2 2026, and H2 2025 vs H1 2026")
    ll.add_argument("--recompute", action="store_true", help="Recompute every day instead of reusing daily.csv")
    ll.add_argument("--site-dir", help="Site folder (default ~/env/LUIGI_MARIO/site)")
    ll.set_defaults(func=execute_leadlag)

    s = sub.add_parser(
        "serve",
        help="Serve the report site as static files (read-only, long-running)",
        description=(
            "Serves the site folder over HTTP. Read-only: no API, no jobs.\n"
            "Binds 127.0.0.1:9776 by default, which the pisces-01 tunnel exposes as\n"
            "http://<public ip>:7700/makalu/. Run under systemd: deploy/makalu-web.service."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=9776)
    s.add_argument("--site-dir", help="Site folder (default ~/env/LUIGI_MARIO/site)")
    s.set_defaults(func=execute_serve)
