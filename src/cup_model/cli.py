import argparse
import json
from pathlib import Path

from cup_model.contracts import Result, Snapshot, timestamp
from cup_model.demo import generate
from cup_model.evaluation import backtest
from cup_model.features import build_race
from cup_model.pipeline import Forecaster, train
from cup_model.simulation import Simulation, simulate
from cup_model.store import Dataset, Store


def parser():
    root = argparse.ArgumentParser(prog="cup-model")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="generate explicitly synthetic pipeline inputs")
    demo.add_argument("--output", required=True)
    demo.add_argument("--days", type=int, default=40)
    demo.add_argument("--seed", type=int, default=42)
    schema = commands.add_parser("schema", help="print input record JSON schemas")
    schema.add_argument("--output")
    ingest = commands.add_parser("ingest", help="atomically import timestamped JSONL records")
    ingest.add_argument("--db", required=True)
    ingest.add_argument("--input", required=True)
    fitting = commands.add_parser("train", help="fit base, ensemble and calibration windows")
    fitting.add_argument("--db", required=True)
    fitting.add_argument("--frozen-at", required=True, type=timestamp)
    fitting.add_argument("--output", required=True)
    audit = commands.add_parser("backtest", help="run a retrospective race-day walk-forward audit")
    audit.add_argument("--db", required=True)
    audit.add_argument("--start", required=True, type=timestamp)
    audit.add_argument("--score-as-of", required=True, type=timestamp)
    audit.add_argument("--output", required=True)
    for command in (fitting, audit):
        command.add_argument("--trees", action="store_true")
        command.add_argument("--tree-iterations", type=int, default=500)
        command.add_argument("--l2", type=float, default=1)
        command.add_argument("--seed", type=int, default=42)
    prediction = commands.add_parser("predict", help="select one runner and append the ledger")
    prediction.add_argument("--db", required=True)
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--race-id", required=True)
    prediction.add_argument("--cutoff", type=timestamp)
    prediction.add_argument("--output", required=True)
    simulation = commands.add_parser("simulate", help="run experimental scenario sensitivity")
    simulation.add_argument("--input", required=True)
    simulation.add_argument("--output", required=True)
    return root


def write_output(data, path=None):
    text = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path:
        Path(path).write_text(text)
    else:
        print(text, end="")


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    store = None
    try:
        if args.command == "demo":
            write_output(generate(args.output, args.days, args.seed))
            return
        if args.command == "schema":
            write_output(
                {"snapshot": Snapshot.model_json_schema(), "result": Result.model_json_schema()},
                args.output,
            )
            return
        if args.command == "simulate":
            config = Simulation.model_validate_json(Path(args.input).read_text())
            write_output(simulate(config), args.output)
            return
        store = Store(args.db)
        if args.command == "ingest":
            write_output({"imported": store.import_jsonl(args.input)})
            return
        dataset = Dataset(store.records())
        if args.command == "train":
            model = train(
                dataset,
                args.frozen_at,
                trees=args.trees,
                l2=args.l2,
                seed=args.seed,
                tree_iterations=args.tree_iterations,
            )
            model.save(args.output)
            write_output(
                {"model_id": model.manifest["model_id"], "synthetic": model.manifest["synthetic"]}
            )
        elif args.command == "predict":
            snapshot = dataset.snapshot(args.race_id, args.cutoff)
            card = Forecaster.load(args.model).predict(build_race(dataset, snapshot))
            card = store.save_prediction(card)
            write_output(card, args.output)
            write_output(card["selection"])
        elif args.command == "backtest":
            report = backtest(
                dataset,
                args.start,
                args.score_as_of,
                trees=args.trees,
                l2=args.l2,
                seed=args.seed,
                tree_iterations=args.tree_iterations,
            )
            write_output(report, args.output)
            write_output(
                {
                    "races": report["race_count"],
                    "summary": report["summary"],
                    "synthetic": report["synthetic"],
                }
            )
    except (ValueError, OSError, ImportError) as error:
        root.error(str(error))
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    main()
