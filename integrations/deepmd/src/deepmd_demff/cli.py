import argparse

from .manifest import register


def main():
    parser = argparse.ArgumentParser(description="Register original DEM-FF weights for DeePMD Python inference")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("register")
    command.add_argument("model", help="Trusted original DEMFF.model checkpoint")
    command.add_argument("output", help="New .demff manifest (weights are not copied or converted)")
    export = sub.add_parser("export", help="Export native DeePMD DEM-FF checkpoint to MACE-LES plus manifest")
    export.add_argument("checkpoint")
    export.add_argument("output")
    args = parser.parse_args()
    if args.command == "export":
        from .export import export_checkpoint
        print(export_checkpoint(args.checkpoint,args.output))
    else:
        print(register(args.model, args.output))
