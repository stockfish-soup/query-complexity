"""List or export the installed research workspace."""
import argparse
from . import export, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List bundled files')
    command = sub.add_parser('export', help='Export to a new or empty directory')
    command.add_argument('directory')
    args = parser.parse_args()
    if args.command == 'list':
        print('\n'.join(manifest()))
    else:
        print(export(args.directory))


if __name__ == '__main__':
    main()
