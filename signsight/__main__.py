import sys

# Must use python 3.12
if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
    print(f"error: python {sys.version} in use, python 3.12 is required")
    sys.exit(1)

from signsight.train import train_model


def main() -> None:
    """Entry point for the SignSight CLI."""

    train_model()


if __name__ == "__main__":
    main()
