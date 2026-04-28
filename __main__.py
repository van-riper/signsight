import sys

if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
    print("error: signsight requires python 3.12")
    print(
        f"detected version: {sys.version_info.major}.{sys.version_info.minor}"
    )
    sys.exit(1)


def main() -> None:
    pass


if __name__ == "__main__":
    main()
