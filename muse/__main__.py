import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
for _noisy in ("pdfminer", "urllib3", "httpcore", "httpx", "hpack"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
