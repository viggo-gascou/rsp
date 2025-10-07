"""Research Project Module."""

import logging

from dotenv import load_dotenv
from termcolor import colored

fmt = (
    colored("[%(asctime)s ⋅ %(levelname)s]", "light_blue")
    + " ⋅ "
    + colored("%(message)s", "green")
)
logging.basicConfig(level=logging.WARNING, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


load_dotenv()
