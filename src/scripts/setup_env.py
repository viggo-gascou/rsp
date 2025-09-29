from pathlib import Path

from dotenv import dotenv_values


def setup_env():
    # setup environment by adding any missing SSH keys to authorized_keys
    ssh_keys = Path("~/.ssh/authorized_keys").expanduser().resolve()
    if ssh_keys.exists():
        existing_keys = set(ssh_keys.read_text().strip().splitlines())
    else:
        existing_keys = set()

    for key, value in dotenv_values().items():
        if key.startswith("SSH") and value and value.strip() not in existing_keys:
            with ssh_keys.open("a") as f:
                f.write(value.strip() + "\n")


if __name__ == "__main__":
    setup_env()
