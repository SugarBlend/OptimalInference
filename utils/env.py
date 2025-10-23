from pathlib import Path


def get_project_root():
    return Path(__file__).parents[1].as_posix()
