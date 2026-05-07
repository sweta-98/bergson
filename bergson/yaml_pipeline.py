"""Run a sequence of bergson commands defined in a YAML file."""

from typing import Any

import yaml


def parse_pipeline(
    config_path: str, command_registry: dict[str, type]
) -> list[tuple[str, Any]]:
    """Read a pipeline YAML and hydrate each step into a command instance.

    The YAML must be a list of single-key mappings. Each entry names a
    registered command and contains that command's full, self-contained
    config (the same shape that command accepts as a single-command YAML):

        - hessian:
            hessian_cfg:
              method: kfac
            index_cfg:
              run_path: tests/build_path
        - build:
            index_cfg:
              run_path: tests/build_path
            preprocess_cfg: {}

    All steps are parsed up-front so config errors fail fast, before any
    expensive step runs.
    """
    with open(config_path) as f:
        steps = yaml.safe_load(f)

    if not isinstance(steps, list):
        raise ValueError(
            f"Pipeline YAML at {config_path} must be a list of step entries; "
            f"got a top-level {type(steps).__name__}."
        )

    parsed: list[tuple[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or len(step) != 1:
            raise ValueError(
                f"Pipeline step {i} must be a single-key mapping "
                f"(e.g. `- hessian: {{...}}`); got {step!r}."
            )
        ((cmd_name, cmd_dict),) = step.items()
        try:
            cmd_cls = command_registry[cmd_name.lower()]
        except KeyError:
            raise ValueError(
                f"Pipeline step {i}: unknown command '{cmd_name}'. "
                f"Valid commands: {sorted(command_registry)}."
            ) from None
        parsed.append((cmd_name, cmd_cls.from_dict(cmd_dict or {})))

    return parsed


def run_pipeline(config_path: str, command_registry: dict[str, type]) -> None:
    """Parse a pipeline YAML and execute every step in order."""
    steps = parse_pipeline(config_path, command_registry)
    total = len(steps)
    for i, (cmd_name, cmd) in enumerate(steps, start=1):
        print(f"\n[pipeline] step {i}/{total}: {cmd_name}")
        cmd.execute()
