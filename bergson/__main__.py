import os
import sys
from dataclasses import dataclass
from typing import Union, get_args

from simple_parsing import ArgumentParser, ConflictResolution, Serializable

from bergson.hessians.pipeline import hessian_pipeline

from .build import build
from .config import (
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    MixConfig,
    PreprocessConfig,
    QueryConfig,
    ScoreConfig,
    TrackstarConfig,
    TrainingConfig,
    ValidationConfig,
)
from .diagnose import DiagnoseConfig, diagnose
from .hessians.hessian_approximations import approximate_hessians
from .magic import MagicConfig, run_magic
from .process_grads import mix_autocorrelation_matrices
from .query.query_index import query
from .score.score import score_dataset
from .trackstar import trackstar
from .utils.worker_utils import validate_run_path
from .yaml_pipeline import run_pipeline


@dataclass
class Build(Serializable):
    """Build a gradient index."""

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    def execute(self):
        """Build the gradient index."""
        if self.index_cfg.skip_index and self.index_cfg.skip_hessians:
            raise ValueError("Either skip_index or skip_hessians must be False")

        validate_run_path(self.index_cfg)

        build(self.index_cfg, self.preprocess_cfg)


@dataclass
class Ekfac(Serializable):
    """Run the full EKFAC influence pipeline end-to-end."""

    index_cfg: IndexConfig

    hessian_cfg: HessianConfig

    score_cfg: ScoreConfig

    preprocess_cfg: PreprocessConfig

    hessian_pipeline_cfg: HessianPipelineConfig

    def execute(self):
        hessian_pipeline(
            self.index_cfg,
            self.hessian_cfg,
            self.score_cfg,
            self.preprocess_cfg,
            self.hessian_pipeline_cfg,
        )


@dataclass
class Hessian(Serializable):
    """Approximate Hessian matrices using KFAC or EKFAC.

    Output layout:

        <run_path>/<method>/
            hessian_config.yaml
            index_config.yaml
            ...method-specific artifacts...
    """

    hessian_cfg: HessianConfig
    index_cfg: IndexConfig

    def execute(self):
        """Compute Hessian approximation."""

        validate_run_path(self.index_cfg)

        if self.hessian_cfg.method == "autocorrelation":
            self.index_cfg.run_path = (
                self.index_cfg.run_path + f"/{self.hessian_cfg.method}"
            )
            self.index_cfg.skip_index = True
            self.index_cfg.skip_hessians = False
            build(self.index_cfg, PreprocessConfig())
        else:
            approximate_hessians(self.index_cfg, self.hessian_cfg)


@dataclass
class Magic(MagicConfig):
    """Run MAGIC attribution."""

    def execute(self):
        """Run MAGIC attribution."""
        run_magic(self)


@dataclass
class Mix(MixConfig):
    """Mix two autocorrelation hessians into a single GradientProcessor.

    Loads autocorrelation hessians from ``query_path`` and ``index_path``,
    computes a mixing coefficient via the §A.1.3 procedure of Chang et al.
    (2024), and writes the mixed GradientProcessor to ``output_path``.
    """

    def execute(self):
        if not self.query_path or not self.index_path or not self.output_path:
            raise ValueError(
                "mix requires --query_path, --index_path, "
                "and --output_path to be set."
            )
        mix_autocorrelation_matrices(
            query_path=self.query_path,
            index_path=self.index_path,
            output_path=self.output_path,
            target_downweight_components=self.target_downweight_components,
        )


@dataclass
class Query(QueryConfig):
    """Query an existing gradient index."""

    def execute(self):
        """Query an existing gradient index."""
        query(self)


@dataclass
class Reduce(Serializable):
    """Reduce a gradient index."""

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    def execute(self):
        """Reduce a gradient index."""
        if self.index_cfg.projection_dim != 0:
            print(f"Using a projection dimension of {self.index_cfg.projection_dim}. ")

        validate_run_path(self.index_cfg)
        build(self.index_cfg, self.preprocess_cfg)


@dataclass
class Score(Serializable):
    """Score a dataset against an existing gradient index."""

    score_cfg: ScoreConfig

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    def execute(self):
        """Score a dataset against an existing gradient index."""
        assert self.score_cfg.query_path

        if self.index_cfg.projection_dim != 0:
            print(f"Using a projection dimension of {self.index_cfg.projection_dim}. ")

        validate_run_path(self.index_cfg)
        score_dataset(self.index_cfg, self.score_cfg, self.preprocess_cfg)


@dataclass
class Trackstar(Serializable):
    """Run hessians, build, and score as a single pipeline."""

    index_cfg: IndexConfig

    trackstar_cfg: TrackstarConfig

    def execute(self):
        trackstar(self.index_cfg, self.trackstar_cfg)


@dataclass
class Train(TrainingConfig):
    """Train a model with the MAGIC trainer, but don't actually run MAGIC."""

    def execute(self):
        """Train the model."""
        run_magic(self)


@dataclass
class Test_Model_Configuration:
    """Test gradient consistency across padding and batch composition.

    Tests whether a model produces consistent gradients regardless of how
    documents are batched together. If inconsistencies are found, recommends
    using --force_math_sdp on build/score/trackstar commands."""

    diagnose_cfg: DiagnoseConfig

    def execute(self):
        """Run the diagnostic."""
        diagnose(self.diagnose_cfg)


@dataclass
class Validate(ValidationConfig):
    """Run leave-k-out validation of attribution scores."""

    scores: str = ""
    """Path to saved attribution scores for validation."""

    def execute(self):
        """Run the validation."""
        assert self.scores, "Path to attribution scores must be provided."
        run_magic(self, score_path=self.scores)


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[
        Build,
        Ekfac,
        Hessian,
        Magic,
        Mix,
        Query,
        Reduce,
        Score,
        Trackstar,
        Train,
        Test_Model_Configuration,
        Validate,
    ]

    def execute(self):
        """Run the script."""
        self.command.execute()


def main():
    """Parse CLI arguments and dispatch to the selected subcommand.

    Three input shapes are supported:
      `bergson <command> --flag value ...`  — single-command CLI-flag mode
      `bergson pipeline <file.yaml>`        — multi-step pipeline mode
      `bergson <command> <file.yaml|json>`  — single-command config-file mode

    """
    args = sys.argv[1:]

    # Build the {command_name: command_class} lookup once; both the pipeline
    # branch and the single-command branch use it to resolve the user's verb.
    command_classes = get_args(Main.__dataclass_fields__["command"].type)
    command_registry = {cls.__name__.lower(): cls for cls in command_classes}

    # Pipeline mode: a YAML listing several commands to run in
    # sequence. Delegates parsing + execution to `run_pipeline`.
    if len(args) == 2 and args[0].lower() == "pipeline" and os.path.isfile(args[-1]):
        run_pipeline(args[1], command_registry)
        return

    # Single-command config-file mode: the user passes a command
    # name and a path to a YAML or JSON file.
    if len(args) == 2 and os.path.isfile(args[-1]):
        cmd_str, config_path = args
        try:
            cmd_cls = command_registry[cmd_str.lower()]
        except KeyError as e:
            raise ValueError(
                f"Invalid command '{cmd_str}'. "
                f"Valid commands are: {list(command_registry)}"
            ) from e

        try:
            prog = cmd_cls.load(config_path)
        except RuntimeError as e:
            print(f"Failed to load config file {config_path}: {e}")
            sys.exit(1)

    # CLI-flag mode: standard argparse-style flag parsing.
    else:
        parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
        parser.add_arguments(Main, dest="prog")
        prog: Main = parser.parse_args().prog

    prog.execute()


if __name__ == "__main__":
    main()
