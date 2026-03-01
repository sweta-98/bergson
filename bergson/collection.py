from datasets import Dataset
from transformers import PreTrainedModel

from bergson.collector.collector import CollectorComputer
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import AttentionConfig, IndexConfig, PreprocessConfig, ReduceConfig
from bergson.gradients import GradientProcessor
from bergson.score.scorer import Scorer


def collect_gradients(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    cfg: IndexConfig,
    *,
    batches: list[list[int]] | None = None,
    target_modules: set[str] | None = None,
    attention_cfgs: dict[str, AttentionConfig] | None = None,
    scorer: Scorer | None = None,
    reduce_cfg: ReduceConfig | None = None,
    preprocess_cfg: PreprocessConfig | None = None,
):
    """
    Compute gradients using the hooks specified in the GradientCollector.
    """
    collector = GradientCollector(
        model=model.base_model,  # type: ignore
        cfg=cfg,
        processor=processor,
        target_modules=target_modules,
        data=data,
        scorer=scorer,
        reduce_cfg=reduce_cfg,
        preprocess_cfg=preprocess_cfg,
        attention_cfgs=attention_cfgs or {},
        filter_modules=cfg.filter_modules,
    )

    computer = CollectorComputer(
        model=model,  # type: ignore
        data=data,
        collector=collector,
        batches=batches,
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="New worker - Collecting gradients")
