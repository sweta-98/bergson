Utilities
=========

Auto Batch Size Module
----------------------

Overview
^^^^^^^^

The ``auto_batch_size.py`` module finds the maximum viable ``token_batch_size`` value for the current hardware and run configuration.

In-Memory Benchmark Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from bergson.utils.auto_batch_size import determine_batch_size_in_memory, get_optimal_batch_size

    # With caching
    optimal_size = get_optimal_batch_size(
        cache_path=run_path / "batch_size_cache.json",
        model_hf_id="EleutherAI/pythia-70m",
        fsdp=False,
        starting_batch_size=16384,
        determine_fn=lambda: determine_batch_size_in_memory(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            max_length=1024,
            starting_batch_size=16384,
        ),
    )

Caching
^^^^^^^

The module automatically caches determined batch sizes to avoid redundant testing.

**Cache file format** (``batch_size_cache.json``):

.. code-block:: json

    {
      "model_name": "EleutherAI/pythia-70m",
      "token_batch_size": 8192,
      "fsdp": false,
      "gpu_name": "NVIDIA A100-SXM4-40GB",
      "gpu_memory_gb": 42.0
    }

**Cache validation**:

- Checks model name matches
- Checks FSDP setting matches
- Warns if GPU changed (doesn't invalidate, just warns)

**Cache location**:

- Stored in benchmark run directory
- Shared across multiple runs for same model/hardware

Algorithm
^^^^^^^^^

1. Start with ``starting_batch_size`` (default: 16384)
2. Test with current size:

   - **In-memory**: Run forward/backward pass with test dataset
   - **CLI**: Execute ``bergson build`` with small dataset

3. If successful:

   - Increase by 1.5x and retry
   - Continue until OOM

4. If OOM:

   - Reduce by ~30% (multiply by 0.7)
   - Retry until success

5. Round final successful size to power of 2
6. Save to cache

**Time complexity**: Typically 3-6 test iterations (1-3 minutes)

Adapted From
^^^^^^^^^^^^

Based on HuggingFace Accelerate's ``find_executable_batch_size``:

- https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/memory.py#L120

Adapted for Bergson's token-based batching system where:

.. code-block:: python

    token_batch_size = max(seq_length) * num_documents
