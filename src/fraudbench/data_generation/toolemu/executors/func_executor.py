import logging
import threading
from argparse import ArgumentParser, Namespace

import tqdm

from fraudbench.data_generation.toolemu.utils import append_jsonl, batchify, thread_pool_executor
from fraudbench.data_generation.toolemu.utils.my_typing import *

logger = logging.getLogger(__name__)

# Lock for thread-safe file writes
_file_write_lock = threading.Lock()


class BaseFuncExecutor:
    args_list = [
        "batch_size",
        "num_threads",
        "threading_request_timeout",
        "sequential_generation",
        "disable_batch_progress",
        "disable_timer",
    ]

    def __init__(
            self,
            batch_size: int = 5,
            num_threads: int = None,
            threading_request_timeout: int = 3000,
            target_num: int = None,
            unordered: bool = False,
            sequential_generation: bool = False,
            disable_batch_progress: bool = False,
            disable_timer: bool = False,
            multiple_results: bool = False,
            verbose: bool = True,
    ):
        self._batch_size = batch_size
        # Default num_threads to batch_size if not specified
        self._num_threads = num_threads if num_threads is not None else batch_size
        self._threading_request_timeout = threading_request_timeout
        self._target_num = target_num
        self._unordered = unordered
        self._sequential_generation = sequential_generation
        self._disable_batch_progress = disable_batch_progress
        self._disable_timer = disable_timer
        self._multiple_results = multiple_results
        self._verbose = verbose

    def _write_to_file(self, output_file: str, results: Any):
        """Write the results to the output file (thread-safe)."""
        with _file_write_lock:
            if self._multiple_results:
                for res in results:
                    append_jsonl(output_file, res)
            else:
                append_jsonl(output_file, results)

    def _run_on_dataset(
            self,
            func: Callable,
            output_file: str,
            dataset: List[Any],
            result_parse_func: Callable = None,
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Run the function on the dataset."""
        batched_dataset = batchify(dataset, self._batch_size)
        succ_results = []
        remaining_dataset = []
        failed_results = []

        total = self._target_num or len(dataset)
        current = 0
        
        # Track success count across callback invocations
        success_count_lock = threading.Lock()
        batch_success_count = [0]  # Use list to allow mutation in nested function
        
        def on_result_complete(res):
            """Callback called immediately when each result completes - saves to file right away."""
            nonlocal succ_results, remaining_dataset, failed_results
            
            if result_parse_func is not None:
                failed_item, result = result_parse_func(res)
            else:
                failed_item, result = res
            
            with success_count_lock:
                if failed_item is None:  # the function execution is successful
                    self._write_to_file(output_file, result)
                    succ_results.append(result)
                    batch_success_count[0] += 1
                    if self._verbose:
                        logger.info(f"Saved successful result to {output_file} (total: {len(succ_results)})")
                else:  # the function execution failed, to be retried
                    remaining_dataset.append(failed_item)
                    failed_results.append(result)
        
        with tqdm.tqdm(total=total) as pbar:
            for batch_inputs in batched_dataset:
                batch_success_count[0] = 0
                
                if self._batch_size <= 1:  # directly call the function
                    results = [func(batch_inputs[0])]
                    # Process single result through callback
                    for res in results:
                        on_result_complete(res)
                else:
                    # Use callback for immediate saving as results complete
                    results = thread_pool_executor(
                        func,
                        batch_inputs,
                        unordered=self._unordered,
                        sequential_generation=self._sequential_generation,
                        show_progress=not self._disable_batch_progress,
                        num_threads=self._num_threads,
                        request_timeout=self._threading_request_timeout,
                        enable_timer=not self._disable_timer,
                        on_result_callback=on_result_complete,
                    )

                if self._verbose:
                    logger.info(
                        f"Batch complete: {batch_success_count[0]} successful "
                        f"(out of {len(batch_inputs)}) to {output_file}."
                    )
                if self._target_num is None:
                    pbar.update(len(batch_inputs))
                else:
                    pbar.update(batch_success_count[0])
                    current += batch_success_count[0]
                    if current >= self._target_num:
                        break
        return succ_results, remaining_dataset, failed_results

    def run(
            self,
            func: Callable,
            output_file: str,
            dataset: List[Any] = None,
            result_parse_func: Callable = None,
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        if dataset is None:
            raise ValueError("dataset cannot be None.")
        return self._run_on_dataset(
            func, output_file, dataset, result_parse_func=result_parse_func
        )

    @classmethod
    def register_args(
            cls,
            parser: ArgumentParser,
            prefix: str = None,
            shortprefix: str = None,
            default_batch_size: int = 5,
            default_num_threads: int = 1,
            default_timeout: int = 3000,
    ):
        """Register arguments for the function executor."""
        if prefix is None:
            prefix = ""
            shortprefix = ""
        else:
            prefix += "-"
            shortprefix = shortprefix or prefix[0]
        parser.add_argument(
            f"--{prefix}batch-size",
            f"-{shortprefix}bs",
            type=int,
            default=default_batch_size,
            help="The number of items to process per batch",
        )
        parser.add_argument(
            f"--{prefix}num-threads",
            f"-{shortprefix}nt",
            type=int,
            default=default_num_threads,
            help="The number of threads for parallel execution (defaults to batch_size)",
        )
        parser.add_argument(
            f"--{prefix}threading-request-timeout",
            f"-{shortprefix}trt",
            type=int,
            default=default_timeout,
            help="Timeout for a single function execution request",
        )
        parser.add_argument(
            f"--{prefix}sequential-generation",
            f"-{shortprefix}sg",
            action="store_true",
            help="Run function sequentially over the dataset instead of in parallel",
        )
        parser.add_argument(
            f"--{prefix}disable-batch-progress",
            f"-{shortprefix}xprog",
            action="store_true",
            help="Disable progress bar for multi-threaded execution",
        )
        parser.add_argument(
            f"--{prefix}disable-timer",
            f"-{shortprefix}xtimer",
            action="store_true",
            help="Disable timer for multi-threaded execution",
        )

    @classmethod
    def from_args(
            cls,
            args: Namespace,
            prefix: str = None,
            **kwargs,
    ):
        """Create an executor from arguments."""
        if prefix is None:
            prefix = ""
        else:
            prefix += "_"
        args_dict = {}
        for name in cls.args_list:
            args_dict[name] = getattr(args, prefix + name)
        args_dict.update(kwargs)
        return cls(**args_dict)


class FuncExecutorWithRetry(BaseFuncExecutor):
    args_list = BaseFuncExecutor.args_list + ["num_retries"]

    def __init__(self, num_retries: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._num_retries = num_retries

    def run(
            self,
            func: Callable,
            output_file: str,
            dataset: List[Any] = None,
            result_parse_func: Callable = None,
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Run the function on the dataset."""
        if dataset is None:
            raise ValueError("dataset cannot be None.")
        for retries in range(self._num_retries + 1):
            if self._verbose:
                msg = "Executing the function" if retries == 0 else "Retrying"
                logger.info(f"{msg} for {len(dataset)} data points.")
            # time.sleep(10)  # Avoid exceeding the API rate limit.
            _, dataset, failed_results = self._run_on_dataset(
                func, output_file, dataset, result_parse_func
            )
            if len(dataset) == 0:
                break
            if self._verbose:
                logger.warning(
                    f"Failed to execute the function for {len(dataset)} "
                    f"data points at round {retries}.",
                )
                if "idx" in dataset[0]:
                    logger.warning(f"Failed indexes: {[item['idx'] for item in dataset]}")

        if len(failed_results):
            for result in failed_results:
                self._write_to_file(output_file, result)
            logger.info(
                f"Saved {len(failed_results)} failed results to the end of {output_file}."
            )

    @classmethod
    def register_args(
            cls,
            parser: ArgumentParser,
            prefix: str = None,
            shortprefix: str = None,
            default_num_retries: int = 0,
            **kwargs,
    ):
        """Register arguments for the function executor."""
        super().register_args(parser, prefix, shortprefix, **kwargs)
        if prefix is None:
            prefix = ""
            shortprefix = ""
        else:
            prefix += "-"
            shortprefix = shortprefix or prefix[0]
        parser.add_argument(
            f"--{prefix}num-retries",
            f"-{shortprefix}nr",
            type=int,
            default=default_num_retries,
            help=(
                "The number of retries when the function execution fails. "
                "Do not set a too large number as it may take a long time to finish."
            ),
        )


class GenFuncExecutor(BaseFuncExecutor):
    args_list = BaseFuncExecutor.args_list + ["target_num", "max_attempts"]

    def __init__(
            self,
            target_num: int,
            max_attempts: int = None,
            unordered: bool = True,
            **kwargs,
    ):
        super().__init__(target_num=target_num, unordered=unordered, **kwargs)
        self._max_attempts = max_attempts

    def run(
            self,
            func: Callable,
            output_file: str,
            dataset: List[Any] = None,
            result_parse_func: Callable = None,
    ):
        if dataset is None:
            if self._max_attempts is None:
                raise ValueError("dataset and max_attempts cannot both be None.")
            dataset = [i for i in range(self._max_attempts)]  # default is indexes
        return self._run_on_dataset(
            func, output_file, dataset, result_parse_func=result_parse_func
        )

    @classmethod
    def register_args(
            cls,
            parser: ArgumentParser,
            prefix: str = None,
            shortprefix: str = None,
            **kwargs,
    ):
        """Register arguments for the function executor."""
        super().register_args(parser, prefix, shortprefix, **kwargs)
        if prefix is None:
            prefix = ""
            shortprefix = ""
        else:
            prefix += "-"
            shortprefix = shortprefix or prefix[0]
        parser.add_argument(
            f"--{prefix}target-num",
            f"-{shortprefix}target",
            type=int,
            default=None,
            help="The target number of generated data points",
        )
        parser.add_argument(
            f"--{prefix}max-attempts",
            f"-{shortprefix}ma",
            type=int,
            default=None,
            help="The maximum number of attempts to generate data points",
        )
