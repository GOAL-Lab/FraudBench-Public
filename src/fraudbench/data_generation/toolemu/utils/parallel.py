import concurrent.futures
import logging
import time
from multiprocessing.pool import ThreadPool

import tqdm

from .my_typing import *

logger = logging.getLogger(__name__)


def batchify(data: List[Any], batch_size: int):
    """Split data into batches."""
    return [data[i: i + batch_size] for i in range(0, len(data), batch_size)]


def thread_pool_executor(
        gen_func: Callable,
        batch_inputs: List[Any],
        unordered: bool = True,
        sequential_generation: bool = False,
        show_progress: bool = True,
        num_threads: int = 10,
        request_timeout: int = 60,
        enable_timer: bool = True,
        max_timeout_retries: int = 0,
        on_result_callback: Callable[[Any], None] = None,
) -> List[Any]:
    """Run a generation function in parallel using a thread pool executor.
    Modified from https://github.com/openai/evals/blob/main/evals/eval.py#L148
    
    Args:
        max_timeout_retries: Maximum number of times to retry on timeout (default 0 = no retries).
        on_result_callback: Optional callback function called immediately when each result completes.
            This allows processing/saving results as they finish rather than waiting for the batch.
    """

    def worker_thread(inputs):
        """
        Worker thread for evaluating a single sample.
        
        Returns the result from gen_func, or a (failed_item, error_dict) tuple on timeout/error
        to allow the retry logic to handle it gracefully.
        """
        retries = 0
        while True:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(gen_func, inputs)
            try:
                result = future.result(timeout=request_timeout)
                return result
            except concurrent.futures.TimeoutError as e:
                executor.shutdown(wait=False)
                retries += 1
                if retries > max_timeout_retries:
                    logger.error(f"Task timed out after {request_timeout}s (attempt {retries}/{max_timeout_retries + 1})")
                    # Return a failed result tuple instead of raising, so batch processing continues
                    # This matches the (failed_item, outputs) format expected by FuncExecutorWithRetry
                    return (inputs, {"error": f"Timeout after {request_timeout}s ({retries} attempts)"})
            except Exception as e:
                # Catch any other exception and return as failed result
                # This prevents a single failed task from crashing the entire batch
                executor.shutdown(wait=False)
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(f"Task failed with {error_type}: {error_msg}")
                return (inputs, {"error": f"{error_type}: {error_msg}"})

    start = time.time()
    results = []
    with ThreadPool(num_threads) as pool:
        if sequential_generation:
            logger.info("Running in sequential mode!")
            iter = map(gen_func, batch_inputs)
        else:
            logger.info(f"Running in threaded mode with {num_threads} threads!")
            if unordered:
                iter = pool.imap_unordered(worker_thread, batch_inputs)
            else:
                iter = pool.imap(worker_thread, batch_inputs)
        
        # Process results as they complete, calling callback immediately for each
        with tqdm.tqdm(total=len(batch_inputs), disable=not show_progress) as pbar:
            for result in iter:
                results.append(result)
                if on_result_callback is not None:
                    on_result_callback(result)
                pbar.update(1)
    
    if enable_timer:
        logger.info(f"Run {len(batch_inputs)} calls took {time.time() - start:.2f}s")
    return results
