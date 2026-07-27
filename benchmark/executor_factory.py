"""Select the validation/test execution backend without changing agent logic."""

from benchmark.executor import BenchmarkExecutor


def make_executor(*args, eval_backend=None, **kwargs):
    if eval_backend is None:
        config = args[0] if args else kwargs.get("config", {})
        eval_backend = config.get("_execution_backend", {}).get("name", "local")
    if eval_backend == "ssh":
        from benchmark.ssh_executor import SSHExecutor

        return SSHExecutor(*args, **kwargs)
    if eval_backend != "local":
        raise ValueError(
            f"Unknown eval_backend {eval_backend!r}; expected 'local' or 'ssh'."
        )
    return BenchmarkExecutor(*args, **kwargs)
