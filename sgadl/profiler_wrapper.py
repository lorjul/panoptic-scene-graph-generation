import torch


def get_profiler(wait: int, warmup: int, active: int, repeat: int, log_dir):
    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=acts,
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )
