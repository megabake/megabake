"""Measure the fusion budget against an actual Inductor MLP on this GPU."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/tmp/megabake-design-research/inductor')
import torch
from probe import samples


def mlp(x, gate, up, down):
    return (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(456)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_stream(torch.cuda.Stream())
    from torch._inductor.utils import is_big_gpu
    report = {'inductor_gemm_autotune_eligible': is_big_gpu(0), 'torch': torch.__version__, 'gpu': str(torch.cuda.get_device_properties(0)),
              'mode': 'max-autotune-no-cudagraphs, then externally captured for GPU timing',
              'seed': 456, 'cache': 'repeated buffers', 'records': []}
    compiled = torch.compile(mlp, fullgraph=True, mode='max-autotune-no-cudagraphs')
    for k, n in [(576, 1536), (4096, 11008)]:
        for m in [1, 256, 1024]:
            x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16) / (k ** 0.5)
            gate, up = [torch.randn((n, k), device='cuda', dtype=torch.bfloat16) / (k ** 0.5) for _ in range(2)]
            down = torch.randn((k, n), device='cuda', dtype=torch.bfloat16) / (n ** 0.5)
            ref = mlp(x, gate, up, down)
            row = {'m': m, 'n': n, 'k': k}
            for name, fn in [('eager_graph', mlp), ('inductor_graph', compiled)]:
                start = time.perf_counter()
                result = fn(x, gate, up, down)
                torch.cuda.synchronize()
                setup = time.perf_counter() - start
                error = (result.float() - ref.float()).norm() / ref.float().norm()
                assert error.item() < 0.01, error.item()
                assert torch.isfinite(result).all().item()
                row[name] = samples(lambda: fn(x, gate, up, down), 32)
                row[name].update(setup_seconds=setup, relative_l2_error=error.item())
            # Library GEMMs with actual boundary shapes. This sum is a diagnostic,
            # not a lower bound or an equivalent standalone end-to-end workload.
            intermediate = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            output = torch.empty_like(x)
            components = [(x, gate, intermediate), (x, up, intermediate),
                          (intermediate, down, output)]
            # Initialize every input before diagnostic timing.
            intermediate.normal_()
            row['library_gemm_sum_us'] = sum(
                samples(lambda a=a, w=w, c=c: torch.mm(a, w.T, out=c), 32)['median_us']
                for a, w, c in components)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                compiled(x, gate, up, down)
                torch.cuda.synchronize()
            trace_path = Path('/tmp/megabake-design-research/mlp-trace.json')
            prof.export_chrome_trace(str(trace_path))
            trace = json.loads(trace_path.read_text())
            row['inductor_kernel_names'] = [e['name'] for e in trace['traceEvents'] if e.get('cat') == 'kernel']
            report['records'].append(row)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'m':m, 'n':n, 'k':k, 'eager_us':row['eager_graph']['median_us'],
                              'inductor_us':row['inductor_graph']['median_us'],
                              'gemm_sum_us':row['library_gemm_sum_us'],
                              'kernel_count':len(row['inductor_kernel_names'])}), flush=True)


if __name__ == '__main__':
    main()
