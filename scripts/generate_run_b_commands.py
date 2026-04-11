
import sys

def generate_commands():
    base_cmd = "python3 scripts/run_experiment.py"
    output_dir = "analysis/run/simulation-distrub-local/results"
    num_queries = 500
    
    # Qwen model family for CAG research (CPU-only vLLM)
    # Start with smaller models for iteration, scale up for final benchmarks
    models = [
        "Qwen/Qwen3-4B",           # ~8GB RAM, 256K context - fast iteration
        "Qwen/Qwen3-8B",           # ~16GB RAM, 40K context - standard benchmark
        "Qwen/Qwen2.5-7B-Instruct", # ~20GB RAM, 128K context - Phase 4 (fits 24GB)
    ]
    datasets = ["squad_v2", "trivia_qa"]
    workloads = ["single", "batched", "multi_turn"]
    
    commands = []
    
    for model in models:
        for dataset in datasets:
            # Dataset specific args
            ds_args = f"--dataset {dataset}"
            if dataset == "trivia_qa":
                ds_args += " --truncate-prompt-tokens 1024 --max-context-chars 300 --max-context-docs 2"
            
            for workload in workloads:
                common = (
                    f"{base_cmd} --model {model} {ds_args} --workload-mode {workload} "
                    f"--num-queries {num_queries} --output-dir {output_dir} --max-tokens 256"
                )

                commands.append(f"{common} --baseline no_cache")
                commands.append(f"{common} --baseline prefix_cache")
                commands.append(f"{common} --baseline rag --top-k-sweep --top-k-values 1,3,5,10")
                commands.append(
                    f"{common} --baseline redis --baseline-label redis_retrieval_cache_cold "
                    "--flush-redis-namespace --redis-key-prefix redis_retrieval_cache_cold "
                    "--top-k-sweep --top-k-values 1,3,5,10"
                )
                commands.append(
                    f"{common} --baseline hybrid --baseline-label hybrid_retrieval_cache_cold "
                    "--flush-redis-namespace --redis-key-prefix hybrid_retrieval_cache_cold "
                    "--top-k-sweep --top-k-values 1,3,5,10"
                )
                commands.append(
                    f"{common} --baseline hybrid --baseline-label hybrid_retrieval_cache_warm "
                    f"--flush-redis-namespace --redis-key-prefix hybrid_retrieval_cache_warm "
                    f"--warmup-queries {num_queries} --top-k-sweep --top-k-values 1,3,5,10"
                )
                commands.append(
                    f"{common} --baseline distributed --baseline-label distributed_router_replicated "
                    "--api-base http://localhost:8000 --sharding-policy replicated"
                )
    
    print("\n".join(commands))

if __name__ == "__main__":
    generate_commands()
