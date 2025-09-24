import json
import os


def gather_results(base_dir: str, output_file: str, filter_str: str|None=None):
    results = {}
    for dir in os.listdir(base_dir):
        if filter_str is None or filter_str in dir:
            results[dir] = json.load(open(os.path.join(base_dir, dir, "sab_results.json")))
    
    with open(output_file, "w") as f:
        json.dump(results, f)
    
    print(f"Results saved to {output_file}")

    scores = [v[0] for v in results.values()]
    average_scores = [sum(s) / len(s) for s in zip(*scores)]
    print(average_scores)

    latencies = [v[1]['median'] for v in results.values()]
    average_latency = sum(latencies) / len(latencies)
    print(average_latency)


if __name__ == "__main__":
    import fire
    fire.Fire(gather_results)