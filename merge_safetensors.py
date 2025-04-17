# merge_safetensors/cli.py

import json
import time
import threading
import sys
from safetensors import safe_open
from safetensors.torch import save_file


# Spinner thread function for visual feedback during saving
def spinner_running(done_flag):
    spinner_chars = ['|', '/', '-', '\\']
    idx = 0
    while not done_flag["done"]:
        sys.stdout.write(f"\r[INFO] Saving... {spinner_chars[idx % len(spinner_chars)]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    sys.stdout.write("\r[INFO] Saving... Done!         \n")
    sys.stdout.flush()


def merge_safetensors():
    start_time = time.perf_counter()

    # Load the index file
    with open("model.safetensors.index.json", "r") as f:
        index = json.load(f)

    # Get shard filenames from the index
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    # Collect all tensors from all shards
    merged_tensors = {}

    print(f"[INFO] Starting to load {len(shard_files)} shard(s)...")
    load_start = time.perf_counter()

    for shard_file in shard_files:
        print(f"Loading: {shard_file}")
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                print(f"  - Reading tensor: {key}")
                merged_tensors[key] = f.get_tensor(key)

    load_end = time.perf_counter()
    print(f"[INFO] Finished loading tensors in {load_end - load_start:.2f} seconds.")

    # Prompt user for output filename
    user_input = input("\nEnter output filename (leave blank for 'model-merged.safetensors'): ").strip()
    if not user_input:
        output_name = "model-merged.safetensors"
    else:
        if not user_input.endswith(".safetensors"):
            user_input += ".safetensors"
        output_name = user_input

    print(f"\n[INFO] Saving merged model to: {output_name} (this may take a moment...)")
    done = {"done": False}
    spinner_thread = threading.Thread(target=spinner_running, args=(done,))
    spinner_thread.start()

    save_start = time.perf_counter()
    save_file(merged_tensors, output_name)
    save_end = time.perf_counter()

    done["done"] = True
    spinner_thread.join()

    # Final timing report
    print(f"[INFO] Saved merged model in {save_end - save_start:.2f} seconds.")
    print(f"[DONE] Total time: {save_end - start_time:.2f} seconds.")
