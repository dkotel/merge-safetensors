# merge_safetensors.py

import json
import time
import threading
import sys
import argparse
import itertools
import os
import logging # Import the logging module
from typing import Dict, List, Optional, Any

# --- Dependency Imports with Error Handling ---

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    # Logging might not be configured yet, so print directly to stderr
    print("[ERROR] `safetensors` or `torch` not installed. Please install them to use this script.", file=sys.stderr)
    print("Suggestion: pip install safetensors torch", file=sys.stderr)
    sys.exit(1)

# Colorama is now primarily just for the spinner's ANSI codes on Windows
try:
    import colorama
    colorama.init(autoreset=True)
except ImportError:
    # Define dummy init/deinit if colorama is not available
    class DummyColorama:
        def init(self, autoreset=False): pass
        def deinit(self): pass
    colorama = DummyColorama()


# --- Basic Logging Configuration ---
# Configure logging at the module level or early in main()
# This sets up timestamped logging to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout # Send INFO level logs to stdout by default
)

# --- Spinner Thread ---

def spinner_running(stop_event: threading.Event, message: str = "Saving..."):
    """Displays a simple spinner in the console (writes directly to stdout)."""
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        try:
            # Spinner writes directly, bypasses logging formatting
            sys.stdout.write(f"\r{message} {next(spinner)}\033[K")
            sys.stdout.flush()
        except Exception:
             break # Avoid crashing thread if stdout writing fails
        time.sleep(0.1)
    # Clear the line upon stopping
    try:
        sys.stdout.write(f"\r{message} Done!{' ' * 10}\033[K\n")
        sys.stdout.flush()
    except Exception:
        print() # Fallback newline if stdout writing fails


# --- Core Logic Functions ---

def load_index(index_path: str) -> Dict[str, Any]:
    """Loads the JSON index file."""
    logging.info(f"Loading index file: {index_path}")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except FileNotFoundError:
        logging.error(f"Index file not found: {index_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in index file: {index_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to read index file {index_path}: {e}")
        sys.exit(1)

    if "weight_map" not in index:
        logging.error(f"'weight_map' key not found in index file: {index_path}")
        sys.exit(1)

    return index


def group_keys_by_shard(weight_map: Dict[str, str], index_dir: str) -> Dict[str, List[str]]:
    """Groups tensor keys by their shard filenames, resolving paths relative to the index file."""
    shards_to_load: Dict[str, List[str]] = {}
    logging.debug("Grouping tensor keys by shard file...")
    for key, shard_file_relative in weight_map.items():
        shard_file_absolute = os.path.abspath(os.path.join(index_dir, shard_file_relative))
        if shard_file_absolute not in shards_to_load:
            shards_to_load[shard_file_absolute] = []
        shards_to_load[shard_file_absolute].append(key)
        logging.debug(f"  Mapping key '{key}' to shard '{shard_file_absolute}'")
    logging.debug(f"Found {len(shards_to_load)} unique shard files to load.")
    return shards_to_load


def load_tensors(shards_to_load: Dict[str, List[str]], verbose: bool = False) -> Dict[str, Any]:
    """Loads tensors from shards, only loading required keys."""
    merged_tensors: Dict[str, Any] = {}
    num_shards = len(shards_to_load)
    logging.info(f"Starting to load {num_shards} shard(s)...")
    load_start = time.perf_counter()

    sorted_shard_files = sorted(shards_to_load.keys())

    for i, shard_file in enumerate(sorted_shard_files, 1):
        keys_in_shard = shards_to_load[shard_file]
        num_keys = len(keys_in_shard)
        logging.info(f"[{i}/{num_shards}] Loading {num_keys} tensor(s) from: {os.path.basename(shard_file)}")

        try:
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                shard_keys_available = f.keys() # Get actual keys in shard once
                for key in keys_in_shard:
                    if key not in shard_keys_available:
                        logging.warning(f"Key '{key}' specified in index not found in shard '{os.path.basename(shard_file)}'. Skipping.")
                        continue
                    if verbose:
                        # Use debug level for high verbosity, requires setting basicConfig level=logging.DEBUG
                        logging.debug(f"  - Reading tensor: {key}")
                    try:
                        merged_tensors[key] = f.get_tensor(key)
                    except Exception as e:
                        logging.warning(f"Could not read tensor '{key}' from shard '{os.path.basename(shard_file)}': {e}")

        except FileNotFoundError:
            logging.error(f"Shard file not found: {shard_file}. Aborting.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to open or read {shard_file}: {e}. Aborting.")
            sys.exit(1)

    load_end = time.perf_counter()
    logging.info(f"Finished loading tensors in {load_end - load_start:.2f} seconds.")
    return merged_tensors


def get_output_filename(output_arg: Optional[str]) -> str:
    """Determines the output filename, prompting if necessary."""
    if output_arg:
        output_name = output_arg
        logging.info(f"Using specified output filename: {output_name}")
    else:
        # Prompt user for output filename - direct input/output, not logging
        try:
            prompt_message = "\nEnter output filename (leave blank for 'model-merged.safetensors'): "
            user_input = input(prompt_message).strip()
            output_name = user_input if user_input else "model-merged.safetensors"
            if not user_input:
                 logging.info("No output filename provided. Using default: model-merged.safetensors")

        except EOFError: # Handle cases like piping input or Ctrl+D
             logging.info("No output filename provided via prompt. Using default: model-merged.safetensors")
             output_name = "model-merged.safetensors"
        except KeyboardInterrupt:
             logging.warning("\nUser interrupted filename prompt. Exiting.")
             sys.exit(130)


    # Ensure the filename ends with .safetensors
    if not output_name.lower().endswith(".safetensors"):
        logging.info(f"Appending '.safetensors' to output filename: {output_name}.safetensors")
        output_name += ".safetensors"

    return output_name


def save_merged_file(tensors: Dict[str, Any], output_filename: str):
    """Saves the merged tensors to a file with spinner feedback."""
    logging.info(f"Preparing to save merged model to: {output_filename} (this may take a moment...)")

    stop_spinner_event = threading.Event()
    # Use a message that makes sense without the spinner chars if redirecting output
    spinner_message = f"Saving to {os.path.basename(output_filename)}..."
    spinner_thread = threading.Thread(target=spinner_running, args=(stop_spinner_event, spinner_message), daemon=True)
    spinner_thread.start()

    save_start = time.perf_counter()
    try:
        save_file(tensors, output_filename)
        # Wait for spinner to finish its 'Done!' message *before* logging success
        stop_spinner_event.set()
        spinner_thread.join(timeout=2.0) # Wait a bit longer for spinner thread cleanup
    except Exception as e:
        # Ensure spinner stops before logging the error
        if not stop_spinner_event.is_set():
             stop_spinner_event.set()
             spinner_thread.join(timeout=1.0)
        logging.error(f"Failed to save merged file to {output_filename}: {e}")
        sys.exit(1)
    # No finally needed here as join() happens in try block on success

    save_end = time.perf_counter()
    save_duration = save_end - save_start
    logging.info(f"Successfully saved merged model in {save_duration:.2f} seconds.")
    return save_duration


# --- Main Execution ---

def main():
    """Main function to parse arguments and orchestrate the merge."""
    parser = argparse.ArgumentParser(
        description="Merge Safetensors shards into a single file based on an index file.",
        epilog="Example: python merge_safetensors.py model.safetensors.index.json -o merged_model.safetensors"
    )
    parser.add_argument(
        "index_file",
        nargs='?',
        default="model.safetensors.index.json",
        help="Path to the *.safetensors.index.json file (default: model.safetensors.index.json)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path for the merged output *.safetensors file. If not provided, you will be prompted."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging."
    )
    parser.add_argument(
        "--log-file",
        help="Optional path to a file to write logs to (in addition to console)."
    )
    args = parser.parse_args()

    # --- Add Simple Title Banner ---
    # Print the title *before* configuring/using logging for this specific output
    print("======================================")
    print("=        Merge Safetensors         =") # Updated Title
    print("======================================")
    print() # Add a blank line for spacing
    # -------------------------------

    # --- Reconfigure logging if needed (verbose or file) ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_handlers = [logging.StreamHandler(sys.stdout)] # Default: console handler

    if args.log_file:
        # Use direct print here as logging might not be ready for file handler yet
        print(f"[INFO] Attempting to log also to file: {args.log_file}")
        try:
            file_handler = logging.FileHandler(args.log_file, mode='w') # Overwrite log file each run
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            log_handlers.append(file_handler)
        except Exception as e:
            print(f"[WARN] Could not open log file {args.log_file} for writing: {e}", file=sys.stderr)

    # Apply the new configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=log_handlers, # Use the configured handlers
        force=True # Allow reconfiguring basicConfig
    )
    # --------------------------------------------------------

    start_time = time.perf_counter()
    logging.info("Merge script started.") # First log message will appear AFTER the title
    logging.debug(f"Arguments received: {args}")

    # Resolve the index file path and its directory
    try:
        index_path_abs = os.path.abspath(args.index_file)
        index_dir = os.path.dirname(index_path_abs)
        if not os.path.exists(index_path_abs):
             logging.error(f"Index file path does not exist: {index_path_abs}")
             sys.exit(1)
    except Exception as e:
         logging.error(f"Invalid index file path provided: {args.index_file} - {e}")
         sys.exit(1)

    # 1. Load Index and Prepare Shard Info
    index_data = load_index(index_path_abs)
    shards_to_load = group_keys_by_shard(index_data["weight_map"], index_dir)

    # 2. Load Tensors
    merged_tensors = load_tensors(shards_to_load, args.verbose) # Pass verbose flag

    if not merged_tensors:
       logging.error("No tensors were loaded. Check the index file, shard availability, and paths relative to the index.")
       sys.exit(1)
    logging.info(f"Successfully loaded {len(merged_tensors)} unique tensor keys.")

    # 3. Determine Output Filename
    output_filename = get_output_filename(args.output)
    try:
         output_filename_abs = os.path.abspath(output_filename)
         output_dir = os.path.dirname(output_filename_abs)
         if output_dir:
             logging.debug(f"Ensuring output directory exists: {output_dir}")
             os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
         logging.error(f"Invalid output path or could not create directory: {output_filename} - {e}")
         sys.exit(1)

    # Warn if output is same as an input shard
    if output_filename_abs in shards_to_load:
       logging.warning(f"Output filename '{output_filename}' is the same as one of the input shards. This will overwrite the shard.")

    # 4. Save Merged File
    save_duration = save_merged_file(merged_tensors, output_filename_abs)

    # Final Timing Report
    total_end_time = time.perf_counter()
    logging.info(f"Merge script finished. Total time: {total_end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # Let sys.exit pass through without logging an exception
        pass
    except KeyboardInterrupt:
        # Use logging for interrupt message if possible
        logging.warning("\nProcess interrupted by user.")
        # Ensure colorama is cleaned up even on interrupt
        colorama.deinit()
        sys.exit(130)
    except Exception as e:
        # Log critical errors using logging
        logging.critical(f"An unexpected critical error occurred: {e}", exc_info=True) # exc_info=True adds traceback
        colorama.deinit()
        sys.exit(1)
    finally:
        # Deinitialize colorama - restores original terminal state on Windows
        colorama.deinit()
        logging.shutdown() # Ensure all logging handlers are closed properly