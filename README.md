# merge-safetensors

A feature-rich command-line tool to merge split `.safetensors` model shards into a single `.safetensors` file.  
Ideal for Hugging Face models that distribute weights across multiple files (e.g. `model-00001-of-00017.safetensors`).

---

## 🛠 What It Does

Given a folder containing:
- Split `.safetensors` files (e.g. `model-00001-of-00017.safetensors`, etc.)
- The corresponding `model.safetensors.index.json` file (must be downloaded with the model)

This tool:
- Loads and merges the tensors based on the index
- Lets you name the output file or defaults to `model-merged.safetensors`
- Displays a real-time spinner during save
- Logs the process with timestamps and optional verbosity
- Detects missing keys or shard mismatches
- Accepts command-line arguments for scripting or automation

---

## ⚙️ Requirements

Python 3.8+  
Install dependencies with:

```
pip install safetensors numpy colorama
```

---

## 🚀 How to Use

Place the script or install the package in the same folder as your model shards. Then run:

```
python merge_safetensors.py
```

Or, use argument flags:

```
python merge_safetensors.py path/to/model.safetensors.index.json -o llama8b.safetensors
```

---

## 🔧 Command Line Options

```
usage: merge_safetensors.py [index_file] [-o OUTPUT] [-v] [--log-file LOG_FILE]

positional arguments:
  index_file           Path to the index file (default: model.safetensors.index.json)

options:
  -o, --output         Output filename (.safetensors will be added if missing)
  -v, --verbose        Enable detailed debug logging
  --log-file           Path to write log output (in addition to console)
```

If no output filename is provided, you’ll be prompted interactively.

---

## 📂 Example Folder Layout

```
my-model/
├── model.safetensors.index.json
├── model-00001-of-00017.safetensors
├── ...
├── model-00017-of-00017.safetensors
└── merge-safetensors.py
```

---

## 📦 Optional: Install as a CLI Tool

To make this script usable anywhere:

```
pip install .
```

Then you can simply run:

```
merge-safetensors
```

---

## 🆓 License

**Do whatever you want with it.**  
This project is provided with no license restrictions.
