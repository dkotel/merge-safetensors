# merge-safetensors

A simple command-line tool to merge split `.safetensors` model shards into a single `.safetensors` file.  
Useful for Hugging Face models that distribute weights across multiple files (e.g. `model-00001-of-00017.safetensors`).

---

## 🛠 What It Does

Given a folder containing:
- Split `.safetensors` files (e.g. `model-00001-of-00017.safetensors`, etc.)
- The corresponding `model.safetensors.index.json` file (must be downloaded with the model)

This tool merges everything into one file called:

```
model-merged.safetensors
```

---

## ⚙️ Requirements

Python 3.8+  
Install dependencies with:

```bash
pip install safetensors numpy
```

---

## 🚀 How to Use

1. Clone this repository or copy the script into the folder with your model shards
2. Ensure the folder contains **all split `.safetensors` files** **and** the `model.safetensors.index.json`
3. Run:

```
merge-safetensors
```

That’s it. It will show loading progress, then spin while saving.

---

## 📦 Installation (optional)

To make it pip-installable:

```
pip install .
```

Then you can run `merge-safetensors` from anywhere.

---

## 📂 Example Folder Layout

```
my-model/
├── model.safetensors.index.json
├── model-00001-of-00017.safetensors
├── ...
├── model-00017-of-00017.safetensors
└── merge-safetensors (script or CLI)
```

---

## 🆓 License

**Do whatever you want with it.**  
This project is provided with no license restrictions.
