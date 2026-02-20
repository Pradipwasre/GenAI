# Ollama Beginner Guide

##  What is Ollama?
Ollama is a tool that lets you **run Large Language Models (LLMs) locally** on your computer.  
Instead of connecting to cloud services like ChatGPT or Claude, Ollama allows you to download and run open‑source models directly on your machine.

---

##  Why Use Ollama?
- **Privacy** → Your data stays on your computer.  
- **Cost‑effective** → No API tokens or subscription fees.  
- **Control** → Choose which models to run and experiment freely.  
- **Offline capability** → Once downloaded, models can run without internet.  

👉 Think of Ollama as *owning the car instead of renting a taxi*—you decide where to drive.

---

## 🔗 Official Resources
- [Download Ollama](https://ollama.com)  
- [Ollama Documentation](https://ollama.com/docs)  
- [Ollama Model Library](https://ollama.com/library)  

---

## ⚙️ Installation
1. Go to [ollama.com](https://ollama.com) and download for your OS (Windows, macOS, Linux).  
2. Install and open your **command prompt/terminal**.  
3. Run your first model:

```bash
ollama run llama2


Common Commands

# Run a model
ollama run llama2

# List all downloaded models
ollama list

# Pull (download) a new model
ollama pull gemma

# Stop a running model
Ctrl + C


Example Prompt : Once a model is running, type directly into the terminal:

> Explain AI to me like I’m 10 years old


## 📂 Where Models Are Stored
On **Windows**:

- Go to `C:\Users\<YourName>`
- Enable **View Hidden Files**
- Look for the folder: `.ollama`
- Inside, you’ll find all downloaded models.

---

## 🧩 Model Categories
Ollama supports many models. Here are some popular ones:

| Category        | Models Available                | Use Case                          |
|-----------------|---------------------------------|-----------------------------------|
| **General LLMs** | LLaMA 3.1, LLaMA 3.2, Mistral   | Text generation, Q&A              |
| **Reasoning**   | DeepSeek‑R1, Qwen               | Logical tasks, step‑by‑step reasoning |
| **Coding**      | Claude Code, Codex, OpenCode    | Programming help, code generation |
| **Embeddings**  | Nomic‑embed‑text                | Search, semantic similarity       |
| **Vision**      | Gemma Vision                    | Image + text tasks                |

👉 Explore the full library here: [Ollama Model Library](https://ollama.com/library)
