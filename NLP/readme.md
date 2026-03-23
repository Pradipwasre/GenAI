# How to Install TensorFlow with Python 3.11 (Windows)

Follow these steps one by one. Don’t skip!  
Think of it like building Lego blocks — each step adds a piece.

---

## 🧩 Step 1: Check Python Versions
Open **PowerShell** and type:

```powershell
py --list
You should see something like:

 -V:3.14   Python 3.14
 -V:3.12   Python 3.12
 -V:3.11   Python 3.11


If you see 3.11, you’re ready. If not, install Python 3.11 from 
Link : https://www.python.org/ftp/python/3.11.4/python-3.11.4-amd64.exe

## Step 2: Make a Virtual Environment

cd C:\Users\admin\Desktop\GenAI 
py -3.11 -m venv nlp_env
nlp_env\Scripts\activate

-- Now your prompt will show (nlp_env) — that means you’re inside the playground.

## Step 3: Install TensorFlow

python -m pip install --upgrade pip
python -m pip install tensorflow

-- This puts TensorFlow inside your playground.

## Step 4: Connect to Jupyter

python -m pip install ipykernel
python -m ipykernel install --user --name=nlp_env --display-name "Python 3.11 (nlp_env)"


## Step 5: Switch Kernel in Jupyter

1. Open Jupyter Notebook.
2. Go to Kernel → Change Kernel.
3.Pick Python 3.11 (nlp_env).

- Now you’re using the right Python version.