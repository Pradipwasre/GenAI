# LangChain Classroom Notes
## Session 1 + 2: Models & Prompts
> Beginner-friendly notes for teaching in class with diagrams and code examples

---

# PART 1 : Models in LangChain

## 1.1 What Problem Does LangChain Solve?

Each AI company (OpenAI, Google, Anthropic) has its own way of communicating. LangChain acts like a **universal remote** : one consistent way to talk to all of them.

```
[ OpenAI  ]  ──┐
[ Anthropic] ──┼──▶  [ LangChain ]  ──▶  [ Your App ]
[ Google   ]  ──┘       (.invoke)
```

> **Board tip:** Draw three boxes on the left pointing into one box (LangChain), then one arrow to "Your App".

---

## 1.2 Two Families of Models

### Language Model
- **Text goes in → Text comes out**
- Used for: Chatbots, Q&A, Summarization, Translation, Coding

### Embedding Model
- **Text goes in → Numbers come out**
- Used for: Semantic Search, Finding Similar Documents, RAG Apps

```
Language Model:   "What is AI?"  →  [ Model ]  →  "AI stands for..."
Embedding Model:  "What is AI?"  →  [ Model ]  →  [0.23, 0.91, 0.44, ...]
```

> **Notepad sketch:** Two boxes side by side. Left: text in → text out. Right: text in → numbers out.

---

## 1.3 Inside Language Models : LLM vs Chat Model

Think of it like messaging:
- **LLM** = sending a one-time letter (no memory)
- **Chat Model** = WhatsApp conversation (remembers everything)

| Feature | LLM | Chat Model |
|---|---|---|
| Input | One text block | Full conversation |
| Output | One text block | Reply message |
| Memory | None | Remembers chat |
| Knows roles? | No | System / User / AI |
| Status today | Being retired ⚠️ | Use this one ✅ |

> Always use **Chat Models** for new projects. LLMs are slowly being phased out.

---

## 1.4 Temperature : The Creativity Dial

```
Low (near 0) ────────────────────────── High (1.5+)
  Same answer                            Creative,
  every time                             varied answers
  (safe, predictable)                    (good for stories)
```

- **Use low temperature** when you need consistent, repeatable answers
- **Use high temperature** when you want creative or diverse output
- Too high → random, nonsensical output

> **Board tip:** Draw a dial or slider from 0 to 2. Label left side "Robot-like" and right side "Creative."

---

## 1.5 Closed Source vs Open Source Models

```
CLOSED SOURCE (Paid)              OPEN SOURCE (Free)
─────────────────────             ──────────────────────
GPT-4, Claude, Gemini             LLaMA, Mistral, Falcon
Easy (just API key)               Needs GPU + setup
Data goes to their server         Data stays on your machine
Polished output                   Decent, improving fast
Pay per API call                  Completely free
```

> Open source models live on **Hugging Face** : the biggest library of free AI models.

**Two ways to use open source models:**
1. **Hugging Face API** : runs on their server, free up to a limit
2. **Download locally** : runs on your machine, totally free and private (needs strong GPU)

> **Best practice:** Always store your API key in a `.env` file. Never paste it directly in code.

---

# PART 2 : Prompts in LangChain

## 2.1 What is a Prompt?

A prompt is simply the message you send to the AI.

> Small change in prompt = Big change in output

This sensitivity is why "**Prompt Engineering**" is now a real job role.

Prompts can be: text, images, audio, or video, but we focus on **text prompts**.

---

## 2.2 Static vs Dynamic Prompts

### Static Prompt (avoid in real apps)
```
"Write a five-line poem on cricket."
```
- Hardcoded : user must retype for every new query
- Typos cause bad or unexpected output
- No control over formatting or consistency

### Dynamic Prompt (use this)
```
"Summarize the paper {paper_name} in a {style} style. Keep it {length}."
```
- Uses placeholders `{}` filled at runtime
- User picks from dropdowns, no typos possible
- Consistent, controlled output every time

```
User selects:
  Paper    → [ Dropdown ]   ← no spelling mistakes
  Style    → [ Simple / Math-heavy / Code-heavy ]
  Length   → [ Short / Medium / Long ]
        ↓
Template fills in → Final Prompt → Sent to AI
```

> **Board tip:** Draw a form with three dropdowns feeding into one prompt box.

---

## 2.3 Why Use `PromptTemplate` Instead of f-strings?

Python f-strings work, but `PromptTemplate` gives three extra benefits:

| Benefit | What it means |
|---|---|
| **Validation** | Warns you instantly if a placeholder is missing |
| **Reusability** | Save as JSON file, load in any project |
| **Integration** | Plugs directly into Chains |

---

## 2.4 Chains: Prompt + Model in One Step

Instead of two separate steps (format prompt, then call model), a **Chain** does both in one go.

```
[ Template ] → fill placeholders → [ Prompt ] → send to → [ Model ] → [ Answer ]
└──────────────────── Chain (one call) ────────────────────────────────────────┘
```

The `|` pipe symbol connects prompt to model:
```python
chain = prompt | model
answer = chain.invoke({"topic": "gravity"})
```

---

## 2.5 The Three Message Types for Chatbots

```
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│   SystemMessage    │   │   HumanMessage     │   │    AIMessage       │
│                    │   │                    │   │                    │
│ Sets AI behavior   │──▶│  User's input      │──▶│  Model's reply     │
│                    │   │                    │   │                    │
│ "You are a         │   │ "What is Python?"  │   │ "Python is a       │
│  helpful tutor"    │   │                    │   │  language..."      │
└────────────────────┘   └────────────────────┘   └────────────────────┘
   Set once at start        Every user turn           Every AI turn
```

---

## 2.6 Chat History: How the Bot Remembers

Without history → bot forgets everything after each reply.  
With history → every message is stored and sent together each time.

```
Turn 1:  [ System ] + [ Human: "Hi" ]  → AI replies → store AIMessage
Turn 2:  [ System ] + [ Human: "Hi" ] + [ AI: "Hello!" ] + [ Human: "What is ML?" ] → AI replies
Turn 3:  All previous messages + new message → AI replies
```

> The whole list grows with every turn. The model always sees the full conversation.

---

## 2.7 `ChatPromptTemplate` and `MessagesPlaceholder`

Used when your prompt has **multiple message types** with placeholders.

`MessagesPlaceholder` is a reserved slot where old chat history gets injected.

**Real-world example:**
```
Day 1: Customer asks for refund → bot confirms → conversation saved to database
Day 2: Customer returns → "Did my refund go through?"
         ↓
Load old chat from database
Inject into prompt via MessagesPlaceholder
Bot now has full context to answer correctly
```

---

# PART 3 : Python Code Examples

## Code 1: Talking to a Chat Model

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads API key from .env file

llm = ChatOpenAI(model="gpt-4")
reply = llm.invoke("Explain gravity in simple words.")
print(reply.content)
```

> `load_dotenv()` reads your secret API key from `.env` so you never paste it in code.

---

## Code 2: Making a Reusable Prompt Template

```python
from langchain_core.prompts import PromptTemplate

my_template = PromptTemplate(
    input_variables=["subject", "audience"],
    template="Teach me about {subject} as if I am a {audience}."
)

ready_prompt = my_template.invoke({
    "subject": "blockchain",
    "audience": "school student"
})
print(ready_prompt)
# "Teach me about blockchain as if I am a school student."
```

> Curly braces `{}` are the blank spaces that get filled at runtime.

---

## Code 3: Chaining Prompt and Model Together

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

chat_model = ChatOpenAI(model="gpt-4")

prompt = PromptTemplate(
    input_variables=["concept"],
    template="Give me a one-line definition of {concept}."
)

pipeline = prompt | chat_model  # prompt feeds directly into model

answer = pipeline.invoke({"concept": "recursion"})
print(answer.content)
```

> The `|` symbol connects steps : like a factory conveyor belt.

---

## Code 4: Chatbot That Remembers the Conversation

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

bot = ChatOpenAI(model="gpt-4")

conversation = [
    SystemMessage(content="You are a friendly math tutor.")
]

while True:
    user_text = input("Student: ")
    if user_text.lower() == "exit":
        break

    conversation.append(HumanMessage(content=user_text))
    bot_reply = bot.invoke(conversation)
    print("Tutor:", bot_reply.content)
    conversation.append(AIMessage(content=bot_reply.content))
```

> The whole `conversation` list goes to the model each time: that's how it remembers.

---

## Code 5: Loading Old Chat History into a New Session

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    MessagesPlaceholder(variable_name="past_chat"),  # old messages slot
    ("human", "{new_question}")
])

# Imagine these were loaded from a database
previous_session = [
    HumanMessage(content="Book me a flight to Delhi."),
    AIMessage(content="Done! Your flight is booked for Monday.")
]

final_prompt = chat_prompt.invoke({
    "role": "travel assistant",
    "past_chat": previous_session,
    "new_question": "What time is my flight?"
})
print(final_prompt)
```

> `MessagesPlaceholder` is a reserved slot: old messages slide in there automatically.

---

## Code 6: Finding the Most Similar Document

```python
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embed = OpenAIEmbeddings(model="text-embedding-3-large")

knowledge_base = [
    "Einstein developed the theory of relativity.",
    "Mango is the national fruit of India.",
    "Newton discovered the law of gravitation."
]

user_query = "Who worked on gravity?"

knowledge_vectors = embed.embed_documents(knowledge_base)
query_vector = embed.embed_query(user_query)

similarity_scores = cosine_similarity([query_vector], knowledge_vectors)[0]
top_result = knowledge_base[np.argmax(similarity_scores)]

print("Most relevant:", top_result)
# "Newton discovered the law of gravitation."
```

> Higher cosine similarity score = sentences are closer in meaning.

---

# Quick Revision Table

| Topic | What it does | Key class / tool |
|---|---|---|
| Language Model | Text in → Text out | `ChatOpenAI`, `ChatAnthropic` |
| Embedding Model | Text in → Numbers out | `OpenAIEmbeddings` |
| Temperature | Controls creativity (0 = strict, high = creative) | model parameter |
| Static prompt | Hardcoded string | plain Python string |
| Dynamic prompt | Template with placeholders | `PromptTemplate` |
| Chain | Prompt + Model in one pipe | `template \| model` |
| System message | Sets AI behavior at start | `SystemMessage` |
| Human message | User's typed input | `HumanMessage` |
| AI message | Model's reply | `AIMessage` |
| Chat history | List of all past messages | Python list |
| Old chat injection | Loads previous session into prompt | `MessagesPlaceholder` |
| Document search | Finds most similar text using vectors | `cosine_similarity` |

---

> **Golden rules for students:**
> - Always store API keys in `.env`, never in your code
> - Always use Chat Models, not plain LLMs
> - Use `PromptTemplate` over f-strings in real projects
> - Chat history = the memory of your chatbot
