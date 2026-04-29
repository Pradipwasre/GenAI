# Structured Output

---

## Picking Up from Last Time

Last session we built prompts, sent them to a model, and got replies.
Everything worked well, but the replies always came back as a block of text.

Now think about this situation:

You built a cricket score app. You ask the LLM for today's match summary.
It replies: "Mumbai Indians defeated Chennai Super Kings by 6 wickets in a thrilling
encounter at Wankhede Stadium. Rohit Sharma scored 74 off 48 balls."

Great reply. But how do you show the winning team in a red banner and the
runs scored in a separate scoreboard widget on your app?

You cannot split that sentence reliably. The LLM chose its own words.
Next time it might say it differently. Your app would break.

This is the exact problem structured output solves.

---

## 1. The Core Idea

When the LLM replies in plain text, it is writing for a human reader.
When we need the output to go into a program, database, or UI widget,
we need it shaped like data, not like a sentence.

```
Plain text reply         -->  Hard to split, unpredicatable, breaks programs
Structured data reply    -->  Fixed shape, predictable, programs can read it directly
```

Look at the same cricket information two ways:

```
WAY 1 : Plain text:
"Mumbai Indians beat CSK by 6 wickets. Rohit scored 74 off 48."

WAY 2 : Structured:
{
  "winner":       "Mumbai Indians",
  "loser":        "CSK",
  "margin":       "6 wickets",
  "top_scorer":   "Rohit Sharma",
  "runs":         74,
  "balls":        48
}
```

Way 2 can directly populate your app's UI fields, get stored in a database,
or be sent as an API response, without any extra processing.

> Board sketch: Two boxes. Left has a sentence. Right has labeled fields.
> Draw arrows from the right box going to "Database", "Mobile app", "API".
> Nothing from the left box connects to anything.

---

## 2. Real Situations Where This Matters

### Situation 1 : IPL Fantasy App

Your fantasy cricket app needs to update player stats after every IPL 2026 match.
You ask the LLM to summarise a player's performance.

If it returns a paragraph, you have to guess where the runs end and wickets begin.
If it returns structured fields : `runs`, `wickets`, `strike_rate`, `match_date`,
your app can update every player card automatically without any manual work.

### Situation 2 : College Exam Result Checker

Students paste their marksheet text into your app.
The LLM reads it and pulls out subject names, marks, and pass or fail status.

Structured output means you can directly build a table, calculate the percentage,
and highlight failed subjects, all from one LLM call.

### Situation 3: Movie Recommendation Bot

A student asks: "Suggest a good movie for tonight."
The LLM suggests a movie. But your app wants to show the poster, rating,
genre tag, and a one-line reason, each in a separate UI card section.

Structured output gives you exactly those fields to map to each section.

---

## 3. How LangChain Makes This Happen

LangChain gives one clean method for this:

```python
structured_model = model.with_structured_output(YourSchema)
```

You define the shape once. LangChain handles writing the instructions to the model.
The model returns data that matches your shape. You never touch the prompt formatting.

```
You write  :  model.with_structured_output(schema)
LangChain  :  adds instructions to system prompt automatically
Model      :  returns data shaped exactly as you defined
Your code  :  receives clean structured data, ready to use
```

---

## 4. Three Ways to Define the Shape

Before using `with_structured_output`, you need to describe the shape you want.
LangChain accepts three formats for this.

---

### Option A: TypedDict

Built into Python. No installation needed.
You define a class that says which keys exist and what type each one should be.

Limitation worth knowing: Python does not check the types at runtime.
If the model sends the wrong type, your code will not immediately complain.
The problem shows up later and is harder to trace.

Best situation to use: Learning, quick experiments, small personal projects.

---

### Option B : Pydantic

A separate library (`pip install pydantic`) that adds real checking on top.
If the model returns the wrong type for any field, Pydantic raises a clear
error right away, before that bad data touches anything else.

You can also add rules like "this number must be between 1 and 100"
or "this field is optional" or add a description to guide the model.

Best situation to use: Any real project where data accuracy matters.

---

### Option C : JSON Schema

A dictionary-based format that any programming language understands.
Not tied to Python at all. If your backend is Python and your frontend
is React or Flutter, both sides can share the same schema definition.

Best situation to use: Projects where the same data shape is needed
across Python, JavaScript, or any other language.

---

### At a Glance

```
                    TypedDict     Pydantic      JSON Schema
Catches type errors     No           Yes            No
Default values          No           Yes            No
Works across languages  No           No             Yes
Needs extra library     No           Yes            No
Good for                Learning     Real apps      Multi-language projects
```

> Flowchart to draw on board:
>
> Only Python project?
>   Yes --> Need type validation? --> Yes --> Pydantic
>                                --> No  --> TypedDict
>   No  --> Multiple languages?  --> Yes --> JSON Schema

---

## 5. Code Examples: Using Groq API

Groq gives a free and fast API for running LLMs.
In LangChain, swapping OpenAI for Groq is just changing one import and one class name.
Everything else, `.with_structured_output()`, `.invoke()`, chains, stays identical.

Setup:
```
pip install langchain-groq
```

Add to your `.env` file:
```
GROQ_API_KEY=your_key_here
```

Get a free key at: https://console.groq.com

---

### Code 1: IPL 2026 Match Summary: TypedDict

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, List

load_dotenv()

class MatchSummary(TypedDict):
    team_won: str
    team_lost: str
    winning_margin: str
    venue: str
    top_batter: str
    top_batter_runs: int
    key_moment: str

llm = ChatGroq(model="llama3-8b-8192")
ipl_bot = llm.with_structured_output(MatchSummary)

response = ipl_bot.invoke(
    "Describe a fictional IPL 2026 final match between Mumbai Indians and RCB."
)

print("Winner       :", response["team_won"])
print("Venue        :", response["venue"])
print("Top batter   :", response["top_batter"], "—", response["top_batter_runs"], "runs")
print("Key moment   :", response["key_moment"])
```

---

### Code 2: Movie Night Picker : Pydantic

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

class MovieSuggestion(BaseModel):
    movie_name: str
    release_year: int
    genre: List[str] = Field(description="List of genres like action, comedy, thriller")
    why_watch_tonight: str = Field(description="One compelling reason to watch it today")
    imdb_rating: float = Field(ge=1.0, le=10.0)
    available_in_hindi: Optional[bool] = None

llm = ChatGroq(model="llama3-8b-8192")
movie_bot = llm.with_structured_output(MovieSuggestion)

pick = movie_bot.invoke(
    "Suggest one great movie for a college student to watch on a Friday night."
)

print("Movie   :", pick.movie_name, f"({pick.release_year})")
print("Genre   :", ", ".join(pick.genre))
print("Why     :", pick.why_watch_tonight)
print("Rating  :", pick.imdb_rating, "/ 10")
```

> `ge=1.0, le=10.0` means the rating must stay between 1 and 10.
> Pydantic throws an error if the model tries to sneak in an 11.

---

### Code 3: Exam Result Breakdown: JSON Schema

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

result_schema = {
    "title": "ExamResult",
    "type": "object",
    "properties": {
        "student_name":  { "type": "string" },
        "total_marks":   { "type": "integer" },
        "percentage":    { "type": "number" },
        "grade":         { "type": "string" },
        "passed":        { "type": "boolean" },
        "weak_subjects": { "type": "array", "items": { "type": "string" } }
    },
    "required": ["student_name", "total_marks", "percentage", "grade", "passed"]
}

llm = ChatGroq(model="llama3-8b-8192")
result_bot = llm.with_structured_output(result_schema)

data = result_bot.invoke(
    "A student named Riya scored 340 out of 500. She struggled in Physics and Maths. Analyse her result."
)

print("Name       :", data["student_name"])
print("Percentage :", data["percentage"], "%")
print("Grade      :", data["grade"])
print("Passed     :", data["passed"])
print("Needs work :", data["weak_subjects"])
```

> JSON Schema is a plain dictionary, no special classes.
> If your college app also has a React frontend, it can use this exact same schema.

---

### Code 4: Chaining Prompt + Structured Output Together

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class GameReview(BaseModel):
    game_title: str
    platform: str = Field(description="PC, Mobile, PS5 etc")
    fun_rating: int = Field(ge=1, le=10, description="How fun is it out of 10")
    best_feature: str
    worth_buying: bool
    similar_games: List[str] = Field(description="Two or three games with similar feel")

llm = ChatGroq(model="llama3-8b-8192")
review_bot = llm.with_structured_output(GameReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a gaming expert who gives honest reviews for college students."),
    ("human", "Review the game: {game_name}")
])

chain = prompt | review_bot

verdict = chain.invoke({"game_name": "BGMI"})

print("Game         :", verdict.game_title)
print("Platform     :", verdict.platform)
print("Fun rating   :", verdict.fun_rating, "/ 10")
print("Best feature :", verdict.best_feature)
print("Worth buying :", verdict.worth_buying)
print("Similar to   :", ", ".join(verdict.similar_games))
```

> This combines everything, prompt template, system message, chain, and structured output.
> One `chain.invoke()` call and you get back a fully shaped, validated object.

---

## Quick Revision

| Thing | What it does |
|---|---|
| Unstructured output | Plain text, good for reading, bad for programs |
| Structured output | Shaped data, programs, databases, APIs can use it directly |
| `with_structured_output` | LangChain method that tells the model what shape to return |
| TypedDict | Simplest schema, no validation, good for learning |
| Pydantic | Schema plus real validation, use in actual projects |
| JSON Schema | Works across languages, share between Python and JavaScript |
| Groq API | Free and fast LLM API, same code as OpenAI |

---

## What We Are Doing Next, And Why It Gets Interesting

So far we used `with_structured_output` and it worked smoothly.

But here is something we need to talk about honestly:
this method only works well with certain models, mainly the paid ones like GPT-4 or Claude.

What happens when you try this with a free open-source model?
Most of them simply ignore your schema and reply however they want.
Some return broken JSON. Some return plain text as if you never asked for structure.

So the question becomes, how do we get structured data out of a model
that refuses to cooperate?

That is where Output Parsers come in.

Instead of asking the model to return data in our format,
Output Parsers take whatever the model gives back, even messy text,
and extract the structure from it on our end.

There are four types, each more powerful than the last.
By the end of that session, no matter which LLM you use, paid or free,
you will always be able to get back the data shape your program needs.

That is a genuinely useful skill and it also sets the foundation for building
agents, which are coming up a few sessions later.
