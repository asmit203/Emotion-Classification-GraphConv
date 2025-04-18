import os
import gradio as gr
import nltk
from groq import Groq
import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Clause splitting function
HONORIFICS = r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Ave)\s+\.\s+\w+"
ABBREVIATIONS = r"\b(?:U\.S\.A|U\.K|Ph\.D|i\.e|e\.g|vs)\b"
NUMERIC_DECIMALS = r"\b\d+\.\d+\b"
NON_SPLIT_PATTERNS = f"({HONORIFICS}|{ABBREVIATIONS}|{NUMERIC_DECIMALS})"
DELIMITERS = r'(\s+[.,!?;:]\s+)'


def split_into_clauses(text):
    """
    Splits a given text into meaningful clauses while preserving abbreviations, honorifics, and numeric decimals.
    """

    SUBORDINATING_CONJUNCTIONS = [
        "because", "although", "though", "since", "as", "when", "while", 
        "after", "before", "if", "unless", "until", "whereas"
    ]

    # Preserve non-splitting patterns by replacing spaces with underscores
    def protect_patterns(match):
        return match.group(0).replace(" ", "_")

    text = re.sub(NON_SPLIT_PATTERNS, protect_patterns, text)

    # Process with spaCy
    doc = nlp(text)
    clauses = []
    current_clause = []

    for token in doc:
        if token.text.lower() in SUBORDINATING_CONJUNCTIONS and current_clause:
            clauses.append(" ".join(current_clause).strip())
            current_clause = [token.text]  # Start new clause with the conjunction
        elif token.dep_ in ["ROOT", "conj"]:
            current_clause.append(token.text)
            clauses.append(" ".join(current_clause).strip())
            current_clause = []
        else:
            current_clause.append(token.text)

    if current_clause:
        clauses.append(" ".join(current_clause).strip())

    # Further split based on delimiters
    refined_clauses = []
    for clause in clauses:
        sub_clauses = re.split(DELIMITERS, clause, flags=re.IGNORECASE)
        sub_clauses = [sub.strip().replace("_", " ") for sub in sub_clauses if sub.strip()]
        refined_clauses.extend(sub_clauses)

    # Remove empty or single-character clauses
    refined_clauses = [clause for clause in refined_clauses if len(clause) > 1]

    return refined_clauses

def classify_clause_with_groq(utterance, clause):
    """
    Uses Groq's LLM to classify a given clause as an emotion clause, cause clause, or neutral.
    """
    prompt = (
        "Determine if the clause is one of the following:\n"
        "1. Emotion clause (e.g., 'I am ecstatic', 'feeling down')\n"
        "2. Cause clause (e.g., 'because of the rain', 'due to the delay')\n"
        "3. Neutral clause (contains no explicit emotion or cause)\n\n"
        "Examples:\n"
        "Example 1:\n"
        "Utterance: 'I am upset because the meeting was canceled.'\n"
        "Clause: 'because the meeting was canceled.'\n"
        "Result: cause_clause\n\n"
        "Example 2:\n"
        "Utterance: 'I feel happy when I see old friends!'\n"
        "Clause: 'I feel happy'\n"
        "Result: emotion_clause\n\n"
        "Example 3:\n"
        "Utterance: 'Let's meet at the usual time.'\n"
        "Clause: 'Let's meet'\n"
        "Result: neutral_clause\n\n"
        "Provide the classification answer for the given clause. Do not add any introduction or conclusion. Just provide the answer.\n\n"
        f"Here is an utterance and one of its clauses:\n\n"
        f"Utterance: '{utterance}'\n"
        f"Clause: '{clause}'\n\n"
    )

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
    )

    output = chat_completion.choices[0].message.content.strip().lower()
    print(f"Utterance: {utterance}\nClause: {clause}\nOutput: {output}")

    # if "emotion_clause" in output:
    #     return "emotion_clause"
    # elif "cause_clause" in output:
    #     return "cause_clause"
    # else:
    #     return "neutral_clause"
    return output

def classify_text(text):
    """
    Classifies each clause in the input text without emotion-cause mapping.
    """
    clauses = split_into_clauses(text)
    classified_clauses = [{"clause": clause, "label": classify_clause_with_groq(text, clause)} for clause in clauses]

    return {"clauses": classified_clauses}

# Gradio Interface
iface = gr.Interface(
    fn=classify_text,
    inputs="text",
    outputs="json",
    title="Clause Classifier",
    description="Enter text to classify its clauses into emotion, cause, or neutral.",
)

# Launch the app for local network access
iface.launch(server_name="0.0.0.0", server_port=5000)
