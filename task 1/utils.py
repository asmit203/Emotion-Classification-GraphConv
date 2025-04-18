import spacy
import re
import json

# Load spaCy's English model.
nlp = spacy.load("en_core_web_sm")

#! v4

# Patterns to preserve abbreviations, honorifics, and numeric decimals
HONORIFICS = r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Ave)\s+\.\s+\w+"
ABBREVIATIONS = r"\b(?:U\.S\.A|U\.K|Ph\.D|i\.e|e\.g|vs)\b"
NUMERIC_DECIMALS = r"\b\d+\.\d+\b"  # Matches decimal numbers like 3.14
NON_SPLIT_PATTERNS = f"({HONORIFICS}|{ABBREVIATIONS}|{NUMERIC_DECIMALS})"

# Clause delimiters
DELIMITERS = r'(\s+[.,!?;:]\s+)'

def split_into_clauses(text):
    """
    Splits a given text into meaningful clauses while preserving abbreviations, honorifics, and numeric decimals.
    """

    # Preserve non-splitting patterns by replacing spaces with underscores
    def protect_patterns(match):
        return match.group(0).replace(" ", "_")

    text = re.sub(NON_SPLIT_PATTERNS, protect_patterns, text)

    # Process with spaCy
    doc = nlp(text)
    clauses = []
    current_clause = []

    for token in doc:
        current_clause.append(token.text)
        if token.dep_ in ["ROOT", "conj"]:
            if current_clause:
                clauses.append(" ".join(current_clause).strip())
                current_clause = []

    if current_clause:
        clauses.append(" ".join(current_clause).strip())

    # Further split based on delimiters
    refined_clauses = []
    for clause in clauses:
        sub_clauses = re.split(DELIMITERS, clause, flags=re.IGNORECASE)
        sub_clauses = [sub.strip().replace("_", " ") for sub in sub_clauses if sub.strip()]
        refined_clauses.extend(sub_clauses)
    
    # remove entries in refined_clauses that are empty strings or length 1
    refined_clauses = [clause for clause in refined_clauses if len(clause) > 1]

    return refined_clauses


def process_reccon_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    for transaction_id, conversations in data.items():
        for conversation in conversations:
            for utterance in conversation:
                if 'utterance' in utterance:
                    utterance['clauses'] = split_into_clauses(utterance['utterance'])

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

#! v3

def split_sentence_into_clauses(sent):
    """
    Split a spaCy sentence (Span) into clauses based on dependency labels and punctuation delimiters,
    avoiding splits after commas followed by abbreviations and dependency splits in proper nouns.
    """
    boundaries = set()
    boundaries.add(0)
    boundaries.add(len(sent))
    
    sentence_ending = {"?", "!"}
    punctuation_for_conjunction = {",", ";", ":"}
    abbreviations = {"Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr"}
    
    for i, token in enumerate(sent):
        # Handle sentence-ending punctuation
        if token.text in sentence_ending:
            # if token.text == "." and i > 0 and sent[i-1].text in abbreviations:
            #     continue  # Skip abbreviation periods
            boundaries.add(i + 1)
        
        # Split at punctuation_for_conjunction unless followed by an abbreviation
        if token.text in punctuation_for_conjunction:
            # if i + 1 < len(sent) and sent[i+1].text in abbreviations:
            #     continue  # Avoid splitting before abbreviations
            boundaries.add(i + 1)
        
        # Dependency-based splits (exclude proper nouns)
        if token.dep_ in {"advcl", "ccomp", "xcomp", "relcl", "acl"} and token.pos_ != "PROPN":
            boundaries.add(i)
        
        # Split after coordinating conjunctions
        if token.dep_ == "cc" and i + 1 < len(sent):
            boundaries.add(i + 1)
    
    boundaries = sorted(boundaries)
    clauses = []
    for j in range(len(boundaries) - 1):
        start, end = boundaries[j], boundaries[j+1]
        clause = sent[start:end].text.strip()
        if clause:
            clauses.append(clause)
    return clauses


#! v2
# def split_sentence_into_clauses(sent):
#     """
#     Split a spaCy sentence (Span) into clauses based on dependency labels and punctuation delimiters,
#     while avoiding false splits when punctuation is merely space-separated.
    
#     This version uses:
#       - Sentence-ending punctuation (".", "?", "!") as clause boundaries (unless the period follows a known abbreviation).
#       - Punctuation like commas, semicolons, or colons as clause boundaries.
#       - Dependency markers (e.g. coordinating conjunctions and subordinate clause indicators) as additional boundaries.
#     """
#     boundaries = set()
#     boundaries.add(0)
#     boundaries.add(len(sent))
    
#     # Define which punctuation normally indicates sentence end.
#     sentence_ending = {".", "?", "!"}
#     # For other punctuation, split always.
#     punctuation_for_conjunction = {",", ";", ":"}
#     # List of abbreviations (without the trailing period).
#     abbreviations = {"Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr"}
    
#     for i, token in enumerate(sent):
#         # Sentence-ending punctuation is a clause boundary.
#         if token.text in sentence_ending:
#             # If the period might be part of an abbreviation, skip it.
#             if token.text == "." and i > 0:
#                 if sent[i - 1].text in abbreviations:
#                     continue
#             if i + 1 < len(sent):
#                 boundaries.add(i + 1)
                
#         # Split at punctuation_for_conjunction (commas, semicolons, colons) unconditionally
#         if token.text in punctuation_for_conjunction:
#             boundaries.add(i + 1)  # Safe even if i+1 is beyond len(sent) (handled by set)
                
#         # Dependency-based boundary: after a coordinating conjunction.
#         if token.dep_ == "cc" and i + 1 < len(sent):
#             boundaries.add(i + 1)
            
#         # Dependency-based boundary: at markers of subordinate clauses.
#         if token.dep_ in {"advcl", "ccomp", "xcomp", "relcl", "acl"} and i != 0:
#             boundaries.add(i)
    
#     boundaries = sorted(boundaries)
#     clauses = []
#     for j in range(len(boundaries) - 1):
#         span = sent[boundaries[j]:boundaries[j + 1]]
#         clause_text = span.text.strip()
#         if clause_text:
#             clauses.append(clause_text)
#     return clauses
#! v1
# def split_sentence_into_clauses(sent):
#     """
#     Split a spaCy sentence (Span) into clauses based on dependency labels and punctuation delimiters,
#     while avoiding false splits when punctuation is merely space-separated.
    
#     This version uses:
#       - Sentence-ending punctuation (".", "?", "!") as clause boundaries (unless the period follows a known abbreviation).
#       - Punctuation like commas, semicolons, or colons as clause boundaries.
#       - Dependency markers (e.g. coordinating conjunctions and subordinate clause indicators) as additional boundaries.
#     """
#     boundaries = set()
#     boundaries.add(0)
#     boundaries.add(len(sent))
    
#     # Define which punctuation normally indicates sentence end.
#     sentence_ending = {".", "?", "!"}
#     # For other punctuation, split always.
#     punctuation_for_conjunction = {",", ";", ":"}
#     # List of abbreviations (without the trailing period).
#     abbreviations = {"Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr"}
    
#     for i, token in enumerate(sent):
#         # Sentence-ending punctuation is a clause boundary.
#         if token.text in sentence_ending:
#             # If the period might be part of an abbreviation, skip it.
#             if token.text == "." and i > 0:
#                 if sent[i - 1].text in abbreviations:
#                     continue
#             if i + 1 < len(sent):
#                 boundaries.add(i + 1)
                
#         # Split at punctuation_for_conjunction (commas, semicolons, colons) unconditionally
#         if token.text in punctuation_for_conjunction:
#             boundaries.add(i + 1)  # Safe even if i+1 is beyond len(sent) (handled by set)
                
#         # Dependency-based boundary: after a coordinating conjunction.
#         if token.dep_ == "cc" and i + 1 < len(sent):
#             boundaries.add(i + 1)
            
#         # Dependency-based boundary: at markers of subordinate clauses.
#         if token.dep_ in {"advcl", "ccomp", "xcomp", "relcl", "acl"} and i != 0:
#             boundaries.add(i)
    
#     boundaries = sorted(boundaries)
#     clauses = []
#     for j in range(len(boundaries) - 1):
#         span = sent[boundaries[j]:boundaries[j + 1]]
#         clause_text = span.text.strip()
#         if clause_text:
#             clauses.append(clause_text)
#     return clauses