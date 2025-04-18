import gradio as gr
import torch
import json
import re
import spacy
import networkx as nx
from torch_geometric.nn import GCNConv
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data

# Load spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")

# Load the trained model
class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=3):
        super(GraphTransformer, self).__init__()
        self.conv1 = GCNConv(768, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, 256)
        self.conv4 = GCNConv(256, 128)
        self.conv5 = GCNConv(128, 10)
        self.fc = torch.nn.Linear(10, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index).relu()
        return self.fc(x)

model = GraphTransformer(input_dim=768, hidden_dim=64, output_dim=3)
model.load_state_dict(torch.load("/DATA/sujit_2021cs35/nlp_project/code/task 3/gnn_model_new_l5_apr2.pt"))
model.eval()

# Load BERT tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
bert_model = AutoModel.from_pretrained("google-bert/bert-base-cased")

def get_embeddings(clauses):
    inputs = tokenizer(clauses, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

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

# Graph creation function
def create_graph(clauses):
    G = nx.DiGraph()
    for i, clause in enumerate(clauses):
        G.add_node(i, text=clause)
        if i > 0:
            G.add_edge(i - 1, i)
    return G

def prepare_data(G, embeddings):
    nodes = list(G.nodes())
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    data = Data(x=embeddings, edge_index=edge_index)
    return data

# Function to process input and predict
def classify_text(input_text):
    clauses = split_into_clauses(input_text)
    embeddings = get_embeddings(clauses)
    G = create_graph(clauses)
    data = prepare_data(G, embeddings)
    with torch.no_grad():
        outputs = model(data.x, data.edge_index)
        predictions = outputs.argmax(dim=1).tolist()
    label_map = ["emotion_clause", "cause_clause", "neutral"]
    return {clause: label_map[pred] for clause, pred in zip(clauses, predictions)}

iface = gr.Interface(fn=classify_text, inputs="text", outputs="json", title="Clause Classifier", description="Enter a conversation text to classify its clauses into emotion, cause, or neutral.")
iface.launch(server_name="0.0.0.0", server_port=7860)
