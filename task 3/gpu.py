import json
from tqdm import tqdm
import os
import torch
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import pandas as pd

NUM_EPOCHS = 10000
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Load Data
with open("/DATA/sujit_2021cs35/nlp_project/code/task 2/train_set.json", "r") as f:
    data = json.load(f)

all_clauses, all_label = [], []
for conversation in tqdm(data.values(), desc="Processing conversations"):
    for dialogue in conversation:
        if "clauses" in dialogue and dialogue["clauses"]:
            all_clauses.append([clause["clause"] for clause in dialogue["clauses"]])
            all_label.append([clause["label"] for clause in dialogue["clauses"]])

# Label Mapping
label_map = {"emotion_clause": [1, 0, 0], "cause_clause": [0, 1, 0], "neutral": [0, 0, 1]}
annotations = [[label_map.get(label, [0, 0, 1]) for label in ann] for ann in all_label]

# Load BERT Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
bert_model = AutoModel.from_pretrained("google-bert/bert-base-cased", config=config).to(device)

def get_embeddings(clauses):
    if isinstance(clauses, str):
        clauses = [clauses]
    inputs = tokenizer(clauses, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach()

# Graph Creation
def create_graph(clauses, annotations):
    G = nx.DiGraph()
    emotion_clause_indices, cause_clause_indices = [], []
    
    for i, clause in enumerate(clauses):
        annotation = annotations[i] if i < len(annotations) else [0, 0, 1]
        G.add_node(i, text=clause, type=annotation)
        if annotation[0]: emotion_clause_indices.append(i)
        if annotation[1]: cause_clause_indices.append(i)
    
    for i in range(len(clauses) - 1):
        G.add_edge(i, i + 1, edge_type='sequential')
    
    for cause_idx in cause_clause_indices:
        for emotion_idx in emotion_clause_indices:
            if emotion_idx > cause_idx:
                G.add_edge(cause_idx, emotion_idx, edge_type='causal')
    
    return G

def prepare_data(G, embeddings):
    x = embeddings.to(device)
    edges = list(G.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device) if edges else torch.empty((2, 0), dtype=torch.long, device=device)
    labels = torch.tensor([G.nodes[node].get('type', [0, 0, 1]).index(max(G.nodes[node].get('type', [0, 0, 1]))) for node in G.nodes()], dtype=torch.long, device=device)
    return Data(x=x, edge_index=edge_index, y=labels)

# Define GNN Model
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

def train_model(model, data_list, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    batch = Batch.from_data_list(data_list).to(device)
    out = model(batch.x, batch.edge_index)
    loss = criterion(out, batch.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(model, data_list):
    model.eval()
    batch = Batch.from_data_list(data_list).to(device)
    with torch.no_grad():
        pred = model(batch.x, batch.edge_index).argmax(dim=1)
    return (pred == batch.y).sum().item() / batch.y.size(0) if batch.y.size(0) > 0 else 0

# Preparing Data
data_list = [prepare_data(create_graph(clauses, labels), get_embeddings(clauses)) for clauses, labels in tqdm(zip(all_clauses, annotations), total=len(all_clauses), desc="Preparing data")]

# Training Model
model = GraphTransformer(768, 64, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in tqdm(range(NUM_EPOCHS)):
    loss = train_model(model, data_list, optimizer, criterion)
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {evaluate_model(model, data_list) * 100:.2f}%")

# Save Model
torch.save(model.state_dict(), "/DATA/sujit_2021cs35/nlp_project/code/task 3/gnn_model_gpu.pt")
