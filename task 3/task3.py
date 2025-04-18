# -------------------------
# 1. Load Data and Extract Clauses & Labels
# (Code remains unchanged)
# -------------------------
import json
from tqdm import tqdm
import json
import os
from tqdm import tqdm
import torch
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import pandas as pd
from tqdm import tqdm

NUM_EPOCHS = 10000

# with open("/DATA/sujit_2021cs35/nlp_project/code/task 2/results_qwen_9pm.json", "r") as f:
with open("/DATA/sujit_2021cs35/nlp_project/code/task 2/train_set.json", "r") as f:
    data = json.load(f)
    
all_clauses = []
all_label = []

for conversation in tqdm(data.values(), desc="Processing conversations"):
    for dialogue in conversation:
        if "clauses" in dialogue and len(dialogue["clauses"]) != 0:
            temp_clauses = []
            temp_labels = []
            for clause in dialogue["clauses"]:
                temp_clauses.append(clause["clause"])
                temp_labels.append(clause["label"])
            all_clauses.append(temp_clauses)
            all_label.append(temp_labels)

# -------------------------
# 2. Create One-Hot Annotations for Each Clause
# (Code remains unchanged)
# -------------------------

label_map = {
    "emotion_clause": [1, 0, 0],
    "cause_clause":   [0, 1, 0],
    "neutral":        [0, 0, 1]
}

annotations = []
for ann in tqdm(all_label, desc="Mapping labels"):
    anns = []
    for label in ann:
        anns.append(label_map.get(label, [0, 0, 1]))  # default to "neutral" if not found
    annotations.append(anns)

# -------------------------
# 3. Setup for BERT Embeddings (using CUDA)
# (Code remains unchanged)
# -------------------------

device = 'cuda:1'
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
bert_model = AutoModel.from_pretrained("google-bert/bert-base-cased", config=config)
bert_model.to(device)

def get_embeddings(clauses):
    """
    Given a list of clause strings, returns their BERT [CLS] token embeddings.
    """
    if isinstance(clauses, str):
        clauses = [clauses]
    inputs = tokenizer(clauses, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().cpu()

# -------------------------
# 4. Graph Creation and Data Preparation Functions
# -------------------------

def create_graph(clauses, annotations):
    """
    Create a graph where:
    1. Each node is a clause
    2. Consecutive nodes are connected
    3. Cause clauses are connected to emotion clauses
    """
    G = nx.DiGraph()  # Using directed graph now to represent cause->emotion relationship
    
    # Add nodes with their attributes
    emotion_clause_indices = []
    cause_clause_indices = []
    for i, clause in enumerate(clauses):
        annotation = annotations[i] if i < len(annotations) else [0, 0, 1]
        G.add_node(i, text=clause, type=annotation)
        
        # Track indices of emotion and cause clauses
        if annotation[0] == 1:  # emotion_clause [1,0,0]
            emotion_clause_indices.append(i)
        elif annotation[1] == 1:  # cause_clause [0,1,0]
            cause_clause_indices.append(i)
    
    # Add consecutive edges (maintain temporal flow)
    for i in range(len(clauses) - 1):
        G.add_edge(i, i + 1, edge_type='sequential')
    
    # Add edges from cause clauses to emotion clauses
    for cause_idx in cause_clause_indices:
        for emotion_idx in emotion_clause_indices:
            # Only connect causes to emotions that come after them in the conversation
            # This reflects the typical causality direction (cause precedes effect)
            if emotion_idx > cause_idx:
                G.add_edge(cause_idx, emotion_idx, edge_type='causal')
    
    return G

def prepare_data(G, embeddings):
    """
    Prepares a PyTorch Geometric Data object using node embeddings and graph structure.
    """
    x = embeddings  # Tensor of shape (num_nodes, embedding_dim)
    nodes = list(G.nodes())
    
    # Create edge_index tensor from graph edges.
    edges = list(G.edges)
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Extract node labels from graph attributes.
    labels = []
    for node in nodes:
        ann = G.nodes[node].get('type', [0, 0, 1])
        # Convert one-hot list to class index.
        label = ann.index(max(ann)) if isinstance(ann, list) else ann
        labels.append(label)
    labels = torch.tensor(labels, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=labels)
    return data

# -------------------------
# 5. Define the GNN Model and Training Functions
# (Code remains unchanged)
# -------------------------

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
    batch = Batch.from_data_list(data_list)
    out = model(batch.x, batch.edge_index)
    loss = criterion(out, batch.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(model, data_list):
    """
    Evaluates the model's node classification accuracy on the provided data.
    """
    model.eval()
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
    correct = (pred == batch.y).sum().item()
    total = batch.y.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy

# -------------------------
# 6. Train the Model on All Clauses and Annotations
# -------------------------

# Preparing data for all conversations.
data_list = []
for clauses, labels in tqdm(zip(all_clauses, annotations), total=len(all_clauses), desc="Preparing data"):
    embeddings = get_embeddings(clauses)
    G = create_graph(clauses, labels)
    data = prepare_data(G, embeddings)
    data_list.append(data)

# Define the model, optimizer, and loss criterion.
model = GraphTransformer(input_dim=768, hidden_dim=64, output_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop.
for epoch in tqdm(range(NUM_EPOCHS)):
    loss = train_model(model, data_list, optimizer, criterion)
    if (epoch + 1) % 20 == 0:
        acc = evaluate_model(model, data_list)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

# Final evaluation after training.
acc_final = evaluate_model(model, data_list)
print("Final node classification accuracy (Train): {:.2f}%".format(acc_final * 100))

# -------------------------
# 7. Test the Model on New Data
# -------------------------

# Save the trained model
model_path = "/DATA/sujit_2021cs35/nlp_project/code/task 3/gnn_model_new_l5_apr2.pt"
torch.save(model.state_dict(), model_path)
print(f"Trained model saved to {model_path}")

# Load test data from JSON
test_json_path = "/DATA/sujit_2021cs35/nlp_project/code/task 2/test_set.json"
test_clauses = []
test_labels = []

# Load test data from JSON
with open(test_json_path, "r") as f:
    test_data = json.load(f)

for conversation in tqdm(test_data.values(), desc="Processing test conversations"):
    for dialogue in conversation:
        if "clauses" in dialogue and len(dialogue["clauses"]) != 0:
            temp_clauses = []
            temp_labels = []
            for clause in dialogue["clauses"]:
                temp_clauses.append(clause["clause"])
                temp_labels.append(clause["label"])
            test_clauses.append(temp_clauses)
            test_labels.append(temp_labels)

print(f"Loaded {len(test_clauses)} conversations for testing")

# Create one-hot annotations for test labels
test_annotations = []
for ann in tqdm(test_labels, desc="Mapping test labels"):
    anns = []
    for label in ann:
        anns.append(label_map.get(label, [0, 0, 1]))  # default to "neutral" if not found
    test_annotations.append(anns)

# Prepare test data for evaluation
test_data_list = []
for clauses, labels in tqdm(zip(test_clauses, test_annotations), total=len(test_clauses), desc="Preparing test data"):
    embeddings = get_embeddings(clauses)
    G = create_graph(clauses, labels)
    data = prepare_data(G, embeddings)
    test_data_list.append(data)

# Evaluate the model on test data
model.eval()
test_acc = evaluate_model(model, test_data_list)
print("Node classification accuracy (Test): {:.2f}%".format(test_acc * 100))

# Optional: Generate predictions for each test sample
def generate_predictions(model, data_list):
    model.eval()
    all_predictions = []
    
    for data in data_list:
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_predictions.append(preds)
    
    return all_predictions

# Get predictions
test_predictions = generate_predictions(model, test_data_list)
print(f"Generated predictions for {len(test_predictions)} test conversations")

# Save predictions to a JSON file
output_path = "/DATA/sujit_2021cs35/nlp_project/code/task 3/test_predictions_new_l5_apr2.json"
with open(output_path, "w") as f:
    json.dump([pred.tolist() for pred in test_predictions], f)
print(f"Predictions saved to {output_path}")
# -------------------------
