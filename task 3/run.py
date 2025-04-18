# -------------------------
# 1. Load Data and Extract Clauses & Labels
# (Code remains unchanged)
# -------------------------

with open("/DATA/sujit_2021cs35/nlp_project/code/task 2/results_qwen_9pm.json", "r") as f:
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

device = 'cuda:0'
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
config = AutoConfig.from_pretrained("google-bert/bert-base-cased")
bert_model = AutoModel.from_pretrained("google-bert/bert-base-cased", config=config)
bert_model.to(device)

# -------------------------
# 4. Graph Creation and Data Preparation Functions
# (Code remains unchanged)
# -------------------------

def create_graph(clauses, annotations):
    """
    Create a simple graph where each node is a clause and consecutive nodes are connected.
    """
    G = nx.Graph()
    for i, clause in enumerate(clauses):
        annotation = annotations[i] if i < len(annotations) else [0, 0, 1]
        G.add_node(i, text=clause, type=annotation)
    # Add consecutive edges.
    for i in range(len(clauses) - 1):
        G.add_edge(i, i + 1)
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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphTransformer, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
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
for epoch in range(100):
    loss = train_model(model, data_list, optimizer, criterion)
    if (epoch + 1) % 20 == 0:
        acc = evaluate_model(model, data_list)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

# Final evaluation after training.
acc_final = evaluate_model(model, data_list)
print("Final node classification accuracy: {:.2f}%".format(acc_final * 100))
