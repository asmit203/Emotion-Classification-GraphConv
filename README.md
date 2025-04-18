# Emotion and Cause Classification in Conversational Text

## Project Structure

The project is organized into three main tasks:

- **Task 1**: Clause segmentation and preprocessing
- **Task 2**: Dataset preparation and annotation
- **Task 3**: Graph-based emotion and cause classification

### Files and Directories

- utils.py: Contains utilities for clause segmentation and preprocessing
- task2.py: LLM-based classification of clauses into emotion/cause types
- task3.py: Main GNN implementation for emotion and cause classification
- inference.py: Script for model inference on test data
- run.py: Alternative training script with different configurations
- gpu.py: GPU-optimized version of the training script

## Key Features

### 1. Clause Extraction

The system splits utterances into semantic clauses using a combination of dependency parsing with spaCy and rule-based heuristics. This is implemented in utils.py.

### 2. Graph Construction

For each conversation, we build a directed graph where:
- Nodes represent clauses
- Sequential edges connect consecutive clauses
- Causal edges connect cause clauses to emotion clauses

### 3. Graph Neural Network Architecture

The emotion and cause classification model uses a Graph Transformer architecture with:
- BERT embeddings for node features
- Multiple GCN layers (5 in total)
- Final classification into three classes: emotion, cause, and neutral

```python
class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=3):
        super(GraphTransformer, self).__init__()
        self.conv1 = GCNConv(768, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, 256)
        self.conv4 = GCNConv(256, 128)
        self.conv5 = GCNConv(128, 10)
        self.fc = torch.nn.Linear(10, output_dim)
```

## Usage

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install torch torch_geometric transformers networkx spacy tqdm
   python -m spacy download en_core_web_sm
   ```

### Data Preparation

1. Process your dialogue dataset with clause segmentation:
   ```python
   from task1.utils import process_reccon_data
   process_reccon_data("input.json", "output_with_clauses.json")
   ```

2. Prepare training and test sets using task2.py

### Training

To train the GNN model:

```bash
python task3/task3.py
```

The script will:
1. Load training data
2. Extract clause embeddings using BERT
3. Create graph representations
4. Train the GNN model
5. Save the trained model and evaluate on test data

### Inference

Run inference on new data:

```bash
python task3/inference.py
```

## Model Details

- Input: BERT embeddings (768 dimensions) for each clause
- Graph structure: Directed graph with sequential and causal edges
- Output: Classification of each clause as emotion, cause, or neutral
- Training: Using CrossEntropyLoss and Adam optimizer

## Results

The model achieves high accuracy in distinguishing between emotion-expressing clauses, cause clauses, and neutral clauses in conversational text. Predictions are saved as JSON files for further analysis.

## Extensions

- Multiple pre-trained model checkpoints are available in the task3 directory
- GPU-optimized versions for faster training on CUDA-enabled devices
- Visualization utilities for graph structures

## License

This project is provided as-is without warranty. Please cite this repository if you use the code or concepts in your research.