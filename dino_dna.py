import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from data import DNADataset
from model import DNATransformer_ALiBi
from transformers import AutoTokenizer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pad sequences to context length
def pad_to_context_length(sequences, context_length, pad_token_id):
    padded_sequences = []
    for seq in sequences:
        if seq.size(1) < context_length:
            padding = torch.full((seq.size(0), context_length - seq.size(1)), pad_token_id, device=seq.device)
            padded_seq = torch.cat([seq, padding], dim=1)
        else:
            padded_seq = seq[:, :context_length]
        padded_sequences.append(padded_seq)
    return padded_sequences

# Generate subsequence views
def generate_subsequence_views(global_view, n, fraction, context_length, pad_token_id):
    views = []
    seq_len = global_view.size(1)
    subseq_len = int(seq_len * fraction)
    for _ in range(n):
        start = random.randint(0, seq_len - subseq_len)
        subseq = global_view[:, start:start + subseq_len]
        views.append(subseq)
    return pad_to_context_length(views, context_length, pad_token_id)

# Generate masked views
def generate_masked_views(global_view, m, mask_prob, mask_token_id, context_length, pad_token_id):
    views = []
    for _ in range(m):
        masked_view = global_view.clone()
        mask_indices = torch.rand(global_view.size(), device=global_view.device) < mask_prob
        masked_view[mask_indices] = mask_token_id
        views.append(masked_view)
    return pad_to_context_length(views, context_length, pad_token_id)

# DINO loss function
def dino_loss(student_output, teacher_output, tps, tpt, center, loss_type="cls"):
    if loss_type == "cls":
        student_softmax = nn.functional.softmax(student_output[0] / tps, dim=1)  # CLS projection
        teacher_softmax = nn.functional.softmax((teacher_output[0] - center) / tpt, dim=1)  # CLS projection
    elif loss_type == "avg_pool":
        student_softmax = nn.functional.softmax(student_output[1] / tps, dim=1)  # Avg pool projection
        teacher_softmax = nn.functional.softmax((teacher_output[1] - center) / tpt, dim=1)  # Avg pool projection
    else:
        raise ValueError("Invalid loss_type. Choose 'cls' or 'avg_pool'.")
    loss = -(teacher_softmax * torch.log(student_softmax)).sum(dim=1).mean()
    return loss

# Training function
def train_dino(model, teacher_model, dataloader, optimizer, device, num_epochs, 
               n_subseq, m_masked, fraction, mask_prob, mask_token_id, pad_token_id, 
               l, m, tps, tpt, loss_type="cls"):
    """
    Train the DINO-DNA framework with modifications.
    """
    center = torch.zeros(model.projection_head[-1].out_features).to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            global_view = batch["input_ids"].to(device)  # [batch_size, context_length]

            # Generate local views
            subseq_views = generate_subsequence_views(global_view, n_subseq, fraction, model.max_len, pad_token_id)
            masked_views = generate_masked_views(global_view, m_masked, mask_prob, mask_token_id, model.max_len, pad_token_id)
            
            # Combine views for student
            student_views = [global_view] + subseq_views + masked_views

            # Student forward pass
            student_outputs = [model(view) for view in student_views]

            # Teacher forward pass (global view only)
            with torch.no_grad():
                teacher_output = teacher_model(global_view)

            # Compute loss
            loss = 0
            num_pairs = len(student_outputs)
            for s_output in student_outputs:
                print(dino_loss(s_output, teacher_output, tps, tpt, center, loss_type))
                loss += dino_loss(s_output, teacher_output, tps, tpt, center, loss_type)
            loss /= num_pairs

            # Update student
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher with EMA
            for param_s, param_t in zip(model.parameters(), teacher_model.parameters()):
                param_t.data = l * param_t.data + (1 - l) * param_s.data

            # Update center
            with torch.no_grad():
                if loss_type == "cls":
                    batch_output = teacher_output[0]
                else:
                    batch_output = teacher_output[1]
                center = m * center + (1 - m) * batch_output.mean(dim=0)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 10
    embed_dim = 256
    num_layers = 4
    num_heads = 4
    dim_feedforward = 4 * embed_dim
    projection_dim = embed_dim
    max_len = 512  # Context length
    dropout = 0.1
    num_epochs = 10
    n_subseq = 2
    m_masked = 2
    fraction = 0.5
    mask_prob = 0.15
    l = 0.996
    m = 0.996
    tps = 0.1
    tpt = 0.04
    loss_type = "cls"  # or "avg_pool"

    # Tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    dataset = DNADataset(max_length=max_len, dataset_size=5000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    model = DNATransformer_ALiBi(
        vocab_size=4096,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        projection_dim=projection_dim,
        dropout=dropout
    ).to(device)

    teacher_model = DNATransformer_ALiBi(
        vocab_size=4096,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        projection_dim=projection_dim,
        dropout=dropout
    ).to(device)

    teacher_model.load_state_dict(model.state_dict())

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    # Train
    train_dino(model, teacher_model, dataloader, optimizer, device, num_epochs, 
               n_subseq, m_masked, fraction, mask_prob, mask_token_id, pad_token_id, 
               l, m, tps, tpt, loss_type)