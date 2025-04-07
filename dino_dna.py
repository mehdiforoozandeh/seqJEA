import torch.nn as nn
import torch.optim as optim
import random, gc, os, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import DNADataset
from model import DNATransformer_ALiBi  # using the ALiBi version
from transformers import AutoTokenizer
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Define two devices: one for the student and one for the teacher.
device_student = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
device_teacher = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device_student)

print(device_student, device_teacher)
####################################
# Utility Functions for Views
####################################

def pad_to_context_length(sequences, context_length, pad_token_id):
    """Pad each sequence (assumed to be a tensor of shape [batch, seq_len]) to context_length."""
    padded_sequences = []
    for seq in sequences:
        if seq.size(1) < context_length:
            padding = torch.full((seq.size(0), context_length - seq.size(1)), pad_token_id, device=seq.device)
            padded_seq = torch.cat([seq, padding], dim=1)
        else:
            padded_seq = seq[:, :context_length]
        padded_sequences.append(padded_seq)
    return padded_sequences

def generate_subsequence_views(global_view, n, fraction, context_length, pad_token_id):
    """
    Generate n local subsequence views from the global_view.
    Each view is a random contiguous subsequence (of length = fraction*global_view length)
    padded to context_length.
    """
    views = []
    seq_len = global_view.size(1)
    subseq_len = int(seq_len * fraction)
    for _ in range(n):
        start = random.randint(0, seq_len - subseq_len)
        subseq = global_view[:, start:start + subseq_len]
        views.append(subseq)
    return pad_to_context_length(views, context_length, pad_token_id)

def generate_masked_views(global_view, m, mask_prob, mask_token_id, context_length, pad_token_id):
    """
    Generate m masked views from the global_view.
    Each masked view randomly replaces tokens with mask_token_id (with probability mask_prob).
    """
    views = []
    for _ in range(m):
        masked_view = global_view.clone()
        mask_indices = torch.rand(global_view.size(), device=global_view.device) < mask_prob
        # Ensure that the CLS token (assumed at index 0) remains unchanged.
        mask_indices[:, 0] = False
        masked_view[mask_indices] = mask_token_id
        views.append(masked_view)
    return pad_to_context_length(views, context_length, pad_token_id)

####################################
# Updated DINO Loss Function
####################################
def dino_loss(student_output, teacher_output, student_temp, teacher_temp, center):
    """
    Compute the DINO loss between teacher and student outputs.
    
    Both teacher_output and student_output are assumed to be of shape [batch_size, projection_dim].
    Temperature scaling is applied to both, with teacher outputs centered.
    """
    # Stop gradient on teacher.
    teacher_output = teacher_output.detach()
    
    # Apply softmax with temperature scaling.
    s_probs = F.softmax(student_output / student_temp, dim=1)
    t_probs = F.softmax((teacher_output - center) / teacher_temp, dim=1)
    
    # Cross-entropy loss; add a small epsilon to avoid log(0).
    loss_val = - (t_probs * torch.log(s_probs + 1e-7)).sum(dim=1).mean()
    return loss_val

####################################
# Training Function with Two GPUs
####################################

import math
import gc
from tqdm import tqdm
import torch.nn.functional as F

# def train_dino(model, teacher_model, dataloader, optimizer, num_epochs, 
#                n_subseq, m_masked, fraction, mask_prob, mask_token_id, pad_token_id, 
#                l, m, tps, tpt, loss_type="cls"):
#     """
#     Train the DINO-DNA framework with modifications.
    
#     - Student network is on device_student, teacher network on device_teacher.
#     - For each batch, additional augmented views (local subsequences and masked views) are generated.
#     - The teacher processes the global view on device_teacher, and its output is moved back to device_student.
#     - The loss is computed over all teacher-student pairs.
#     - If NaN loss or CUDA OOM error is detected, the update for that batch is skipped.
#     - Student parameters are updated by backpropagation; teacher parameters are updated via EMA.
#     - The center vector is updated based on teacher outputs.
#     - At the end of each epoch, the average loss, average teacher feature standard deviation, normalized teacher entropy, 
#       and normalized student entropy (averaged across all views) are printed to monitor for collapse.
#     """
#     # Initialize center vector from the projection dimension (on device_student).
#     center = torch.zeros(model.projection_head[-1].out_features, device=device_student)
    
#     for epoch in range(num_epochs):
#         total_loss = 0.0
#         total_teacher_std = 0.0
#         total_teacher_entropy = 0.0
#         total_student_entropy = 0.0
#         batch_count = 0
        
#         # Wrap dataloader with tqdm for progress visualization.
#         for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
#             try:
#                 optimizer.zero_grad()
#                 # Move global view to student device.
#                 global_view = batch["input_ids"].to(device_student)
                
#                 # Generate additional views.
#                 subseq_views = generate_subsequence_views(global_view, n_subseq, fraction, model.max_len, pad_token_id)
#                 masked_views = generate_masked_views(global_view, m_masked, mask_prob, mask_token_id, model.max_len, pad_token_id)
                
#                 # Combine views for the student.
#                 student_views = [global_view] + subseq_views + masked_views

#                 # Student forward pass on all views.
#                 student_outputs = [model(view) for view in student_views]
                
#                 # Teacher forward pass on the global view: move global view to teacher device.
#                 with torch.no_grad():
#                     teacher_output = teacher_model(global_view.to(device_teacher))
#                     teacher_output = teacher_output.to(device_student)
                    
#                     # Compute teacher feature standard deviation (across batch).
#                     batch_teacher_std = teacher_output.std(dim=0).mean().item()
                    
#                     # Compute teacher entropy without centering or temperature scaling.
#                     teacher_probs = F.softmax(teacher_output, dim=1)
#                     teacher_entropy = - (teacher_probs * torch.log(teacher_probs + 1e-7)).sum(dim=1).mean().item()
#                     # Normalize teacher entropy by maximum possible entropy: log(num_features).
#                     max_entropy = math.log(teacher_output.size(1))
#                     normalized_teacher_entropy = teacher_entropy / max_entropy
                    
#                     # Compute student entropy for each student view without temperature scaling.
#                     student_entropies = []
#                     for s_out in student_outputs:
#                         s_probs = F.softmax(s_out, dim=1)
#                         s_entropy = - (s_probs * torch.log(s_probs + 1e-7)).sum(dim=1).mean().item()
#                         student_entropies.append(s_entropy)
#                     avg_student_entropy = sum(student_entropies) / len(student_entropies)
#                     normalized_student_entropy = avg_student_entropy / max_entropy
                
#                 # Compute loss: average DINO loss over all student views.
#                 loss = 0
#                 num_pairs = len(student_outputs)
#                 for s_out in student_outputs:
#                     loss += dino_loss(s_out, teacher_output, tps, tpt, center)
#                 loss /= num_pairs

#                 # Check if loss is NaN.
#                 if torch.isnan(loss):
#                     # print("NaN loss detected, skipping parameter update for this batch.")
#                     optimizer.zero_grad()
#                     del global_view, subseq_views, masked_views, student_views, student_outputs, teacher_output, loss
#                     torch.cuda.empty_cache()
#                     gc.collect()
#                     continue

#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print("CUDA OOM error encountered, cleaning up and skipping this batch.")
#                     optimizer.zero_grad()
#                     # del global_view, subseq_views, masked_views, student_views, student_outputs
#                     torch.cuda.empty_cache()
#                     gc.collect()
#                     continue
#                 else:
#                     raise e

#             # Backward pass.
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # Update teacher network using EMA.
#             for param_s, param_t in zip(model.parameters(), teacher_model.parameters()):
#                 param_t.data = l * param_t.data + (1 - l) * param_s.data.to(device_teacher)

#             # Update center using teacher output.
#             with torch.no_grad():
#                 center = m * center + (1 - m) * teacher_output.mean(dim=0)

#             total_loss += loss.item()
#             total_teacher_std += batch_teacher_std
#             total_teacher_entropy += normalized_teacher_entropy
#             total_student_entropy += normalized_student_entropy
#             batch_count += 1

#             # Clean up intermediate variables.
#             del global_view, subseq_views, masked_views, student_views, student_outputs, teacher_output, loss
#             torch.cuda.empty_cache()
#             gc.collect()
        
#         # Compute epoch averages.
#         avg_loss = total_loss / batch_count if batch_count > 0 else float('nan')
#         avg_teacher_std = total_teacher_std / batch_count if batch_count > 0 else float('nan')
#         avg_teacher_entropy = total_teacher_entropy / batch_count if batch_count > 0 else float('nan')
#         avg_student_entropy = total_student_entropy / batch_count if batch_count > 0 else float('nan')
#         print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.3}, Avg T_Std: {avg_teacher_std:.3f}, "
#               f"Avg T_Ent: {avg_teacher_entropy:.3f}, Avg S_Ent: {avg_student_entropy:.3}")

import torch
from torch.cuda.amp import autocast, GradScaler

def train_dino(model, teacher_model, dataloader, optimizer, num_epochs, 
               n_subseq, m_masked, fraction, mask_prob, mask_token_id, pad_token_id, 
               l, m, tps, tpt):
    # Initialize mixed precision scaler
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Zero gradients at the start of the batch
            optimizer.zero_grad()
            
            # Move global view to student device
            global_view = batch["input_ids"].to(device_student)  # Shape: [batch_size, context_length]
            
            # Teacher processes global view (no gradients)
            with torch.no_grad():
                teacher_output = teacher_model(global_view.to(device_teacher)).to(device_student)
            
            # Generate augmented views
            views = [global_view]  # Start with global view
            subseq_views = generate_subsequence_views(global_view, n_subseq, fraction, pad_token_id)
            masked_views = generate_masked_views(global_view, m_masked, mask_prob, mask_token_id, pad_token_id)
            views.extend(subseq_views)
            views.extend(masked_views)
            
            # Process each view sequentially
            total_loss = 0.0
            num_views = len(views)
            for view in views:
                with autocast():
                    # Student forward pass
                    student_output = model(view)
                    # Compute DINO loss (e.g., cross-entropy between student and teacher outputs)
                    loss = dino_loss(student_output, teacher_output, tps, tpt, center=l, momentum=m)
                    total_loss += loss.item()
                    
                    # Scale loss and accumulate gradients
                    scaled_loss = loss / num_views
                    scaler.scale(scaled_loss).backward()
                
                # Free memory by deleting intermediate tensors
                del view, student_output, loss, scaled_loss
                torch.cuda.empty_cache()
            
            # Perform optimizer step with accumulated gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update teacher weights (e.g., exponential moving average) and center
            with torch.no_grad():
                for param_t, param_s in zip(teacher_model.parameters(), model.parameters()):
                    param_t.data = tps * param_t.data + (1 - tps) * param_s.data.to(device_teacher)
                # Update center if applicable
                # center = update_center(center, teacher_output, momentum=m)
            
            # Optional: Log total_loss for monitoring
            print(f"Epoch {epoch}, Loss: {total_loss / num_views:.4f}")

####################################
# Example Usage
####################################
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 10
    embed_dim = 512
    num_layers = 4
    num_heads = 4
    dim_feedforward = 2 * embed_dim
    projection_dim = embed_dim
    max_len_seq = 20000  # maximum sequence length for dataset
    context_length = 500  # model's context length (max_len for transformer)
    dropout = 0.1
    num_epochs = 100
    n_subseq = 2
    m_masked = 2
    fraction = 0.8
    mask_prob = 0.2
    l = 0.996
    m = 0.996
    tps = 0.1
    tpt = 0.07  # Adjusted teacher temperature for stability
    loss_type = "cls"  # Using average pooling representation

    # Load tokenizer and obtain token IDs for special tokens.
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    # Create dataset and dataloader.
    dataset = DNADataset(min_length=max_len_seq//2, max_length=max_len_seq, dataset_size=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate student and teacher models on their respective devices.
    model = DNATransformer_ALiBi(
        vocab_size=4096,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        max_len=context_length,
        projection_dim=projection_dim,
        dropout=dropout
    ).to(device_student)

    teacher_model = DNATransformer_ALiBi(
        vocab_size=4096,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        max_len=context_length,
        projection_dim=projection_dim,
        dropout=dropout
    ).to(device_teacher)

    # Initialize teacher with student's weights.
    teacher_model.load_state_dict(model.state_dict())

    # Optimizer for student model.
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the DINO-DNA framework.
    train_dino(model, teacher_model, dataloader, optimizer, num_epochs, 
               n_subseq, m_masked, fraction, mask_prob, mask_token_id, pad_token_id, 
               l, m, tps, tpt)