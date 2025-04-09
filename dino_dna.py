import torch.nn as nn
import torch.optim as optim
import random, gc, os, torch, math
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import DNADataset
from model import *  # using the ALiBi version
from transformers import AutoTokenizer
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device_teacher = device_student = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
# DINO Loss Function
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
# DINO Training Function
####################################

def train_dino(model, teacher_model, dataloader, optimizer, num_epochs, 
               n_subseq, m_masked, fraction, mask_prob, mask_token_id, pad_token_id, 
               l, m, tps, tpt):
    """
    Train the DINO-DNA framework with improved memory management.

    Changes made:
    - Merge global, subsequence, and masked views into one tensor (i.e. one batch) so that the 
      student network forward pass is computed in a single call.
    - Reshape the merged output to separate the different views.
    - Compute losses and entropies per view, then average.
    - Remove intermediate tensors as soon as they are no longer needed.
    """
    # Initialize center vector from the projection dimension (on device_student).
    center = torch.zeros(model.projection_head[-1].out_features, device=device_student)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_teacher_std = 0.0
        total_teacher_entropy = 0.0
        total_student_entropy = 0.0
        batch_count = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            try:
                optimizer.zero_grad()
                # Move global view to student device.
                global_view = batch["input_ids"].to(device_student)
                
                # Teacher forward pass on the global view (computed once).
                with torch.no_grad():
                    teacher_output = teacher_model(global_view.to(device_teacher))
                    teacher_output = teacher_output.to(device_student)
                
                # Compute teacher statistics.
                batch_teacher_std = teacher_output.std(dim=0).mean().item()
                teacher_probs = F.softmax(teacher_output, dim=1)
                teacher_entropy = - (teacher_probs * torch.log(teacher_probs + 1e-7)).sum(dim=1).mean().item()
                max_entropy = math.log(teacher_output.size(1))
                normalized_teacher_entropy = teacher_entropy / max_entropy
                
                # Generate additional views.
                subseq_views = generate_subsequence_views(global_view, n_subseq, fraction, model.max_len, pad_token_id)
                masked_views = generate_masked_views(global_view, m_masked, mask_prob, mask_token_id, model.max_len, pad_token_id)
                
                # Combine views: global + subsequence + masked.
                student_views = [global_view] + subseq_views + masked_views
                n_views = len(student_views)  # e.g., 1 + n_subseq + m_masked
                
                # Merge all views into one tensor along the batch dimension.
                # Each view should have shape [batch_size, context_length]
                merged_views = torch.cat(student_views, dim=0)  # [n_views * batch_size, context_length]
                
                # Student forward pass in one batch.
                merged_student_outputs = model(merged_views)  # [n_views * batch_size, projection_dim]
                
                # Reshape to separate views: [n_views, batch_size, projection_dim]
                batch_size = global_view.size(0)
                student_outputs = merged_student_outputs.view(n_views, batch_size, -1)
                
                # Compute loss and student entropy for each view.
                loss_sum = 0.0
                student_entropies = []
                for view_out in student_outputs:
                    # Compute DINO loss per view.
                    loss_sum += dino_loss(view_out, teacher_output, tps, tpt, center)
                    # Compute entropy for the student view.
                    s_probs = F.softmax(view_out, dim=1)
                    s_entropy = - (s_probs * torch.log(s_probs + 1e-7)).sum(dim=1).mean().item()
                    student_entropies.append(s_entropy)
                
                loss = loss_sum / n_views
                avg_student_entropy = sum(student_entropies) / len(student_entropies)
                normalized_student_entropy = avg_student_entropy / max_entropy
                
                # Check if loss is NaN.
                if torch.isnan(loss):
                    optimizer.zero_grad()
                    del (global_view, subseq_views, masked_views, student_views, 
                         merged_views, merged_student_outputs, student_outputs, teacher_output, loss)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("CUDA OOM error encountered, cleaning up and skipping this batch.")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher network using EMA.
            for param_s, param_t in zip(model.parameters(), teacher_model.parameters()):
                param_t.data = l * param_t.data + (1 - l) * param_s.data.to(device_teacher)

            # Update center vector using teacher output.
            with torch.no_grad():
                center = m * center + (1 - m) * teacher_output.mean(dim=0)

            total_loss += loss.item()
            total_teacher_std += batch_teacher_std
            total_teacher_entropy += normalized_teacher_entropy
            total_student_entropy += normalized_student_entropy
            batch_count += 1

            # Clean up intermediate variables.
            del (global_view, subseq_views, masked_views, student_views, 
                 merged_views, merged_student_outputs, student_outputs, teacher_output, loss)
            torch.cuda.empty_cache()
            gc.collect()
        
        # Compute and print epoch averages.
        avg_loss = total_loss / batch_count if batch_count > 0 else float('nan')
        avg_teacher_std = total_teacher_std / batch_count if batch_count > 0 else float('nan')
        avg_teacher_entropy = total_teacher_entropy / batch_count if batch_count > 0 else float('nan')
        avg_student_entropy = total_student_entropy / batch_count if batch_count > 0 else float('nan')
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.3}, Avg T_Std: {avg_teacher_std:.3f}, "
              f"Avg T_Ent: {avg_teacher_entropy:.3f}, Avg S_Ent: {avg_student_entropy:.3}")

####################################

class DINO:
    """
    DINO class for training a student and teacher network in a DINO-DNA framework.
    
    This class contains utility methods:
      - pad_to_context_length: Pad each view to a fixed context length.
      - generate_subsequence_views: Generate local subsequence views from the global view.
      - generate_masked_views: Generate masked views by randomly masking tokens.
      - compute_normalized_entropy: Compute the normalized entropy of a view.
      - update_teacher: Update teacher network parameters via EMA from the student.
      - update_center: Update the center vector as the moving mean of teacher outputs.
      - clean_up_intermediates: Clean up intermediate variables and clear caches.
      - train_dino: Run the overall training loop.
    
    The dino_loss() method is a placeholder and should be replaced with your actual loss.
    """
    def __init__(self, model, teacher_model, dataloader, optimizer, num_epochs, 
                 n_subseq, m_masked, fraction, mask_prob, mask_token_id, pad_token_id, 
                 l, m, tps, tpt, device_student, device_teacher):
        """
        Initialize the DINO training framework.
        
        Args:
            model (nn.Module): The student network.
            teacher_model (nn.Module): The teacher network.
            dataloader (iterable): DataLoader that yields batches.
            optimizer (torch.optim.Optimizer): Optimizer for the student network.
            num_epochs (int): Number of epochs to train.
            n_subseq (int): Number of local subsequence views to generate.
            m_masked (int): Number of masked views to generate.
            fraction (float): Fraction of the sequence length used for subsequence views.
            mask_prob (float): Token masking probability for masked views.
            mask_token_id (int): Token ID to replace masked tokens.
            pad_token_id (int): Token ID used for padding.
            l (float): EMA decay coefficient for updating teacher parameters.
            m (float): Coefficient used to update the center vector.
            tps, tpt: Temperature parameters (passed into the loss function).
            device_student (torch.device): Device where the student model is located.
            device_teacher (torch.device): Device where the teacher model is located.
        """
        self.model = model.to(device_student)
        self.teacher_model = teacher_model.to(device_teacher)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.n_subseq = n_subseq
        self.m_masked = m_masked
        self.fraction = fraction
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.l = l
        self.m = m
        self.tps = tps
        self.tpt = tpt
        self.device_student = device_student
        self.device_teacher = device_teacher
        
        # Initialize center vector from the output (projection) dimension on student device.
        self.center = torch.zeros(self.model.projection_head[-1].out_features, device=device_student)

    def pad_to_context_length(self, sequences, context_length):
        """
        Pad each sequence (view) to a fixed context length.
        
        Args:
            sequences (list[Tensor]): List of tensors each of shape [batch, seq_len].
            context_length (int): Desired final sequence length.
        
        Returns:
            list[Tensor]: List with each tensor padded (or truncated) to context_length.
        """
        padded_sequences = []
        for seq in sequences:
            if seq.size(1) < context_length:
                padding = torch.full((seq.size(0), context_length - seq.size(1)), 
                                     self.pad_token_id, device=seq.device)
                padded_seq = torch.cat([seq, padding], dim=1)
            else:
                padded_seq = seq[:, :context_length]
            padded_sequences.append(padded_seq)
        return padded_sequences

    def generate_subsequence_views(self, global_view):
        """
        Generate local subsequence views from the global view.
        Each view is a random contiguous subsequence (of length = fraction * seq_len)
        padded to the network's maximum context length.
        
        Args:
            global_view (Tensor): [batch, seq_len] tensor (global view).
        
        Returns:
            list[Tensor]: List of subsequence view tensors.
        """
        views = []
        seq_len = global_view.size(1)
        subseq_len = int(seq_len * self.fraction)
        for _ in range(self.n_subseq):
            start = torch.randint(0, seq_len - subseq_len + 1, (1,)).item()
            subseq = global_view[:, start:start+subseq_len]
            views.append(subseq)
        return self.pad_to_context_length(views, self.model.max_len)

    def generate_masked_views(self, global_view):
        """
        Generate masked views by randomly replacing tokens with mask_token_id.
        The CLS token (assumed at index 0) remains unmasked.
        
        Args:
            global_view (Tensor): [batch, seq_len] tensor (global view).
        
        Returns:
            list[Tensor]: List of masked view tensors.
        """
        views = []
        for _ in range(self.m_masked):
            masked_view = global_view.clone()
            mask_indices = torch.rand(global_view.size(), device=global_view.device) < self.mask_prob
            mask_indices[:, 0] = False  # preserve CLS token
            masked_view[mask_indices] = self.mask_token_id
            views.append(masked_view)
        return self.pad_to_context_length(views, self.model.max_len)

    def compute_normalized_entropy(self, outputs):
        """
        Compute the normalized entropy for a given output distribution.
        
        Args:
            outputs (Tensor): Logits or projections of shape [batch, dim].
        
        Returns:
            float: Normalized entropy (entropy divided by the maximum entropy).
        """
        probs = F.softmax(outputs, dim=1)
        entropy = - (probs * torch.log(probs + 1e-7)).sum(dim=1).mean().item()
        max_entropy = math.log(outputs.size(1))
        return entropy / max_entropy

    def update_teacher(self):
        """
        Update teacher network parameters using exponential moving average (EMA)
        from the student network. self.l is the decay coefficient.
        """
        for param_s, param_t in zip(self.model.parameters(), self.teacher_model.parameters()):
            param_t.data = self.l * param_t.data + (1 - self.l) * param_s.data.to(self.device_teacher)

    def update_center(self, teacher_output):
        """
        Update the center vector using the teacher output.
        self.m is the update factor.
        
        Args:
            teacher_output (Tensor): Teacher network output [batch, dim].
        """
        self.center = self.m * self.center + (1 - self.m) * teacher_output.mean(dim=0)

    def clean_up_intermediates(self, *args):
        """
        Delete intermediate variables, clear CUDA cache, and trigger garbage collection.
        """
        # Delete provided objects (they will be removed from local scope)
        del args
        torch.cuda.empty_cache()
        gc.collect()

    def dino_loss(self, student_output, teacher_output, tps, tpt, center):
        """
        A placeholder implementation of the DINO loss.
        In practice, DINO loss involves cross-view comparisons, temperature scaling, and centering.
        Here, for demonstration, we compute the mean squared error between normalized outputs.
        
        Args:
            student_output (Tensor): Student network output [batch, projection_dim].
            teacher_output (Tensor): Teacher network output [batch, projection_dim].
            tps, tpt: Temperature parameters.
            center (Tensor): Center vector for student output centering.
        
        Returns:
            Tensor: Computed loss.
        """
        # Normalize outputs after centering for the student.
        student_norm = F.normalize(student_output - center, p=2, dim=1)
        teacher_norm = F.normalize(teacher_output, p=2, dim=1)
        loss = F.mse_loss(student_norm, teacher_norm)
        return loss

    def train_dino(self):
        """
        Train the DINO-DNA framework.
        
        For each batch:
         - Computes the teacher output for the global view.
         - Generates subsequence and masked views.
         - Merges all views into a single tensor to compute the student forward pass in one call.
         - Computes loss and normalized entropies for each view.
         - Performs backpropagation and updates the student optimizer.
         - Updates the teacher network and the center vector.
         - Cleans up intermediate variables to free memory.
        """
        self.teacher_model.eval()
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_teacher_std = 0.0
            total_teacher_entropy = 0.0
            total_student_entropy = 0.0
            batch_count = 0

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False):
                try:
                    self.optimizer.zero_grad()
                    # Move global view (input_ids) to the student device.
                    global_view = batch["input_ids"].to(self.device_student)
                    
                    # Compute teacher output on teacher device then move to student device.
                    with torch.no_grad():
                        teacher_output = self.teacher_model(global_view.to(self.device_teacher))
                        teacher_output = teacher_output.to(self.device_student)
                    
                    # Compute teacher statistics.
                    batch_teacher_std = teacher_output.std(dim=0).mean().item()
                    teacher_probs = F.softmax(teacher_output, dim=1)
                    teacher_entropy = - (teacher_probs * torch.log(teacher_probs + 1e-7)).sum(dim=1).mean().item()
                    max_entropy = math.log(teacher_output.size(1))
                    normalized_teacher_entropy = teacher_entropy / max_entropy

                    # Generate additional views.
                    subseq_views = self.generate_subsequence_views(global_view)
                    masked_views = self.generate_masked_views(global_view)

                    # Combine views: global + subsequence + masked.
                    student_views = [global_view] + subseq_views + masked_views
                    n_views = len(student_views)
                    merged_views = torch.cat(student_views, dim=0)  # shape: [n_views * batch_size, context_length]

                    # Student forward pass in one call.
                    merged_student_outputs = self.model(merged_views)  # [n_views * batch_size, projection_dim]
                    batch_size = global_view.size(0)
                    student_outputs = merged_student_outputs.view(n_views, batch_size, -1)

                    # Compute loss and accumulate normalized student entropy.
                    loss_sum = 0.0
                    student_entropies = []
                    for view_out in student_outputs:
                        loss_sum += self.dino_loss(view_out, teacher_output, self.tps, self.tpt, self.center)
                        s_probs = F.softmax(view_out, dim=1)
                        s_entropy = - (s_probs * torch.log(s_probs + 1e-7)).sum(dim=1).mean().item()
                        student_entropies.append(s_entropy)
                    
                    loss = loss_sum / n_views
                    avg_student_entropy = sum(student_entropies) / len(student_entropies)
                    normalized_student_entropy = avg_student_entropy / max_entropy

                    # Skip the batch if loss is NaN.
                    if torch.isnan(loss):
                        self.optimizer.zero_grad()
                        self.clean_up_intermediates(global_view, subseq_views, masked_views, student_views,
                                                    merged_views, merged_student_outputs, student_outputs, teacher_output, loss)
                        continue

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("CUDA OOM error encountered, cleaning up and skipping this batch.")
                        self.optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e

                # Backward pass and optimization step.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update teacher network via EMA.
                self.update_teacher()

                # Update center vector with teacher output.
                with torch.no_grad():
                    self.update_center(teacher_output)

                total_loss += loss.item()
                total_teacher_std += batch_teacher_std
                total_teacher_entropy += normalized_teacher_entropy
                total_student_entropy += normalized_student_entropy
                batch_count += 1

                self.clean_up_intermediates(global_view, subseq_views, masked_views, student_views,
                                            merged_views, merged_student_outputs, student_outputs, teacher_output, loss)

            # Compute epoch averages.
            avg_loss = total_loss / batch_count if batch_count > 0 else float('nan')
            avg_teacher_std = total_teacher_std / batch_count if batch_count > 0 else float('nan')
            avg_teacher_entropy = total_teacher_entropy / batch_count if batch_count > 0 else float('nan')
            avg_student_entropy = total_student_entropy / batch_count if batch_count > 0 else float('nan')
            print(f"Epoch {epoch+1}/{self.num_epochs}, Avg Loss: {avg_loss:.3}, Avg T_Std: {avg_teacher_std:.3f}, "
                  f"Avg T_Ent: {avg_teacher_entropy:.3f}, Avg S_Ent: {avg_student_entropy:.3f}")




####################################
# Example Usage
####################################
if __name__ == "__main__":
    # Hyperparameters

    model_type = "alibi"
    batch_size = 10
    embed_dim = 384
    num_layers = 6
    num_heads = 6
    dim_feedforward = 2 * embed_dim
    projection_dim = embed_dim
    max_len_seq = 10000  # maximum sequence length for dataset
    context_length = 512  # model's context length (max_len for transformer)
    dropout = 0.1
    num_epochs = 100
    n_subseq = 2
    m_masked = 2
    fraction = 0.5
    mask_prob = 0.5
    l = 0.99
    m = 0.99
    tps = 0.5
    tpt = 0.07  # Adjusted teacher temperature for stability

    # Load tokenizer and obtain token IDs for special tokens.
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
    VOCAB_SIZE = 4096
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    # Create dataset and dataloader.
    dataset = DNADataset(
        min_length=max_len_seq//2, max_length=max_len_seq, 
        context_length=context_length, dataset_size=100)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate student and teacher models on their respective devices.
    model = UnifiedDNATransformer(
        model_type, 
        vocab_size=VOCAB_SIZE, 
        embed_dim=embed_dim, 
        num_layers=num_layers, 
        num_heads=num_heads,
        dim_feedforward=dim_feedforward, 
        max_len=context_length, 
        projection_dim=embed_dim, 
        dropout=dropout
    ).to(device_student)

    teacher_model = UnifiedDNATransformer(
        model_type, 
        vocab_size=VOCAB_SIZE, 
        embed_dim=embed_dim, 
        num_layers=num_layers, 
        num_heads=num_heads,
        dim_feedforward=dim_feedforward, 
        max_len=context_length, 
        projection_dim=embed_dim, 
        dropout=dropout
    ).to(device_teacher)

    # Initialize teacher with student's weights.
    teacher_model.load_state_dict(model.state_dict())

    # Optimizer for student model.
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the DINO-DNA framework.
    # train_dino(model, teacher_model, dataloader, optimizer, num_epochs, 
    #            n_subseq, m_masked, fraction, mask_prob, mask_token_id, pad_token_id, 
    #            l, m, tps, tpt)


    dino = DINO(model, teacher_model, dataloader, optimizer, num_epochs, 
        n_subseq, m_masked, fraction, mask_prob, mask_token_id, pad_token_id, 
        l, m, tps, tpt, device_student, device_teacher)