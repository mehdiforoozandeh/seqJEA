import torch.nn as nn
import torch.optim as optim
import random, gc, os, torch, math
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import DNADataset
from model import *  
from eval import *
from transformers import AutoTokenizer
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if torch.cuda.device_count() >= 2:
    device_teacher = torch.device("cuda:0")
    device_student = torch.device("cuda:1")
else:
    device_teacher = device_student = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                tokenizer, l, m, tps, tpt, device_student, device_teacher):
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

        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.l = l
        self.m = m
        self.tps = tps
        self.tpt = tpt
        self.device_student = device_student
        self.device_teacher = device_teacher
        
        # Use the base_model's attribute here.
        self.center = torch.zeros(self.model.projection_head[-1].out_features, device=device_student)
        self.benchmark = BenchmarkEvaluator(self.model, self.tokenizer)

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
            Compute the DINO loss between teacher and student outputs.
            
            Both teacher_output and student_output are assumed to be of shape [batch_size, projection_dim].
            Temperature scaling is applied to both, with teacher outputs centered.
        """
        # Stop gradient on teacher.
        teacher_output = teacher_output.detach()
        
        # Apply softmax with temperature scaling.
        s_probs = F.softmax(student_output / tps, dim=1)
        t_probs = F.softmax((teacher_output - center) / tpt, dim=1)
        
        # Cross-entropy loss; add a small epsilon to avoid log(0).
        loss_val = - (t_probs * torch.log(s_probs + 1e-7)).sum(dim=1).mean()
        return loss_val

    def train_dino(self, accumulation_steps=10):
        """
        Train the DINO-DNA framework with gradient accumulation to handle small batch sizes.
        
        For each batch:
        - Computes the teacher output for the global view.
        - Generates subsequence and masked views.
        - Merges all views into a single tensor to compute the student forward pass in one call.
        - Computes loss and normalized entropies for each view.
        - Accumulates gradients over `accumulation_steps` batches.
        - Performs backpropagation and updates the student optimizer after accumulation.
        - Updates the teacher network and the center vector.
        - Cleans up intermediate variables to free memory.

        Args:
            accumulation_steps (int): Number of batches to accumulate gradients over before updating parameters.
        """
        self.teacher_model.eval()
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_teacher_std = 0.0
            total_teacher_entropy = 0.0
            total_student_entropy = 0.0
            batch_count = 0
            step_count = 0  # Counter for gradient accumulation steps
            epoch_kl_div = 0.0  # Accumulate batch average KL divergence for the epoch

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)
            for batch in pbar:
                try:
                    # Do not clear gradients immediately; accumulate them
                    global_view = batch["global"].to(self.device_student)

                    # Compute teacher output on teacher device then move to student device
                    with torch.no_grad():
                        teacher_output = self.teacher_model(global_view.to(self.device_teacher))
                        teacher_output = teacher_output.to(self.device_student)

                    # Compute teacher statistics
                    batch_teacher_std = teacher_output.std(dim=0).mean().item()
                    teacher_probs = F.softmax(teacher_output, dim=1)
                    teacher_entropy = - (teacher_probs * torch.log(teacher_probs + 1e-7)).sum(dim=1).mean().item()
                    max_entropy = math.log(teacher_output.size(1))
                    normalized_teacher_entropy = teacher_entropy / max_entropy

                    # Combine views: global + subsequence + masked
                    student_views = [batch[k] for k in batch.keys()]
                    n_views = len(student_views)
                    merged_views = torch.cat(student_views, dim=0).to(self.device_student)  # [n_views * batch_size, context_length]

                    # Student forward pass in one call
                    merged_student_outputs = self.model(merged_views)  # [n_views * batch_size, projection_dim]
                    batch_size = global_view.size(0)
                    student_outputs = merged_student_outputs.view(n_views, batch_size, -1)

                    # Compute loss and accumulate normalized student entropy
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

                    # Compute KL divergence between teacher and each student view,
                    # then average over views to obtain the batch average.
                    kl_div_sum = 0.0
                    for view_out in student_outputs:
                        # Compute student log probabilities (with temperature scaling).
                        student_log_probs = F.log_softmax(view_out, dim=1)
                        # Compute teacher probabilities (with temperature scaling and center adjustment).
                        teacher_probs_scaled = F.softmax(teacher_output, dim=1)
                        kl_div = F.kl_div(student_log_probs, teacher_probs_scaled, reduction='batchmean')
                        kl_div_sum += kl_div.item()
                    batch_avg_kl_div = kl_div_sum / n_views
                    epoch_kl_div += batch_avg_kl_div

                    # Skip the batch if loss is NaN
                    if torch.isnan(loss):
                        self.optimizer.zero_grad()
                        self.clean_up_intermediates(
                            global_view,  student_views, merged_views, merged_student_outputs, 
                            student_outputs, teacher_output, loss)
                        continue

                    # Scale loss to simulate larger batch size and accumulate gradients
                    scaled_loss = loss / accumulation_steps
                    scaled_loss.backward()  # Accumulates gradients

                    # Update metrics for the current batch
                    total_loss += loss.item()
                    total_teacher_std += batch_teacher_std
                    total_teacher_entropy += normalized_teacher_entropy
                    total_student_entropy += normalized_student_entropy
                    batch_count += 1
                    step_count += 1

                    # Perform optimization step after accumulating gradients for `accumulation_steps` batches
                    if step_count % accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Update teacher network via EMA
                        self.update_teacher()

                        # Update center vector with teacher output
                        with torch.no_grad():
                            self.update_center(teacher_output)

                    # Update tqdm postfix with current batch loss
                    pbar.set_postfix({'loss': f"{loss.item():.3f}"})

                    # Clean up intermediates after each batch
                    self.clean_up_intermediates(
                        global_view, student_views, merged_views, 
                        merged_student_outputs, student_outputs,
                         teacher_output, loss)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("CUDA OOM error encountered, cleaning up and skipping this batch.")
                        self.optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e

            # If there are remaining accumulated gradients at the end of the epoch, update parameters
            if step_count % accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.update_teacher()
                with torch.no_grad():
                    self.update_center(teacher_output)

            # Compute epoch averages
            avg_loss = total_loss / batch_count if batch_count > 0 else float('nan')
            avg_teacher_std = total_teacher_std / batch_count if batch_count > 0 else float('nan')
            avg_teacher_entropy = total_teacher_entropy / batch_count if batch_count > 0 else float('nan')
            avg_student_entropy = total_student_entropy / batch_count if batch_count > 0 else float('nan')
            avg_kl_div = epoch_kl_div / batch_count if batch_count > 0 else float('nan')
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.3}, T_Std: {avg_teacher_std:.3f}, "
                f"T_Ent: {avg_teacher_entropy:.3f}, S_Ent: {avg_student_entropy:.3f}, KL_Div: {avg_kl_div:.3f}")

            # Run benchmarks every 150 epochs
            if (epoch + 1) % 100 == 0:
                self.benchmark.model = self.model
                results =self.benchmark.run_all_benchmarks()
                
####################################
# Usage
####################################
if __name__ == "__main__":
    # Hyperparameters
    model_type = "alibi"
    batch_size = 2
    embed_dim = 384
    num_layers = 6
    num_heads = 6
    dim_feedforward = 2 * embed_dim
    projection_dim = embed_dim
    max_len_seq = 8192  # maximum sequence length for dataset
    context_length = 1024  # model's context length (max_len for transformer)
    dropout = 0.05
    num_epochs = 1000
    fractions = [0.25, 0.5, 0.75]
    # learning_rate = 0.0005*(batch_size*5)/256 # following the dino paper
    learning_rate = 2e-4

    l = 0.995
    m = 0.995
    tps = 0.5
    tpt = 0.05  
    
    # l = 0.99
    # m = 0.9
    # tps = 0.4
    # tpt = 0.04

    num_layers = num_layers // 2
    context_length = context_length // 2
    max_len_seq = max_len_seq // 2
    batch_size *= 8

    # Load tokenizer and obtain token IDs for special tokens.
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
    VOCAB_SIZE = 4096
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    # Create dataset and dataloader.
    dataset = DNADataset(
        min_length=max_len_seq//2, max_length=max_len_seq, 
        context_length=context_length, dataset_size=100, 
        subset_fracs=fractions)

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
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    dino = DINO(model, teacher_model, dataloader, optimizer, num_epochs, 
        tokenizer, l, m, tps, tpt, device_student, device_teacher)

    dino.train_dino()