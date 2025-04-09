import torch, pysam, random
from transformers import AutoTokenizer

def get_embeddings(model, tokenizer, sequences, batch_size=32):
    """
    Given a UnifiedDNATransformer model, an initialized tokenizer, and a list of DNA sequences,
    this function computes the CLS token embeddings in batches.

    It:
      1. Tokenizes sequences in batches using the provided tokenizer (with padding and truncation).
      2. Runs the model (in evaluation mode) on each batch.
      3. Extracts and collects the CLS token embeddings.
      4. Concatenates the per-batch embeddings and returns the final tensor.

    Args:
        model (UnifiedDNATransformer): An instance of the unified transformer model.
        tokenizer: An initialized tokenizer object (e.g., from AutoTokenizer).
        sequences (list[str]): A list of DNA sequences (e.g., ["ATCTG", "GATTACA", ...]).
        batch_size (int, optional): Number of sequences to process per batch (default is 32).

    Returns:
        torch.Tensor: A tensor of shape [num_sequences, projection_dim] containing the projected
                      CLS token embeddings for each sequence.
    """
    # List to collect embeddings from each batch
    embeddings_list = []
    
    # Determine the device from the model's parameters
    device = next(model.parameters()).device
    
    # Set the model to evaluation mode and disable gradient computation
    model.eval()
    with torch.no_grad():
        # Process sequences in batches
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i: i + batch_size]
            # Tokenize the batch; padding=True and truncation=True ensure all sequences have equal length
            tokenized = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            # Move the token IDs to the model's device
            input_ids = tokenized["input_ids"].to(device)
            # Run the forward pass to obtain the CLS embedding projection
            batch_embeddings = model(input_ids)
            # Optional: move embeddings to CPU for storage
            embeddings_list.append(batch_embeddings.cpu())
    
    # Concatenate all batch embeddings into a single tensor
    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings
