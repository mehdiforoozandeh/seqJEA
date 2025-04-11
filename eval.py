import torch, pysam, random
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def get_embeddings(model, tokenizer, sequences, context_length, batch_size=64):
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
    
    model.eval()
    with torch.no_grad():
        # Process sequences in batches
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i: i + batch_size]

            tokenized = tokenizer(
                batch, return_tensors="pt", 
                truncation=True,
                padding="max_length",
                max_length=context_length)

            input_ids = tokenized["input_ids"].to(device)
            batch_embeddings = model(input_ids)
            embeddings_list.append(batch_embeddings.cpu())
    
    # Concatenate all batch embeddings into a single tensor
    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings

class BenchmarkEvaluator:
    """
    BenchmarkEvaluator trains a linear probe on a benchmark classification task
    and evaluates it using the ROC-AUC metric.
    
    Each benchmark is assumed to have two CSV files in the specified directory:
      - train.csv (for training the probe)
      - dev.csv (for evaluation)
    
    Each CSV must include the columns:
      - 'sequence': a string representation of a DNA sequence.
      - 'label': a binary label (0 or 1).
    """
    def __init__(
        self, model, tokenizer, 
        benchmark_dirs=[
            "GUE/prom/prom_300_tata/", 
            "GUE/prom/prom_core_tata/",
            "GUE/EMP/H4/",
            # "GUE/EMP/H3/",
            # "GUE/splice/reconstructed/",
            "GUE/tf/4/"], 
        batch_size=64, mode="dev"):
        """
        Initialize the evaluator with the model, tokenizer, and benchmark directories.
        
        Args:
            model (nn.Module): Trained instance of UnifiedDNATransformer.
            tokenizer: Pre-initialized tokenizer object.
            benchmark_dirs (list[str]): List of directory paths for benchmark tasks.
                Each directory should contain train.csv and dev.csv.
            batch_size (int): Batch size for extracting embeddings.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.benchmark_dirs = benchmark_dirs
        self.mode = mode
        self.context_length = self.model.max_len

    def train_probe(self, train_csv):
        """
        Train a linear probe on the training CSV.
        
        Args:
            train_csv (str): File path to train.csv.
        
        Returns:
            probe (sklearn.linear_model.LogisticRegression): The trained linear classifier.
        """
        df = pd.read_csv(train_csv)
        sequences = df['sequence'].tolist()
        labels = df['label'].tolist()
        embeddings = get_embeddings(self.model, self.tokenizer, sequences, self.context_length, self.batch_size)
        X = embeddings.numpy()
        y = np.array(labels)
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)
        return probe

    def evaluate_probe(self, probe, dev_csv):
        """
        Evaluate a trained probe using dev.csv and compute the ROC-AUC score.
        
        Args:
            probe: Trained linear classifier (e.g., LogisticRegression).
            dev_csv (str): File path to dev.csv.
        
        Returns:
            float: ROC AUC score.
        """
        # Load development data.
        df = pd.read_csv(dev_csv)
        sequences = df['sequence'].tolist()
        labels = df['label'].tolist()
        
        # Extract embeddings from the model.
        embeddings = get_embeddings(self.model, self.tokenizer, sequences, self.batch_size)
        X = embeddings.numpy()
        y = np.array(labels)
        
        # Predict probabilities for the positive class.
        # If the problem is binary classification (two unique labels), then predict_proba returns two columns.
        # Otherwise, for multiclass, we need to tell roc_auc_score how to handle them.
        probs = probe.predict_proba(X)
        
        # If binary classification, use only the probability for the positive class.
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y, probs[:, 1])
        else:
            auc = roc_auc_score(y, probs, multi_class='ovr')
        return auc

    def run_benchmark(self, benchmark_dir):
        """
        Run a single benchmark by training a probe on train.csv and evaluating on dev.csv.
        
        Args:
            benchmark_dir (str): Path to benchmark directory containing train.csv and dev.csv.
        
        Returns:
            float: ROC AUC score on the dev set.
        """
        train_csv = f"{benchmark_dir}/train.csv"
        dev_csv = f"{benchmark_dir}/{self.mode}.csv"
        # print(f"Training probe on {train_csv}")
        probe = self.train_probe(train_csv)
        auc = self.evaluate_probe(probe, dev_csv)
        # print(f"Benchmark: {benchmark_dir}, AUC ROC: {auc:.4f}")
        return auc

    def run_all_benchmarks(self, verbose=True):
        """
        Run all benchmarks provided in the initialization.
        
        For each benchmark, the method:
         - Trains a linear probe on train.csv.
         - Evaluates the probe on dev.csv using the ROC-AUC metric.
         
        If verbose is True, results are printed in a table format.
        
        Args:
            verbose (bool, optional): If True, prints a table of benchmark results.
            
        Returns:
            dict: Dictionary mapping benchmark directory to its ROC-AUC score.
        """
        results = {}
        for benchmark in self.benchmark_dirs:
            print(f"Running probing benchmarks {benchmark}...")
            auc = self.run_benchmark(benchmark)
            results[benchmark] = auc
        
        if verbose:
            # Print a table-style result.
            header = f"{'Benchmark':<50} {'AUC ROC':<10}"
            print("\n" + header)
            print("-" * (len(header) + 10))
            for bench, auc in results.items():
                print(f"{bench:<50} {auc:<10.4f}")
            print("-" * (len(header) + 10))
        return results

