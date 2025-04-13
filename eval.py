import torch, pysam, random
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score

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
    BenchmarkEvaluator trains a linear probe on a classification task and a linear regressor for predicting
    GC content of DNA sequences using the same embeddings for both tasks. It evaluates the classification
    performance using the ROC-AUC metric and the regression performance using the R² score.
    
    Each benchmark directory should have:
      - train.csv (for training both probes)
      - dev.csv (or <mode>.csv, for evaluation)
      
    Both CSVs must include:
      - 'sequence': a string representing the DNA sequence.
      - 'label': a binary label (0 or 1) for classification.
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
        Initialize the evaluator.
        
        Args:
            model (nn.Module): Trained instance of UnifiedDNATransformer.
            tokenizer: Pre-initialized tokenizer object.
            benchmark_dirs (list[str]): List of directories containing train.csv and dev.csv.
            batch_size (int): Batch size used for extracting embeddings.
            mode (str): The evaluation CSV mode (typically 'dev').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.benchmark_dirs = benchmark_dirs
        self.mode = mode
        self.context_length = self.model.max_len

    def gc_content(self, sequence):
        """
        Compute GC content for a DNA sequence.
        
        Args:
            sequence (str): DNA sequence.
            
        Returns:
            float: Fraction of bases that are G or C.
        """
        sequence = sequence.upper()
        if len(sequence) == 0:
            return 0.0
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)

    def train_probes(self, train_csv):
        """
        Generate embeddings from the training CSV once, and use them to train both the 
        classification probe and the GC regression probe.
        
        Args:
            train_csv (str): Path to train.csv.
            
        Returns:
            tuple: (classification probe, GC regressor)
        """
        df = pd.read_csv(train_csv)
        sequences = df['sequence'].tolist()
        class_labels = np.array(df['label'].tolist())
        # Compute GC content targets
        gc_labels = np.array([self.gc_content(seq) for seq in sequences])
        
        # Generate embeddings once.
        embeddings = get_embeddings(self.model, self.tokenizer, sequences, self.context_length, self.batch_size)
        X = embeddings.numpy()
        
        # Train classification probe.
        class_probe = LogisticRegression(max_iter=1000)
        class_probe.fit(X, class_labels)
        
        # Train GC regressor.
        gc_regressor = LinearRegression()
        gc_regressor.fit(X, gc_labels)
        
        return class_probe, gc_regressor

    def evaluate_probes(self, class_probe, gc_regressor, dev_csv):
        """
        Generate embeddings from the development CSV once, and use them to evaluate both probes.
        
        Args:
            class_probe: The trained classification probe.
            gc_regressor: The trained GC regressor.
            dev_csv (str): Path to the dev CSV.
            
        Returns:
            tuple: (ROC-AUC score for classification, R² score for GC regression)
        """
        df = pd.read_csv(dev_csv)
        sequences = df['sequence'].tolist()
        class_labels = np.array(df['label'].tolist())
        gc_labels = np.array([self.gc_content(seq) for seq in sequences])
        
        # Generate embeddings once.
        embeddings = get_embeddings(self.model, self.tokenizer, sequences, self.context_length, self.batch_size)
        X = embeddings.numpy()
        
        # Classification evaluation.
        probs = class_probe.predict_proba(X)
        if len(np.unique(class_labels)) == 2:
            auc = roc_auc_score(class_labels, probs[:, 1])
        else:
            auc = roc_auc_score(class_labels, probs, multi_class='ovr')
        
        # GC regression evaluation.
        gc_predictions = gc_regressor.predict(X)
        r2 = r2_score(gc_labels, gc_predictions)
        
        return auc, r2

    def run_benchmark(self, benchmark_dir):
        """
        For a single benchmark, train both probes and evaluate them.
        
        Args:
            benchmark_dir (str): Directory with train.csv and dev.csv.
            
        Returns:
            dict: A dictionary with keys 'roc_auc' and 'gc_r2' for the respective scores.
        """
        train_csv = f"{benchmark_dir}/train.csv"
        dev_csv = f"{benchmark_dir}/{self.mode}.csv"
        
        print(f"Running probing benchmarks {benchmark_dir}...")
        class_probe, gc_regressor = self.train_probes(train_csv)
        auc, r2 = self.evaluate_probes(class_probe, gc_regressor, dev_csv)
        
        return {"roc_auc": auc, "gc_r2": r2}

    def run_all_benchmarks(self, verbose=False):
        """
        Run all benchmarks, training and evaluating probes for each benchmark.
        
        If verbose is True, prints results in table format.
        
        Returns:
            dict: Mapping each benchmark directory to its scores.
        """
        results = {}
        for benchmark in self.benchmark_dirs:
            metrics = self.run_benchmark(benchmark)
            results[benchmark] = metrics
        
        if verbose:
            header = f"{'Benchmark':<50} {'ROC-AUC':<10} {'GC R²':<10}"
            print("\n" + header)
            print("-" * (len(header) + 10))
            for bench, scores in results.items():
                print(f"{bench:<50} {scores['roc_auc']:<10.4f} {scores['gc_r2']:<10.4f}")
            print("-" * (len(header) + 10))
        return results
