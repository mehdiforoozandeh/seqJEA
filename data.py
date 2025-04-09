import torch, pysam, random
from transformers import AutoTokenizer
from intervaltree import IntervalTree

class DNADataset(torch.utils.data.Dataset):
    def __init__(self,
        fasta_file="/project/compbio-lab/encode_data/hg38.fa",
        blacklist_file="hg38_blacklist_v2.bed",
        dynamic=True,
        static_sample_size=10000,
        dataset_size=10000,
        min_length=200,
        max_length=512, 
        context_length=500):
        """
        Initialize the DNADataset.

        Args:
            fasta_file (str): Path to the FASTA file.
            blacklist_file (str): Path to the blacklist BED file.
            dynamic (bool): If True, sample sequences on-the-fly; if False, preload sequences.
            static_sample_size (int): Number of sequences to preload in static mode.
            dataset_size (int): Total number of samples in the dataset.
            min_length (int): Minimum sequence length.
            max_length (int): Maximum sequence length (for padding/truncation).

        Notes:
            - Uses IntervalTree for efficient blacklist overlap checking.
            - Weights chromosome selection by length.
            - Always uses DNABERT tokenizer, returning input_ids and attention_mask.
        """
        self.fasta_file = fasta_file
        self.fasta = pysam.FastaFile(fasta_file)
        self.dynamic = dynamic
        self.static_sample_size = static_sample_size
        self.dataset_size = dataset_size
        self.min_length = min_length
        self.max_length = max_length
        self.context_length = context_length

        # Load blacklist regions into IntervalTrees
        self.blacklist = self.load_blacklist(blacklist_file)

        # Extract chromosome names and lengths
        self.chroms = self.fasta.references
        self.chrom_lengths = dict(zip(self.fasta.references, self.fasta.lengths))

        # Load DNABERT tokenizer
        self.tokenizer = self.load_dnabert_tokenizer()

        # Preload sequences in static mode
        if not self.dynamic:
            self.preloaded_sequences = []
            while len(self.preloaded_sequences) < self.static_sample_size:
                seq = self.sample_sequence()
                if seq is not None:
                    tokens = self.tokenizer(seq)
                    self.preloaded_sequences.append(tokens)

    def load_blacklist(self, blacklist_file):
        """Load blacklist regions from a BED file into IntervalTrees."""
        blacklist = {}
        with open(blacklist_file, 'r') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue
                if chrom not in blacklist:
                    blacklist[chrom] = IntervalTree()
                blacklist[chrom].addi(start, end)
        return blacklist

    def load_dnabert_tokenizer(self):
        """Load DNABERT-2 tokenizer, returning input_ids and attention_mask."""
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        def tokenize_fn(sequence):
            tokenized = tokenizer(sequence.upper(),  # Ensure uppercase for consistency
                                  return_tensors="pt",
                                  padding="max_length",
                                  max_length=self.context_length,
                                  truncation=True)
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            }
        return tokenize_fn

    def is_region_allowed(self, chrom, start, end):
        """Check if a region overlaps with blacklist regions using IntervalTree."""
        if chrom not in self.blacklist:
            return True
        tree = self.blacklist[chrom]
        overlapping = tree.overlap(start, end)
        return len(overlapping) == 0

    def sample_sequence(self):
        """Sample a sequence, weighting chromosomes by length."""
        chrom_lengths = [self.chrom_lengths[chrom] for chrom in self.chroms]
        chrom = random.choices(self.chroms, weights=chrom_lengths, k=1)[0]
        chrom_len = self.chrom_lengths[chrom]
        length = random.randint(self.min_length, self.max_length)
        if chrom_len < length:
            return None
        max_start = chrom_len - length
        start = random.randint(0, max_start)
        end = start + length
        if not self.is_region_allowed(chrom, start, end):
            return None
        sequence = self.fasta.fetch(chrom, start, end).upper()  # Ensure uppercase
        return sequence

    def __len__(self):
        """Return the dataset size."""
        return self.dataset_size if self.dynamic else len(self.preloaded_sequences)

    def __getitem__(self, idx):
        """Get a tokenized sequence (input_ids and attention_mask)."""
        if self.dynamic:
            sequence = None
            attempts = 0
            while sequence is None and attempts < 10:
                sequence = self.sample_sequence()

                attempts += 1
            if sequence is None:
                raise ValueError("Failed to sample a valid sequence after 10 attempts.")
            return self.tokenizer(sequence)
        else:
            return self.preloaded_sequences[idx]
