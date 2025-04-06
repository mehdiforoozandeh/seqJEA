class DNADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fasta_file="/project/compbio-lab/encode_data/hg38.fa",
        blacklist_file="hg38_blacklist_v2.bed",
        tokenizer_type="DNABERT",
        dynamic=True,
        static_sample_size=10000,
        dataset_size=10000,
        min_length=200,
        max_length=512,
        k=3
    ):
        self.fasta_file = fasta_file
        self.fasta = pysam.FastaFile(fasta_file)
        self.tokenizer_type = tokenizer_type
        self.dynamic = dynamic
        self.static_sample_size = static_sample_size
        self.dataset_size = dataset_size
        self.min_length = min_length
        self.max_length = max_length
        self.k = k

        # Load blacklist regions into IntervalTrees
        self.blacklist = self.load_blacklist(blacklist_file)

        # Chromosome names and lengths
        self.chroms = self.fasta.references
        self.chrom_lengths = dict(zip(self.chroms, self.fasta.lengths))

        # Initialize tokenizer
        if self.tokenizer_type == "DNABERT":
            self.tokenizer = self.load_dnabert_tokenizer()
        else:
            self.tokenizer = self.simple_kmer_tokenizer

        # Preload sequences if not dynamic
        if not self.dynamic:
            self.preloaded_sequences = []
            while len(self.preloaded_sequences) < self.static_sample_size:
                seq = self.sample_sequence()
                if seq is not None:
                    tokens = self.tokenizer(seq)
                    self.preloaded_sequences.append(tokens)

    def load_blacklist(self, blacklist_file):
        """Load blacklist regions using IntervalTree."""
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
        """DNABERT tokenizer returning input_ids and attention_mask."""
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        def tokenize_fn(sequence):
            tokenized = tokenizer(
                sequence.upper(),
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            }
        return tokenize_fn

    def sample_sequence(self):
        """Sample a sequence using IntervalTree for blacklist check."""
        chrom = random.choices(self.chroms, weights=self.chrom_lengths.values(), k=1)[0]
        chrom_len = self.chrom_lengths[chrom]
        if chrom_len < self.min_length:
            return None
        length = random.randint(self.min_length, self.max_length)
        max_start = chrom_len - length
        if max_start <= 0:
            return None
        start = random.randint(0, max_start)
        end = start + length
        if not self.is_region_allowed(chrom, start, end):
            return None
        sequence = self.fasta.fetch(chrom, start, end).upper()
        return sequence

    def is_region_allowed(self, chrom, start, end):
        """Check overlap using IntervalTree."""
        if chrom not in self.blacklist:
            return True
        return not self.blacklist[chrom].overlaps(start, end)

    def __getitem__(self, idx):
        if self.dynamic:
            sequence = None
            attempts = 0
            while sequence is None and attempts < 10:
                sequence = self.sample_sequence()
                attempts += 1
            if sequence is None:
                raise ValueError("Failed to sample valid sequence after 10 attempts.")
            return self.tokenizer(sequence)
        else:
            return self.preloaded_sequences[idx]