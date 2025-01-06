class StockDataset(Dataset):
    def __init__(self, data, seq_length=60, target_column='SPY'):
        self.seq_length = seq_length
        self.data = data
        self.target_column = target_column
        self.sequences = []
        self.targets = []
        self.create_sequences()
    
    def create_sequences(self):
        for i in range(len(self.data) - self.seq_length):
            seq = self.data.iloc[i:i+self.seq_length].values
            target = self.data.iloc[i+self.seq_length][self.target_column]
            self.sequences.append(seq)
            self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]).float(), torch.tensor(self.targets[idx]).float()

# Train-Test Split
train_size = int(len(scaled_df) * 0.8)
train_data = scaled_df.iloc[:train_size]
test_data = scaled_df.iloc[train_size:]

# Create Datasets
train_dataset = StockDataset(train_data, seq_length=60, target_column='SPY')
test_dataset = StockDataset(test_data, seq_length=60, target_column='SPY')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
