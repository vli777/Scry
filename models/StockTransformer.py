class MultiStockTransformer(nn.Module):
    def __init__(self, num_features, num_layers=4, nhead=8, d_model=256, dim_feedforward=1024, dropout=0.1):
        super(MultiStockTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)  # Predicting SPY's next day price
    
    def forward(self, src):
        """
        src shape: [batch_size, seq_length, num_features]
        """
        src = src.permute(1, 0, 2)  # [seq_length, batch_size, num_features]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[-1, :, :]  # Take the output of the last time step
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x shape: [seq_length, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Initialize Model (example with original hidden dims)
num_features = scaled_df.shape[1]  # Number of features per time step
model = MultiStockTransformer(
    num_features=num_features,
    num_layers=4,
    nhead=8,
    d_model=128,
    dim_feedforward=512,
    dropout=0.1
).to(device)
