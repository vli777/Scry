def objective(trial):
    # Hyperparameter suggestions
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    nhead = trial.suggest_categorical('nhead', [8, 16])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [512, 1024, 2048])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    
    # Initialize model with suggested hyperparameters
    model = MultiStockTransformer(
        num_features=num_features,
        num_layers=num_layers,
        nhead=nhead,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    # Define optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop (simplified for faster trials)
    num_epochs = 10  # Use fewer epochs for faster trials
    for epoch in range(num_epochs):
        model.train()
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
    
    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(test_loader)
    
    return avg_val_loss

# Create and run study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
print("Best validation loss:", study.best_value)
