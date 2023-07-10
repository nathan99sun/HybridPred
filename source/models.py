import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class AutoEncoder_ElasticNet(nn.Module):
    def __init__(self, n_features, n_cycles=49, alpha=0.5, en_lambda = 0.1, lr = 1e-3, epochs = 500, batch_size=16):
        super(AutoEncoder_ElasticNet, self).__init__()

        self.alpha = alpha
        self.n_features = n_features
        self.en_lambda = en_lambda
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.encoder = nn.Sequential(
            nn.Linear(n_features*n_cycles, n_features*32),
            nn.ReLU(),
            nn.Linear(n_features*32, n_features*16),
            # nn.ReLU(),
            # nn.Linear(n_features*16, n_features*8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # nn.Linear(n_features*8, n_features*16),
            # nn.ReLU(),
            nn.Linear(n_features*16, n_features*32),
            nn.ReLU(),
            nn.Linear(n_features*32, n_features*n_cycles)
        )

        self.prediction = nn.Linear(n_features*16, 1)

    def elastic_net_loss(self):

        l1_norm = self.prediction.weight.abs().sum()
        l2_norm = self.prediction.weight.pow(2).sum()

        return (1-self.alpha)/2 * l2_norm + self.alpha * l1_norm
    
    def elastic_net_predict(self, x):
        return self.prediction(self.encoder(x))
    
    def forward(self, x):
        self.eval()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, train_data, lr = None, epochs = None):

        # Use this to overwrite the lr and epochs defined with the model
        if lr is None: lr = self.lr
        if epochs is None: epochs = self.epochs

        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        loss_function = nn.MSELoss()
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        for ep in range(epochs):
            for batch in train_loader:
                optimiser.zero_grad()
                train_inputs, train_labels = batch

                outputs = self.forward(train_inputs)
                predictions = self.elastic_net_predict(train_inputs)
                loss = loss_function(train_labels, predictions[:, 0]) + loss_function(train_inputs, outputs)
                loss += self.elastic_net_loss()*self.en_lambda

                loss.backward()
                optimiser.step()
                
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {loss.item():.2f}")