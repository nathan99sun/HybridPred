import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class AutoEncoder_ElasticNet(nn.Module):
    def __init__(self, n_features, n_cycles=49, alpha=0.5, lr = 1e-3, epochs = 500, batch_size=16):
        super(AutoEncoder_ElasticNet, self).__init__()

        self.alpha = alpha
        self.n_features = n_features
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
    

    def fit(self, train_data, train_policy, verbose = True, log_loss = False):

        self.train()
        loss_function = nn.MSELoss()

        num_stages = train_policy["num_stages"]
        for stage in range(num_stages):

            if verbose: print("\nStage {}:\n".format(stage+1))

            epochs = train_policy["epochs"][stage]
            lr = train_policy["learning_rates"][stage]
            batch_size = train_policy["batch_sizes"][stage]
            prediction_weight = train_policy["prediction_weights"][stage]
            decoding_weight = train_policy["decoding_weights"][stage]
            en_weight = train_policy["decoding_weights"][stage]

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            optimiser = torch.optim.Adam(self.parameters(), lr=lr)

            for ep in range(epochs):
                for batch in train_loader:

                    train_inputs, train_labels = batch
                    outputs = self.forward(train_inputs)
                    predictions = self.elastic_net_predict(train_inputs)

                    if log_loss:
                        loss = loss_function(train_labels, predictions[:, 0])*prediction_weight 
                    else:
                        loss = loss_function(10**train_labels, 10**predictions[:, 0])*prediction_weight
                    loss += self.elastic_net_loss()*en_weight + loss_function(train_inputs, outputs)*decoding_weight

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                if verbose:    
                    if (ep+1) % int(epochs / 10) == 0: print(f"Epoch {ep+1}/{epochs}, loss: {loss.item():.2f}")

