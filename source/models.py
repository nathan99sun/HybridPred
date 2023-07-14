import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_split import *
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (18,7)

class AutoEncoder_ElasticNet(nn.Module):
    def __init__(self, n_features, n_cycles=49, alpha=0.5):
        super(AutoEncoder_ElasticNet, self).__init__()

        self.alpha = alpha
        self.n_features = n_features
        self.n_cycles = n_cycles

        self.encoder = nn.Sequential(
            nn.Linear(n_features*n_cycles, n_features*32),
            nn.ReLU(),
            nn.Linear(n_features*32, n_features*16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
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
        self.eval()
        return self.prediction(self.encoder(x))
    
    def forward(self, x):
        self.eval()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def evaluate(self, x, y):
        mses = [0.0, 0.0, 0.0]
        for i in train_ind:
            mses[0] += (10**self.elastic_net_predict(x[i]).detach().numpy() - 10**y[i].detach().numpy())**2
        for i in test_ind:
            mses[1] += (10**self.elastic_net_predict(x[i]).detach().numpy() - 10**y[i].detach().numpy())**2
        for i in secondary_ind:
            mses[2] += (10**self.elastic_net_predict(x[i]).detach().numpy() - 10**y[i].detach().numpy())**2

        mses[0] = np.sqrt(mses[0] / len(train_ind))
        mses[1] = np.sqrt(mses[1] / len(test_ind))
        mses[2] = np.sqrt(mses[2] / len(secondary_ind))

        for mse, label in zip(mses, ["\ntrain", "test", "sec"]):
            print(label, "\t", mse[0])


    def plotter(self, x, y, id):
        fig, axs = plt.subplots(1, 2)
        for i in range(self.n_features):
            axs[0].plot(x[id].detach().numpy()[i*self.n_cycles:(i+1)*self.n_cycles], 
                        self.forward(x[id]).detach().numpy()[i*self.n_cycles:(i+1)*self.n_cycles], "o", label = "feature {}".format(i+1))
        axs[0].plot(np.linspace(-5.5, 1.5, 3), np.linspace(-5.5, 1.5, 3), "k", alpha = 0.5)

        axs[0].legend(fontsize = 14)
        axs[0].set_xlabel("True input", fontsize = 16)
        axs[0].set_ylabel("Decoded input", fontsize = 16)
        axs[0].tick_params(axis='x', labelsize=14)
        axs[0].tick_params(axis='y', labelsize=14)
        axs[0].set_title("Decoder performance, cell "+str(id), fontsize = 20)

        axs[1].plot(10**y[train_ind], 10**self.elastic_net_predict(x[train_ind]).detach().numpy(), "o", label = "train")
        axs[1].plot(10**y[test_ind], 10**self.elastic_net_predict(x[test_ind]).detach().numpy(), "o", label = "test")
        axs[1].plot(10**y[secondary_ind], 10**self.elastic_net_predict(x[secondary_ind]).detach().numpy(), "o", label = "secondary")
        axs[1].plot(np.linspace(200, 2400, 3), np.linspace(200, 2400, 3), "k", alpha = 0.5)

        axs[1].legend(fontsize = 14)
        axs[1].set_xlabel("True lifetime", fontsize = 16)
        axs[1].tick_params(axis='x', labelsize=14)
        axs[1].tick_params(axis='y', labelsize=14)
        axs[1].set_ylabel("Predicted lifetime", fontsize = 16)
        axs[1].set_title("Prediction performance", fontsize = 20)

        plt.show()

    def fit(self, x, y, train_policy, verbose = True, plots = True, log_loss = False):

        self.train()
        loss_function = nn.MSELoss()
        train_data = TensorDataset(torch.Tensor(x[train_ind]),torch.Tensor(y[train_ind]))

        num_stages = train_policy["num_stages"]
        for stage in range(num_stages):

            if verbose: print("Stage {}:\n".format(stage+1))

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
                        prediction_loss = loss_function(train_labels, predictions[:, 0])
                    else:
                        prediction_loss = loss_function(10**train_labels, 10**predictions[:, 0])
                    decoding_loss = loss_function(train_inputs, outputs)
                    en_loss = self.elastic_net_loss()
                    loss = en_loss*en_weight + decoding_loss*decoding_weight + prediction_loss*prediction_weight

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                if verbose:    
                    if (ep+1) % int(epochs / 10) == 0: print(f"Epoch {ep+1}/{epochs},   \tdecoding loss: {decoding_loss.item():.2f},    \tprediction loss: {prediction_loss.item():.2f},  \treg_loss: {en_loss.item():.2f}")

            if verbose: self.evaluate(x, y)
            if plots:
                self.plotter(x, y, 30)



class AutoEncoder_Individual(AutoEncoder_ElasticNet):
    def __init__(self, n_features, n_cycles=49, alpha=0.5):
        super(AutoEncoder_Individual, self).__init__(n_features=n_features, n_cycles=n_cycles)

        self.encoders = nn.ModuleList([nn.Sequential(
            nn.Linear(n_cycles, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )]*n_features)

        self.decoders = nn.ModuleList([nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, n_cycles)
        )]*n_features)

        self.prediction = nn.Linear(n_features*16, 1)

    def forward(self, x):
        self.eval()
        if len(x.size()) == 1: x = x.expand(1, len(x))

        x_features = []
        for i in range(self.n_features):
            x_features.append(x[:, i*self.n_cycles:(i+1)*self.n_cycles])

        decoded_list = []
        for i, encoder in enumerate(self.encoders):
            z = encoder(x_features[i])
            decoded = self.decoders[i](z)
            decoded_list.append(decoded)

        decoded_total = torch.cat(decoded_list, dim=1)

        if decoded_total.size()[0] == 1: return decoded_total[0]
        else: return decoded_total
 

    def elastic_net_predict(self, x):
        self.eval()
        if len(x.size()) == 1: x = x.expand(1, len(x))
        x_features = []
        for i in range(self.n_features):
            x_features.append(x[:, i*self.n_cycles:(i+1)*self.n_cycles])
        z_list = []
        for i, encoder in enumerate(self.encoders):
            z = encoder(x_features[i])
            z_list.append(z)  
        z_total = torch.cat(z_list, dim=1)
        return self.prediction(z_total)